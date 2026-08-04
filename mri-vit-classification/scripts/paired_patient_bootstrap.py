"""Compare CNN and ViT on the same patients with paired stratified bootstrap.

The input CSV files must contain one row per patient and class probabilities in
``prob_class0``, ``prob_class1``, ... columns.  The script resamples patient
indices within each true class and always applies the same sampled indices to
both models, preserving the paired design.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CNN_CSV = PROJECT_ROOT / (
    "outputs/repro_cnn_all_axial_patient_split_3class_resnet18_224_reg/metrics/"
    "resnet18_eval_val_patient_top_k_confidence_k5_best_loss_patients.csv"
)
DEFAULT_VIT_CSV = PROJECT_ROOT / (
    "outputs/repro_vit_all_axial_patient_split_3class_deit_small_224_reg/metrics/"
    "vit_eval_val_patient_top_k_confidence_k5_best_loss_patients.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/fair_comparison_all_axial_top5_paired_bootstrap"

METRIC_NAMES = (
    "accuracy",
    "macro_f1",
    "macro_roc_auc",
    "balanced_accuracy",
    "grade0_recall",
    "grade1_recall",
    "grade2plus_recall",
)
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "macro_roc_auc": "Macro ROC-AUC",
    "balanced_accuracy": "Balanced Accuracy",
    "grade0_recall": "Grade 0 Recall",
    "grade1_recall": "Grade 1 Recall",
    "grade2plus_recall": "Grade 2+ Recall",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-csv", type=Path, default=DEFAULT_CNN_CSV)
    parser.add_argument("--vit-csv", type=Path, default=DEFAULT_VIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--analysis-label", default="All-axial top-5: paired patient bootstrap")
    parser.add_argument("--data-scope", default="FL+T1 all axial, patient split, 3 classes")
    parser.add_argument(
        "--limitation",
        default=(
            "The validation cohort was used repeatedly during development. Confidence intervals quantify "
            "patient-sampling uncertainty but do not remove model/pooling selection bias and are not a "
            "substitute for patient-level cross-validation or external validation."
        ),
    )
    parser.add_argument("--cv-folds", type=int, default=None)
    return parser.parse_args()


def _probability_columns(fieldnames: list[str]) -> list[str]:
    columns = [name for name in fieldnames if re.fullmatch(r"prob_class\d+", name)]
    return sorted(columns, key=lambda name: int(name.removeprefix("prob_class")))


def load_patient_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        prob_columns = _probability_columns(reader.fieldnames)
        if not prob_columns:
            raise ValueError(
                f"Class probability columns are missing from {path}. "
                "Re-run src.evaluate after enabling prob_class* CSV output."
            )

        records: dict[str, dict[str, Any]] = {}
        for row in reader:
            patient_id = row["patient_id"]
            if patient_id in records:
                raise ValueError(f"Duplicate patient_id {patient_id} in {path}")
            probs = np.asarray([float(row[name]) for name in prob_columns], dtype=np.float64)
            if not np.isclose(probs.sum(), 1.0, atol=1e-4):
                raise ValueError(f"Probabilities do not sum to one for patient {patient_id} in {path}")
            records[patient_id] = {
                "target": int(row["target"]),
                "pred": int(row["pred"]),
                "probs": probs,
            }
    return records


def align_models(
    cnn_records: dict[str, dict[str, Any]],
    vit_records: dict[str, dict[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cnn_ids = set(cnn_records)
    vit_ids = set(vit_records)
    if cnn_ids != vit_ids:
        raise ValueError(
            "Patient sets differ: "
            f"CNN-only={len(cnn_ids - vit_ids)}, ViT-only={len(vit_ids - cnn_ids)}"
        )

    patient_ids = sorted(cnn_ids)
    targets = np.asarray([cnn_records[pid]["target"] for pid in patient_ids], dtype=np.int64)
    vit_targets = np.asarray([vit_records[pid]["target"] for pid in patient_ids], dtype=np.int64)
    if not np.array_equal(targets, vit_targets):
        mismatches = int(np.count_nonzero(targets != vit_targets))
        raise ValueError(f"Target labels differ for {mismatches} patients")

    cnn_preds = np.asarray([cnn_records[pid]["pred"] for pid in patient_ids], dtype=np.int64)
    vit_preds = np.asarray([vit_records[pid]["pred"] for pid in patient_ids], dtype=np.int64)
    cnn_probs = np.stack([cnn_records[pid]["probs"] for pid in patient_ids])
    vit_probs = np.stack([vit_records[pid]["probs"] for pid in patient_ids])
    if cnn_probs.shape != vit_probs.shape:
        raise ValueError(f"Probability shapes differ: CNN={cnn_probs.shape}, ViT={vit_probs.shape}")
    return patient_ids, targets, cnn_preds, cnn_probs, vit_preds, vit_probs


def compute_metrics(targets: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    labels = np.arange(probs.shape[1])
    recalls = recall_score(targets, preds, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, labels=labels, average="macro", zero_division=0)),
        "macro_roc_auc": float(roc_auc_score(targets, probs, labels=labels, multi_class="ovr", average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "grade0_recall": float(recalls[0]),
        "grade1_recall": float(recalls[1]),
        "grade2plus_recall": float(recalls[2]),
    }


def percentile_interval(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def exact_mcnemar_p(vit_only_correct: int, cnn_only_correct: int) -> float:
    discordant = vit_only_correct + cnn_only_correct
    if discordant == 0:
        return 1.0
    lower_tail = sum(math.comb(discordant, i) for i in range(min(vit_only_correct, cnn_only_correct) + 1))
    return min(1.0, 2.0 * lower_tail / (2.0**discordant))


def bootstrap(
    targets: np.ndarray,
    cnn_preds: np.ndarray,
    cnn_probs: np.ndarray,
    vit_preds: np.ndarray,
    vit_probs: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    if n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")

    point_cnn = compute_metrics(targets, cnn_preds, cnn_probs)
    point_vit = compute_metrics(targets, vit_preds, vit_probs)
    class_indices = [np.flatnonzero(targets == class_id) for class_id in np.unique(targets)]
    rng = np.random.default_rng(seed)

    distributions = {
        metric: {
            "cnn": np.empty(n_bootstrap, dtype=np.float64),
            "vit": np.empty(n_bootstrap, dtype=np.float64),
            "difference_vit_minus_cnn": np.empty(n_bootstrap, dtype=np.float64),
        }
        for metric in METRIC_NAMES
    }

    for iteration in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        rng.shuffle(sampled)
        sampled_targets = targets[sampled]
        cnn_values = compute_metrics(sampled_targets, cnn_preds[sampled], cnn_probs[sampled])
        vit_values = compute_metrics(sampled_targets, vit_preds[sampled], vit_probs[sampled])
        for metric in METRIC_NAMES:
            distributions[metric]["cnn"][iteration] = cnn_values[metric]
            distributions[metric]["vit"][iteration] = vit_values[metric]
            distributions[metric]["difference_vit_minus_cnn"][iteration] = (
                vit_values[metric] - cnn_values[metric]
            )

    metric_results: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        difference = point_vit[metric] - point_cnn[metric]
        difference_ci = percentile_interval(distributions[metric]["difference_vit_minus_cnn"])
        metric_results[metric] = {
            "cnn": {
                "estimate": point_cnn[metric],
                "ci_95": percentile_interval(distributions[metric]["cnn"]),
            },
            "vit": {
                "estimate": point_vit[metric],
                "ci_95": percentile_interval(distributions[metric]["vit"]),
            },
            "difference_vit_minus_cnn": {
                "estimate": difference,
                "ci_95": difference_ci,
                "ci_excludes_zero": bool(difference_ci[0] > 0.0 or difference_ci[1] < 0.0),
            },
        }
    return metric_results, distributions


def save_summary_csv(metric_results: dict[str, Any], output_path: Path) -> None:
    fields = [
        "metric",
        "cnn_estimate",
        "cnn_ci_low",
        "cnn_ci_high",
        "vit_estimate",
        "vit_ci_low",
        "vit_ci_high",
        "difference_vit_minus_cnn",
        "difference_ci_low",
        "difference_ci_high",
        "difference_ci_excludes_zero",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for metric in METRIC_NAMES:
            result = metric_results[metric]
            writer.writerow(
                {
                    "metric": metric,
                    "cnn_estimate": result["cnn"]["estimate"],
                    "cnn_ci_low": result["cnn"]["ci_95"][0],
                    "cnn_ci_high": result["cnn"]["ci_95"][1],
                    "vit_estimate": result["vit"]["estimate"],
                    "vit_ci_low": result["vit"]["ci_95"][0],
                    "vit_ci_high": result["vit"]["ci_95"][1],
                    "difference_vit_minus_cnn": result["difference_vit_minus_cnn"]["estimate"],
                    "difference_ci_low": result["difference_vit_minus_cnn"]["ci_95"][0],
                    "difference_ci_high": result["difference_vit_minus_cnn"]["ci_95"][1],
                    "difference_ci_excludes_zero": result["difference_vit_minus_cnn"]["ci_excludes_zero"],
                }
            )


def save_markdown_report(results: dict[str, Any], output_path: Path) -> None:
    analysis = results["analysis"]
    lines = [
        f"# {analysis['analysis_label']}",
        "",
        f"- Patients: {analysis['num_patients']}",
        f"- Bootstrap resamples: {analysis['n_bootstrap']:,}",
        f"- Seed: {analysis['seed']}",
        "- Method: class-stratified paired patient bootstrap (percentile 95% CI)",
        "- Primary comparison: ViT minus CNN; both use all axial slices, best-loss checkpoint, top-5 pooling",
    ]
    if analysis.get("cv_folds") is not None:
        lines.append(f"- Evaluation: {analysis['cv_folds']}-fold patient-level out-of-fold predictions")
    lines.extend(
        [
            "",
            "| Metric | CNN estimate (95% CI) | ViT estimate (95% CI) | ViT − CNN (95% CI) |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in METRIC_NAMES:
        result = results["metrics"][metric]
        cnn = result["cnn"]
        vit = result["vit"]
        diff = result["difference_vit_minus_cnn"]
        lines.append(
            f"| {METRIC_LABELS[metric]} | "
            f"{cnn['estimate']:.4f} ({cnn['ci_95'][0]:.4f}, {cnn['ci_95'][1]:.4f}) | "
            f"{vit['estimate']:.4f} ({vit['ci_95'][0]:.4f}, {vit['ci_95'][1]:.4f}) | "
            f"{diff['estimate']:+.4f} ({diff['ci_95'][0]:+.4f}, {diff['ci_95'][1]:+.4f}) |"
        )

    mcnemar = results["mcnemar_accuracy"]
    lines.extend(
        [
            "",
            "## Paired accuracy check",
            "",
            f"- ViT only correct: {mcnemar['vit_only_correct']}",
            f"- CNN only correct: {mcnemar['cnn_only_correct']}",
            f"- Both correct: {mcnemar['both_correct']}",
            f"- Both wrong: {mcnemar['both_wrong']}",
            f"- Exact McNemar p-value: {mcnemar['exact_two_sided_p']:.6g}",
            "",
            "## Interpretation limitation",
            "",
            analysis["limitation"],
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_difference_figure(
    distributions: dict[str, dict[str, np.ndarray]],
    metric_results: dict[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    flat_axes = axes.ravel()
    for axis, metric in zip(flat_axes, METRIC_NAMES):
        values = distributions[metric]["difference_vit_minus_cnn"]
        result = metric_results[metric]["difference_vit_minus_cnn"]
        axis.hist(values, bins=45, color="#2E74B5", alpha=0.82)
        axis.axvline(0.0, color="#9C0006", linestyle="--", linewidth=1.2)
        axis.axvline(result["estimate"], color="#1F3B63", linewidth=1.5)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("ViT − CNN")
        axis.set_ylabel("Bootstrap count")
        axis.grid(alpha=0.18)
    for axis in flat_axes[len(METRIC_NAMES):]:
        axis.axis("off")
    fig.suptitle("All-axial top-5 paired bootstrap differences", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cnn_records = load_patient_predictions(args.cnn_csv)
    vit_records = load_patient_predictions(args.vit_csv)
    patient_ids, targets, cnn_preds, cnn_probs, vit_preds, vit_probs = align_models(cnn_records, vit_records)

    metric_results, distributions = bootstrap(
        targets=targets,
        cnn_preds=cnn_preds,
        cnn_probs=cnn_probs,
        vit_preds=vit_preds,
        vit_probs=vit_probs,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    cnn_correct = cnn_preds == targets
    vit_correct = vit_preds == targets
    vit_only_correct = int(np.count_nonzero(vit_correct & ~cnn_correct))
    cnn_only_correct = int(np.count_nonzero(cnn_correct & ~vit_correct))
    mcnemar = {
        "both_correct": int(np.count_nonzero(cnn_correct & vit_correct)),
        "both_wrong": int(np.count_nonzero(~cnn_correct & ~vit_correct)),
        "vit_only_correct": vit_only_correct,
        "cnn_only_correct": cnn_only_correct,
        "exact_two_sided_p": exact_mcnemar_p(vit_only_correct, cnn_only_correct),
    }

    class_counts = {str(class_id): int(np.count_nonzero(targets == class_id)) for class_id in np.unique(targets)}
    results = {
        "analysis": {
            "analysis_label": args.analysis_label,
            "design": "paired class-stratified patient bootstrap",
            "confidence_interval": "percentile 95%",
            "num_patients": len(patient_ids),
            "class_counts": class_counts,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "primary_pooling": "top_k_confidence",
            "top_k": 5,
            "checkpoint_metric": "loss",
            "data_scope": args.data_scope,
            "cv_folds": args.cv_folds,
            "cnn_csv": str(args.cnn_csv),
            "vit_csv": str(args.vit_csv),
            "limitation": args.limitation,
        },
        "metrics": metric_results,
        "mcnemar_accuracy": mcnemar,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_bootstrap_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, allow_nan=False)
    save_summary_csv(metric_results, args.output_dir / "paired_bootstrap_summary.csv")
    save_markdown_report(results, args.output_dir / "paired_bootstrap_report.md")
    save_difference_figure(
        distributions,
        metric_results,
        args.output_dir / "paired_bootstrap_differences.png",
    )

    print(f"Saved paired bootstrap results to: {args.output_dir}")
    for metric in ("accuracy", "macro_f1", "macro_roc_auc", "grade2plus_recall"):
        result = metric_results[metric]["difference_vit_minus_cnn"]
        print(
            f"{metric}: {result['estimate']:+.4f} "
            f"(95% CI {result['ci_95'][0]:+.4f}, {result['ci_95'][1]:+.4f})"
        )
    print(f"Exact McNemar p-value: {mcnemar['exact_two_sided_p']:.6g}")


if __name__ == "__main__":
    main()
