"""Exploratory fixed ensemble, selective prediction, and ordinal OOF analysis.

This is a secondary analysis of frozen patient-level 5-fold OOF probabilities.
It performs no retraining, threshold tuning, or model-weight optimization. The
ensemble is fixed in advance at a 50:50 arithmetic mean of CNN and ViT class
probabilities. Results must not replace the prespecified primary comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OOF_ROOT = PROJECT_ROOT / "outputs/cv5_all_axial_3class/oof_summary"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/cv5_all_axial_3class/oof_ensemble_selective_analysis"
DEFAULT_CNN = OOF_ROOT / "resnet18_oof_patients.csv"
DEFAULT_VIT = OOF_ROOT / "vit_oof_patients.csv"
CLASS_LABELS = ["grade0", "grade1", "grade2+"]
MODEL_LABELS = {"cnn": "ResNet18", "vit": "DeiT-small", "ensemble": "50:50 ensemble"}
MODEL_COLORS = {"cnn": "#C55A11", "vit": "#2E74B5", "ensemble": "#70AD47"}
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-csv", type=Path, default=DEFAULT_CNN)
    parser.add_argument("--vit-csv", type=Path, default=DEFAULT_VIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--ece-bins", type=int, default=10)
    return parser.parse_args()


def normalize_patient_id(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("Empty patient ID")
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return value


def load_oof(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            patient_id = normalize_patient_id(raw["patient_id"])
            if patient_id in rows:
                raise ValueError(f"Duplicate patient in {path}: {patient_id}")
            probs = np.asarray(
                [float(raw["prob_class0"]), float(raw["prob_class1"]), float(raw["prob_class2"])],
                dtype=np.float64,
            )
            if not np.isclose(probs.sum(), 1.0, atol=1e-4):
                raise ValueError(f"Probabilities do not sum to one: {path}, patient={patient_id}")
            rows[patient_id] = {
                "fold": int(raw["cv_fold"]),
                "target": int(raw["target"]),
                "pred": int(raw["pred"]),
                "probs": probs,
            }
    return rows


def align_predictions(
    cnn: dict[str, dict[str, Any]],
    vit: dict[str, dict[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if set(cnn) != set(vit):
        raise ValueError(
            f"Patient sets differ: CNN-only={len(set(cnn) - set(vit))}, "
            f"ViT-only={len(set(vit) - set(cnn))}"
        )
    patient_ids = sorted(cnn)
    folds = np.asarray([cnn[patient_id]["fold"] for patient_id in patient_ids], dtype=np.int64)
    vit_folds = np.asarray([vit[patient_id]["fold"] for patient_id in patient_ids], dtype=np.int64)
    targets = np.asarray([cnn[patient_id]["target"] for patient_id in patient_ids], dtype=np.int64)
    vit_targets = np.asarray([vit[patient_id]["target"] for patient_id in patient_ids], dtype=np.int64)
    if not np.array_equal(folds, vit_folds):
        raise ValueError("CNN and ViT fold assignments differ")
    if not np.array_equal(targets, vit_targets):
        raise ValueError("CNN and ViT targets differ")
    if len(patient_ids) != 1154 or set(folds.tolist()) != {1, 2, 3, 4, 5}:
        raise ValueError(f"Expected 1,154 patients and five folds, got {len(patient_ids)} and {set(folds)}")
    cnn_probs = np.stack([cnn[patient_id]["probs"] for patient_id in patient_ids])
    vit_probs = np.stack([vit[patient_id]["probs"] for patient_id in patient_ids])
    return patient_ids, folds, targets, cnn_probs, vit_probs


def binary_reliability_bins(targets: np.ndarray, probabilities: np.ndarray, n_bins: int) -> list[dict[str, float | int]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (probabilities < upper)
        if index == n_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        count = int(np.count_nonzero(mask))
        if count:
            rows.append(
                {
                    "count": count,
                    "mean_confidence": float(np.mean(probabilities[mask])),
                    "observed_frequency": float(np.mean(targets[mask])),
                }
            )
    return rows


def top_label_ece(targets: np.ndarray, probs: np.ndarray, n_bins: int) -> float:
    preds = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = (preds == targets).astype(np.float64)
    bins = binary_reliability_bins(correct, confidence, n_bins)
    return float(
        sum(
            int(row["count"]) / len(targets)
            * abs(float(row["observed_frequency"]) - float(row["mean_confidence"]))
            for row in bins
        )
    )


def multiclass_nll(targets: np.ndarray, probs: np.ndarray) -> float:
    selected = np.clip(probs[np.arange(len(targets)), targets], 1e-12, 1.0)
    return float(-np.mean(np.log(selected)))


def multiclass_brier(targets: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[targets]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def balanced_accuracy_present_classes(targets: np.ndarray, preds: np.ndarray) -> float:
    present_classes = np.unique(targets)
    return float(
        np.mean([np.mean(preds[targets == class_id] == class_id) for class_id in present_classes])
    )


def metric_bundle(targets: np.ndarray, probs: np.ndarray, ece_bins: int) -> dict[str, float]:
    preds = np.argmax(probs, axis=1)
    absolute_errors = np.abs(preds - targets)
    metrics: dict[str, float] = {
        "accuracy": float(np.mean(preds == targets)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "macro_auc": float(roc_auc_score(targets, probs, multi_class="ovr", average="macro")),
        "balanced_accuracy": balanced_accuracy_present_classes(targets, preds),
        "quadratic_weighted_kappa": float(cohen_kappa_score(targets, preds, weights="quadratic")),
        "ordinal_mae": float(np.mean(absolute_errors)),
        "severe_error_rate": float(np.mean(absolute_errors == 2)),
        "adjacent_error_rate": float(np.mean(absolute_errors == 1)),
        "nll": multiclass_nll(targets, probs),
        "multiclass_brier": multiclass_brier(targets, probs),
        "top_label_ece": top_label_ece(targets, probs, ece_bins),
    }
    recalls = recall_score(targets, preds, labels=[0, 1, 2], average=None, zero_division=0)
    for class_id in range(3):
        binary = (targets == class_id).astype(np.int64)
        metrics[f"recall_class{class_id}"] = float(recalls[class_id])
        metrics[f"roc_auc_class{class_id}"] = float(roc_auc_score(binary, probs[:, class_id]))
        metrics[f"pr_auc_ap_class{class_id}"] = float(
            average_precision_score(binary, probs[:, class_id])
        )
    return metrics


def metric_direction(metric: str) -> str:
    lower_is_better = metric in {
        "ordinal_mae",
        "severe_error_rate",
        "adjacent_error_rate",
        "nll",
        "multiclass_brier",
        "top_label_ece",
    }
    return "lower_is_better" if lower_is_better else "higher_is_better"


def percentile_ci(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def paired_bootstrap(
    targets: np.ndarray,
    probabilities: dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
    ece_bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    point = {model: metric_bundle(targets, probs, ece_bins) for model, probs in probabilities.items()}
    metrics = list(point["cnn"])
    model_distributions = {
        model: {metric: np.empty(n_bootstrap, dtype=np.float64) for metric in metrics}
        for model in probabilities
    }
    comparison_distributions = {
        comparator: {metric: np.empty(n_bootstrap, dtype=np.float64) for metric in metrics}
        for comparator in ("cnn", "vit")
    }
    class_indices = [np.flatnonzero(targets == class_id) for class_id in range(3)]
    rng = np.random.default_rng(seed)
    for iteration in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        rng.shuffle(sampled)
        sampled_targets = targets[sampled]
        sampled_metrics = {
            model: metric_bundle(sampled_targets, probs[sampled], ece_bins)
            for model, probs in probabilities.items()
        }
        for model in probabilities:
            for metric in metrics:
                model_distributions[model][metric][iteration] = sampled_metrics[model][metric]
        for comparator in ("cnn", "vit"):
            for metric in metrics:
                comparison_distributions[comparator][metric][iteration] = (
                    sampled_metrics["ensemble"][metric] - sampled_metrics[comparator][metric]
                )

    model_rows: list[dict[str, Any]] = []
    for model in ("cnn", "vit", "ensemble"):
        for metric in metrics:
            model_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "metric": metric,
                    "direction": metric_direction(metric),
                    "estimate": point[model][metric],
                    "bootstrap_ci_95": percentile_ci(model_distributions[model][metric]),
                }
            )

    comparison_rows: list[dict[str, Any]] = []
    for comparator in ("cnn", "vit"):
        for metric in metrics:
            difference = point["ensemble"][metric] - point[comparator][metric]
            ci = percentile_ci(comparison_distributions[comparator][metric])
            direction = metric_direction(metric)
            if ci[0] > 0:
                favored = "comparator" if direction == "lower_is_better" else "ensemble"
            elif ci[1] < 0:
                favored = "ensemble" if direction == "lower_is_better" else "comparator"
            else:
                favored = "inconclusive"
            comparison_rows.append(
                {
                    "comparison": f"ensemble_minus_{comparator}",
                    "comparator_label": MODEL_LABELS[comparator],
                    "metric": metric,
                    "direction": direction,
                    "ensemble_estimate": point["ensemble"][metric],
                    "comparator_estimate": point[comparator][metric],
                    "difference": difference,
                    "difference_ci_95": ci,
                    "difference_ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
                    "favored": favored,
                }
            )
    return model_rows, comparison_rows


def mcnemar_rows(targets: np.ndarray, probabilities: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    ensemble_correct = np.argmax(probabilities["ensemble"], axis=1) == targets
    rows: list[dict[str, Any]] = []
    for comparator in ("cnn", "vit"):
        comparator_correct = np.argmax(probabilities[comparator], axis=1) == targets
        ensemble_only = int(np.count_nonzero(ensemble_correct & ~comparator_correct))
        comparator_only = int(np.count_nonzero(~ensemble_correct & comparator_correct))
        discordant = ensemble_only + comparator_only
        p_value = float(binomtest(ensemble_only, discordant, 0.5).pvalue) if discordant else 1.0
        rows.append(
            {
                "comparison": f"ensemble_vs_{comparator}",
                "ensemble_only_correct": ensemble_only,
                "comparator_only_correct": comparator_only,
                "discordant_pairs": discordant,
                "exact_mcnemar_p": p_value,
            }
        )
    return rows


def safe_subset_metrics(targets: np.ndarray, probs: np.ndarray) -> dict[str, float | None]:
    if len(targets) == 0:
        return {"accuracy": None, "macro_f1": None, "balanced_accuracy": None}
    preds = np.argmax(probs, axis=1)
    return {
        "accuracy": float(np.mean(preds == targets)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "balanced_accuracy": balanced_accuracy_present_classes(targets, preds),
    }


def agreement_analysis(targets: np.ndarray, probabilities: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    cnn_preds = np.argmax(probabilities["cnn"], axis=1)
    vit_preds = np.argmax(probabilities["vit"], axis=1)
    agree = cnn_preds == vit_preds
    groups: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(len(targets), dtype=bool)),
        ("cnn_vit_agree", agree),
        ("cnn_vit_disagree", ~agree),
    ]
    for class_id, label in enumerate(CLASS_LABELS):
        groups.append((f"agree_on_{label}", agree & (cnn_preds == class_id)))
    for cnn_class in range(3):
        for vit_class in range(3):
            if cnn_class != vit_class:
                mask = (cnn_preds == cnn_class) & (vit_preds == vit_class)
                if np.any(mask):
                    groups.append(
                        (f"cnn_{CLASS_LABELS[cnn_class]}_vit_{CLASS_LABELS[vit_class]}", mask)
                    )

    output: list[dict[str, Any]] = []
    for group, mask in groups:
        count = int(np.count_nonzero(mask))
        row: dict[str, Any] = {
            "group": group,
            "num_patients": count,
            "fraction": count / len(targets),
            "true_grade0": int(np.count_nonzero(mask & (targets == 0))),
            "true_grade1": int(np.count_nonzero(mask & (targets == 1))),
            "true_grade2plus": int(np.count_nonzero(mask & (targets == 2))),
        }
        for model, probs in probabilities.items():
            subset_metrics = safe_subset_metrics(targets[mask], probs[mask])
            row[f"{model}_accuracy"] = subset_metrics["accuracy"]
            row[f"{model}_macro_f1"] = subset_metrics["macro_f1"]
            row[f"{model}_balanced_accuracy"] = subset_metrics["balanced_accuracy"]
            row[f"{model}_mean_confidence"] = float(np.mean(np.max(probs[mask], axis=1))) if count else None
        output.append(row)
    return output


def selective_prediction_analysis(
    targets: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for model, probs in probabilities.items():
        preds = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        order = np.argsort(-confidence, kind="stable")
        for coverage in COVERAGES:
            count = max(1, int(round(len(targets) * coverage)))
            selected = order[:count]
            selected_targets = targets[selected]
            selected_preds = preds[selected]
            recalls = recall_score(
                selected_targets,
                selected_preds,
                labels=[0, 1, 2],
                average=None,
                zero_division=0,
            )
            summary_rows.append(
                {
                    "model": model,
                    "coverage_requested": coverage,
                    "num_patients": count,
                    "coverage_actual": count / len(targets),
                    "confidence_threshold": float(confidence[selected[-1]]),
                    "accuracy": float(np.mean(selected_preds == selected_targets)),
                    "risk": float(np.mean(selected_preds != selected_targets)),
                    "macro_f1": float(
                        f1_score(selected_targets, selected_preds, average="macro", zero_division=0)
                    ),
                    "balanced_accuracy": balanced_accuracy_present_classes(
                        selected_targets, selected_preds
                    ),
                    "recall_grade0": float(recalls[0]),
                    "recall_grade1": float(recalls[1]),
                    "recall_grade2plus": float(recalls[2]),
                    "selected_true_grade0": int(np.count_nonzero(selected_targets == 0)),
                    "selected_true_grade1": int(np.count_nonzero(selected_targets == 1)),
                    "selected_true_grade2plus": int(np.count_nonzero(selected_targets == 2)),
                }
            )
        for count in range(1, len(targets) + 1):
            selected = order[:count]
            curve_rows.append(
                {
                    "model": model,
                    "num_patients": count,
                    "coverage": count / len(targets),
                    "risk": float(np.mean(preds[selected] != targets[selected])),
                    "accuracy": float(np.mean(preds[selected] == targets[selected])),
                    "confidence_threshold": float(confidence[selected[-1]]),
                }
            )
    return summary_rows, curve_rows


def ordinal_error_rows(targets: np.ndarray, probabilities: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, probs in probabilities.items():
        preds = np.argmax(probs, axis=1)
        absolute_errors = np.abs(preds - targets)
        for true_class in range(3):
            mask = targets == true_class
            rows.append(
                {
                    "model": model,
                    "subgroup": CLASS_LABELS[true_class],
                    "num_patients": int(np.count_nonzero(mask)),
                    "exact_count": int(np.count_nonzero(mask & (absolute_errors == 0))),
                    "adjacent_error_count": int(np.count_nonzero(mask & (absolute_errors == 1))),
                    "severe_error_count": int(np.count_nonzero(mask & (absolute_errors == 2))),
                    "exact_rate": float(np.mean(absolute_errors[mask] == 0)),
                    "adjacent_error_rate": float(np.mean(absolute_errors[mask] == 1)),
                    "severe_error_rate": float(np.mean(absolute_errors[mask] == 2)),
                    "ordinal_mae": float(np.mean(absolute_errors[mask])),
                }
            )
        rows.append(
            {
                "model": model,
                "subgroup": "all",
                "num_patients": len(targets),
                "exact_count": int(np.count_nonzero(absolute_errors == 0)),
                "adjacent_error_count": int(np.count_nonzero(absolute_errors == 1)),
                "severe_error_count": int(np.count_nonzero(absolute_errors == 2)),
                "exact_rate": float(np.mean(absolute_errors == 0)),
                "adjacent_error_rate": float(np.mean(absolute_errors == 1)),
                "severe_error_rate": float(np.mean(absolute_errors == 2)),
                "ordinal_mae": float(np.mean(absolute_errors)),
            }
        )
    return rows


def save_patient_predictions(
    path: Path,
    patient_ids: list[str],
    folds: np.ndarray,
    targets: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "patient_id", "cv_fold", "target",
            "cnn_pred", "cnn_confidence",
            "vit_pred", "vit_confidence",
            "ensemble_pred", "ensemble_confidence",
            "ensemble_prob_class0", "ensemble_prob_class1", "ensemble_prob_class2",
            "cnn_vit_agree", "ensemble_correct",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        predictions = {model: np.argmax(probs, axis=1) for model, probs in probabilities.items()}
        confidences = {model: np.max(probs, axis=1) for model, probs in probabilities.items()}
        for index, patient_id in enumerate(patient_ids):
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "cv_fold": int(folds[index]),
                    "target": int(targets[index]),
                    "cnn_pred": int(predictions["cnn"][index]),
                    "cnn_confidence": confidences["cnn"][index],
                    "vit_pred": int(predictions["vit"][index]),
                    "vit_confidence": confidences["vit"][index],
                    "ensemble_pred": int(predictions["ensemble"][index]),
                    "ensemble_confidence": confidences["ensemble"][index],
                    "ensemble_prob_class0": probabilities["ensemble"][index, 0],
                    "ensemble_prob_class1": probabilities["ensemble"][index, 1],
                    "ensemble_prob_class2": probabilities["ensemble"][index, 2],
                    "cnn_vit_agree": bool(predictions["cnn"][index] == predictions["vit"][index]),
                    "ensemble_correct": bool(predictions["ensemble"][index] == targets[index]),
                }
            )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized_row: dict[str, Any] = {}
        for key, value in row.items():
            normalized_row[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
        normalized.append(normalized_row)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(normalized[0].keys()))
        writer.writeheader()
        writer.writerows(normalized)


def metric_lookup(model_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["model"], row["metric"]): row for row in model_rows}


def comparison_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["comparison"], row["metric"]): row for row in rows}


def plot_metric_comparison(model_rows: list[dict[str, Any]], output_dir: Path) -> None:
    metrics = ["accuracy", "macro_f1", "macro_auc", "balanced_accuracy"]
    lookup = metric_lookup(model_rows)
    x = np.arange(len(metrics))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9.5, 5.3))
    for position, model in enumerate(("cnn", "vit", "ensemble")):
        values = [lookup[(model, metric)]["estimate"] for metric in metrics]
        cis = [lookup[(model, metric)]["bootstrap_ci_95"] for metric in metrics]
        lower = [value - ci[0] for value, ci in zip(values, cis)]
        upper = [ci[1] - value for value, ci in zip(values, cis)]
        axis.bar(
            x + (position - 1) * width,
            values,
            width,
            yerr=np.asarray([lower, upper]),
            capsize=3,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    axis.set_xticks(x, ["Accuracy", "Macro-F1", "Macro-AUC", "Balanced Acc"])
    axis.set_ylim(0.5, 0.86)
    axis.set_ylabel("Metric value")
    axis.set_title("Frozen 5-fold OOF model comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "ensemble_metric_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage(curve_rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 5.5))
    for model in ("cnn", "vit", "ensemble"):
        subset = [row for row in curve_rows if row["model"] == model]
        axis.plot(
            [row["coverage"] for row in subset],
            [row["risk"] for row in subset],
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Coverage (fraction automatically classified)")
    axis.set_ylabel("Risk (error rate among retained patients)")
    axis.set_title("Descriptive risk–coverage curves")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "risk_coverage_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_agreement(agreement_rows: list[dict[str, Any]], output_dir: Path) -> None:
    selected_groups = ["all", "cnn_vit_agree", "cnn_vit_disagree"]
    lookup = {row["group"]: row for row in agreement_rows}
    x = np.arange(len(selected_groups))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.5, 5.3))
    for position, model in enumerate(("cnn", "vit", "ensemble")):
        values = [lookup[group][f"{model}_accuracy"] for group in selected_groups]
        bars = axis.bar(
            x + (position - 1) * width,
            values,
            width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        axis.bar_label(bars, fmt="%.3f", fontsize=8)
    labels = [
        f"All\n(n={lookup['all']['num_patients']})",
        f"Agree\n(n={lookup['cnn_vit_agree']['num_patients']})",
        f"Disagree\n(n={lookup['cnn_vit_disagree']['num_patients']})",
    ]
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy by CNN–ViT prediction agreement")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_agreement_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_confusion(targets: np.ndarray, ensemble_probs: np.ndarray, output_dir: Path) -> None:
    preds = np.argmax(ensemble_probs, axis=1)
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2])
    normalized = cm / cm.sum(axis=1, keepdims=True)
    fig, axis = plt.subplots(figsize=(5.8, 5.1))
    image = axis.imshow(normalized, cmap="Greens", vmin=0, vmax=1)
    for row in range(3):
        for col in range(3):
            axis.text(
                col,
                row,
                f"{cm[row, col]}\n{normalized[row, col]:.1%}",
                ha="center",
                va="center",
                color="white" if normalized[row, col] > 0.55 else "black",
            )
    axis.set_xticks(range(3), CLASS_LABELS)
    axis.set_yticks(range(3), CLASS_LABELS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("50:50 ensemble OOF confusion matrix")
    fig.colorbar(image, ax=axis, label="Row-normalized proportion")
    fig.tight_layout()
    fig.savefig(output_dir / "ensemble_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def format_estimate_ci(row: dict[str, Any]) -> str:
    ci = row["bootstrap_ci_95"]
    return f"{row['estimate']:.4f} ({ci[0]:.4f}, {ci[1]:.4f})"


def write_report(
    path: Path,
    model_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    mcnemar: list[dict[str, Any]],
    agreement_rows: list[dict[str, Any]],
    selective_rows: list[dict[str, Any]],
    ordinal_rows: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> None:
    metrics = metric_lookup(model_rows)
    comparisons = comparison_lookup(comparison_rows)
    agreement = {row["group"]: row for row in agreement_rows}
    selective = {(row["model"], row["coverage_requested"]): row for row in selective_rows}
    ordinal = {(row["model"], row["subgroup"]): row for row in ordinal_rows}
    mcnemar_lookup = {row["comparison"]: row for row in mcnemar}

    lines = [
        "# 固定50:50 ensemble・不確実性・順序誤差解析",
        "",
        "## 解析の位置付け",
        "",
        "- 対象: 全1,154患者の患者単位5-fold OOF予測",
        "- ensemble: CNN確率とViT確率の単純平均（50:50固定）",
        "- 再学習・重み最適化・閾値最適化: なし",
        f"- paired class-stratified bootstrap: {n_bootstrap:,}回、seed={seed}",
        "- 本解析は主解析完了後に計画した探索的二次解析であり、主解析を置き換えない",
        "",
        "## 全患者の性能",
        "",
        "| 指標 | CNN (95% CI) | ViT (95% CI) | Ensemble (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    primary_metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("macro_auc", "Macro-AUC"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("quadratic_weighted_kappa", "Quadratic weighted kappa"),
        ("ordinal_mae", "Ordinal MAE"),
        ("nll", "NLL"),
        ("multiclass_brier", "Multiclass Brier"),
        ("top_label_ece", "Top-label ECE"),
    ]
    for metric, label in primary_metrics:
        lines.append(
            f"| {label} | {format_estimate_ci(metrics[('cnn', metric)])} | "
            f"{format_estimate_ci(metrics[('vit', metric)])} | "
            f"{format_estimate_ci(metrics[('ensemble', metric)])} |"
        )

    lines.extend(
        [
            "",
            "## Ensembleと単独モデルのpaired差",
            "",
            "| 比較 | 指標 | 差 (Ensemble－比較モデル) | 95% CI | 判定 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for comparator in ("cnn", "vit"):
        comparison_name = f"ensemble_minus_{comparator}"
        for metric, label in (
            ("accuracy", "Accuracy"),
            ("macro_f1", "Macro-F1"),
            ("macro_auc", "Macro-AUC"),
            ("balanced_accuracy", "Balanced accuracy"),
            ("quadratic_weighted_kappa", "Weighted kappa"),
            ("ordinal_mae", "Ordinal MAE"),
            ("nll", "NLL"),
            ("multiclass_brier", "Brier"),
        ):
            row = comparisons[(comparison_name, metric)]
            ci = row["difference_ci_95"]
            lines.append(
                f"| Ensemble vs {MODEL_LABELS[comparator]} | {label} | "
                f"{row['difference']:+.4f} | {ci[0]:+.4f}～{ci[1]:+.4f} | {row['favored']} |"
            )

    lines.extend(["", "## AccuracyのMcNemar検定", ""])
    for comparator in ("cnn", "vit"):
        row = mcnemar_lookup[f"ensemble_vs_{comparator}"]
        lines.append(
            f"- Ensemble vs {MODEL_LABELS[comparator]}: Ensembleのみ正解={row['ensemble_only_correct']}、"
            f"比較モデルのみ正解={row['comparator_only_correct']}、exact p={row['exact_mcnemar_p']:.6f}"
        )

    agree = agreement["cnn_vit_agree"]
    disagree = agreement["cnn_vit_disagree"]
    lines.extend(
        [
            "",
            "## CNNとViTの予測一致・不一致",
            "",
            "| 群 | 患者数 | 割合 | CNN Acc | ViT Acc | Ensemble Acc |",
            "|---|---:|---:|---:|---:|---:|",
            f"| 一致 | {agree['num_patients']} | {agree['fraction']:.1%} | "
            f"{agree['cnn_accuracy']:.4f} | {agree['vit_accuracy']:.4f} | {agree['ensemble_accuracy']:.4f} |",
            f"| 不一致 | {disagree['num_patients']} | {disagree['fraction']:.1%} | "
            f"{disagree['cnn_accuracy']:.4f} | {disagree['vit_accuracy']:.4f} | {disagree['ensemble_accuracy']:.4f} |",
            "",
            "予測不一致群のAccuracyが低い場合、不一致を自動判定せず人間確認へ回す候補になる。"
            "ただし運用規則は外部データまたはinner validationで固定する必要がある。",
            "",
            "## Selective prediction（記述的解析）",
            "",
            "| Model | Coverage | Patients | Confidence threshold | Accuracy | Macro-F1 | Balanced Acc | Recall grade1 | Recall grade2+ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in ("cnn", "vit", "ensemble"):
        for coverage in COVERAGES:
            row = selective[(model, coverage)]
            lines.append(
                f"| {MODEL_LABELS[model]} | {coverage:.0%} | {row['num_patients']} | "
                f"{row['confidence_threshold']:.4f} | {row['accuracy']:.4f} | "
                f"{row['macro_f1']:.4f} | {row['balanced_accuracy']:.4f} | "
                f"{row['recall_grade1']:.4f} | {row['recall_grade2plus']:.4f} |"
            )

    lines.extend(
        [
            "",
            "Coverageを下げた値は同じOOF上で信頼度順に選んだ探索値であり、臨床閾値ではない。"
            "クラス構成も変化するため、Accuracyだけでなく各クラス数・Recallを併記する。",
            "",
            "## 順序誤差",
            "",
            "| Model | Exact | Adjacent error | Severe error (0↔2+) | Ordinal MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in ("cnn", "vit", "ensemble"):
        row = ordinal[(model, "all")]
        lines.append(
            f"| {MODEL_LABELS[model]} | {row['exact_count']} ({row['exact_rate']:.1%}) | "
            f"{row['adjacent_error_count']} ({row['adjacent_error_rate']:.1%}) | "
            f"{row['severe_error_count']} ({row['severe_error_rate']:.1%}) | {row['ordinal_mae']:.4f} |"
        )

    ensemble_vs_vit_auc = comparisons[("ensemble_minus_vit", "macro_auc")]
    ensemble_vs_vit_acc = comparisons[("ensemble_minus_vit", "accuracy")]
    ensemble_vs_vit_nll = comparisons[("ensemble_minus_vit", "nll")]
    lines.extend(
        [
            "",
            "## 自動解釈",
            "",
            f"- Ensemble－ViT Accuracy: {ensemble_vs_vit_acc['difference']:+.4f}, "
            f"95% CI {ensemble_vs_vit_acc['difference_ci_95'][0]:+.4f}～{ensemble_vs_vit_acc['difference_ci_95'][1]:+.4f}。",
            f"- Ensemble－ViT macro-AUC: {ensemble_vs_vit_auc['difference']:+.4f}, "
            f"95% CI {ensemble_vs_vit_auc['difference_ci_95'][0]:+.4f}～{ensemble_vs_vit_auc['difference_ci_95'][1]:+.4f}。",
            f"- Ensemble－ViT NLL: {ensemble_vs_vit_nll['difference']:+.4f}, "
            f"95% CI {ensemble_vs_vit_nll['difference_ci_95'][0]:+.4f}～{ensemble_vs_vit_nll['difference_ci_95'][1]:+.4f}。",
            "- Ensembleが単独ViTを上回らなくても、ViTがCNNへ補完情報を加えるかはEnsemble vs CNNで評価できる。",
            "- 本解析から新しい主結果を選び直さず、外部検証・repeated CV・将来の患者単位MIL設計の参考にする。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ".gitignore").write_text(
        "# Local patient-level exploratory analysis. Do not stage patient predictions.\n*\n!.gitignore\n",
        encoding="utf-8",
    )

    cnn = load_oof(args.cnn_csv)
    vit = load_oof(args.vit_csv)
    patient_ids, folds, targets, cnn_probs, vit_probs = align_predictions(cnn, vit)
    ensemble_probs = (cnn_probs + vit_probs) / 2.0
    if not np.allclose(ensemble_probs.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Ensemble probabilities do not sum to one")
    probabilities = {"cnn": cnn_probs, "vit": vit_probs, "ensemble": ensemble_probs}

    model_rows, comparison_rows = paired_bootstrap(
        targets=targets,
        probabilities=probabilities,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        ece_bins=args.ece_bins,
    )
    mcnemar = mcnemar_rows(targets, probabilities)
    agreement_rows = agreement_analysis(targets, probabilities)
    selective_rows, curve_rows = selective_prediction_analysis(targets, probabilities)
    ordinal_rows = ordinal_error_rows(targets, probabilities)

    save_csv(args.output_dir / "model_metric_summary.csv", model_rows)
    save_csv(args.output_dir / "ensemble_bootstrap_comparisons.csv", comparison_rows)
    save_csv(args.output_dir / "mcnemar_comparisons.csv", mcnemar)
    save_csv(args.output_dir / "prediction_agreement_analysis.csv", agreement_rows)
    save_csv(args.output_dir / "selective_prediction_summary.csv", selective_rows)
    save_csv(args.output_dir / "risk_coverage_curve.csv", curve_rows)
    save_csv(args.output_dir / "ordinal_error_summary.csv", ordinal_rows)
    save_patient_predictions(
        args.output_dir / "patient_ensemble_predictions.csv",
        patient_ids,
        folds,
        targets,
        probabilities,
    )

    plot_metric_comparison(model_rows, args.output_dir)
    plot_risk_coverage(curve_rows, args.output_dir)
    plot_agreement(agreement_rows, args.output_dir)
    plot_ensemble_confusion(targets, ensemble_probs, args.output_dir)
    write_report(
        args.output_dir / "ensemble_selective_analysis_report.md",
        model_rows=model_rows,
        comparison_rows=comparison_rows,
        mcnemar=mcnemar,
        agreement_rows=agreement_rows,
        selective_rows=selective_rows,
        ordinal_rows=ordinal_rows,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    results = {
        "analysis": {
            "scope": "exploratory_secondary_analysis_of_frozen_cv5_oof",
            "num_patients": len(targets),
            "ensemble_rule": "arithmetic_mean_50_50_cnn_vit_probabilities",
            "ensemble_weight_tuning": False,
            "threshold_tuning": False,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "ece_bins": args.ece_bins,
        },
        "model_metrics": model_rows,
        "ensemble_comparisons": comparison_rows,
        "mcnemar": mcnemar,
        "agreement": agreement_rows,
        "selective_prediction": selective_rows,
        "ordinal_errors": ordinal_rows,
    }
    (args.output_dir / "ensemble_selective_analysis_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lookup = metric_lookup(model_rows)
    comparisons = comparison_lookup(comparison_rows)
    print(f"Saved exploratory OOF ensemble analysis to: {args.output_dir}")
    for model in ("cnn", "vit", "ensemble"):
        print(
            f"{MODEL_LABELS[model]}: accuracy={lookup[(model, 'accuracy')]['estimate']:.4f}, "
            f"F1={lookup[(model, 'macro_f1')]['estimate']:.4f}, "
            f"AUC={lookup[(model, 'macro_auc')]['estimate']:.4f}, "
            f"QWK={lookup[(model, 'quadratic_weighted_kappa')]['estimate']:.4f}, "
            f"NLL={lookup[(model, 'nll')]['estimate']:.4f}"
        )
    for metric in ("accuracy", "macro_f1", "macro_auc", "nll", "multiclass_brier"):
        row = comparisons[("ensemble_minus_vit", metric)]
        print(
            f"Ensemble-ViT {metric}: {row['difference']:+.4f}, "
            f"CI={row['difference_ci_95']}"
        )


if __name__ == "__main__":
    main()
