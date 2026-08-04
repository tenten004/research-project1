"""Detailed analysis of frozen all-axial 5-fold out-of-fold predictions.

This script does not tune thresholds or change predictions. It compares the
fixed DeiT-small and ResNet18 patient top-5 outputs using class-wise ROC/PR
metrics, calibration, paired error overlap, original-grade subgroups, and the
selected slice metadata saved by ``src.evaluate``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
OOF_ROOT = PROJECT_ROOT / "outputs/cv5_all_axial_3class/oof_summary"
DEFAULT_VIT_CSV = OOF_ROOT / "vit_oof_patients.csv"
DEFAULT_CNN_CSV = OOF_ROOT / "resnet18_oof_patients.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/cv5_all_axial_3class/oof_detailed_analysis"
DEFAULT_TEACHER_CSVS = [
    WORKSPACE_ROOT / "教師データ/labeled_image_list_FL_preprocess.csv",
    WORKSPACE_ROOT / "教師データ/labeled_image_list_T1_preprocess.csv",
]
CLASS_LABELS = ["grade0", "grade1", "grade2+"]
MODEL_LABELS = {"cnn": "ResNet18", "vit": "DeiT-small"}
MODEL_COLORS = {"cnn": "#C55A11", "vit": "#2E74B5"}
ERROR_CATEGORIES = ["both_correct", "vit_only_correct", "cnn_only_correct", "both_wrong"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-csv", type=Path, default=DEFAULT_CNN_CSV)
    parser.add_argument("--vit-csv", type=Path, default=DEFAULT_VIT_CSV)
    parser.add_argument("--teacher-csvs", nargs="+", type=Path, default=DEFAULT_TEACHER_CSVS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def probability_columns(fieldnames: list[str]) -> list[str]:
    columns = [name for name in fieldnames if name.startswith("prob_class")]
    return sorted(columns, key=lambda name: int(name.removeprefix("prob_class")))


def load_oof_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        prob_cols = probability_columns(reader.fieldnames)
        if prob_cols != ["prob_class0", "prob_class1", "prob_class2"]:
            raise ValueError(f"Expected three class-probability columns in {path}, got {prob_cols}")
        for raw_row in reader:
            patient_id = normalize_patient_id(raw_row["patient_id"])
            if patient_id in rows:
                raise ValueError(f"Duplicate OOF patient in {path}: {patient_id}")
            probs = np.asarray([float(raw_row[col]) for col in prob_cols], dtype=np.float64)
            if not np.isclose(probs.sum(), 1.0, atol=1e-4):
                raise ValueError(f"Probabilities do not sum to one for patient {patient_id} in {path}")
            rows[patient_id] = {
                "fold": int(raw_row["cv_fold"]),
                "target": int(raw_row["target"]),
                "pred": int(raw_row["pred"]),
                "probs": probs,
                "num_slices": int(raw_row["num_slices"]),
                "max_prob": float(raw_row["max_prob"]),
                "selected_paths": raw_row.get("selected_paths", ""),
                "selected_modalities": raw_row.get("selected_modalities", ""),
                "selected_slice_indices": raw_row.get("selected_slice_indices", ""),
                "selected_confidences": raw_row.get("selected_confidences", ""),
            }
    return rows


def align_predictions(
    cnn_rows: dict[str, dict[str, Any]],
    vit_rows: dict[str, dict[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    if set(cnn_rows) != set(vit_rows):
        raise ValueError(
            f"Patient sets differ: CNN-only={len(set(cnn_rows) - set(vit_rows))}, "
            f"ViT-only={len(set(vit_rows) - set(cnn_rows))}"
        )
    patient_ids = sorted(cnn_rows)
    targets = np.asarray([cnn_rows[pid]["target"] for pid in patient_ids], dtype=np.int64)
    vit_targets = np.asarray([vit_rows[pid]["target"] for pid in patient_ids], dtype=np.int64)
    folds = np.asarray([cnn_rows[pid]["fold"] for pid in patient_ids], dtype=np.int64)
    vit_folds = np.asarray([vit_rows[pid]["fold"] for pid in patient_ids], dtype=np.int64)
    if not np.array_equal(targets, vit_targets):
        raise ValueError("CNN and ViT target labels differ")
    if not np.array_equal(folds, vit_folds):
        raise ValueError("CNN and ViT OOF fold assignments differ")
    if set(folds.tolist()) != {1, 2, 3, 4, 5}:
        raise ValueError(f"Expected all five folds, got {sorted(set(folds.tolist()))}")

    cnn_preds = np.asarray([cnn_rows[pid]["pred"] for pid in patient_ids], dtype=np.int64)
    vit_preds = np.asarray([vit_rows[pid]["pred"] for pid in patient_ids], dtype=np.int64)
    cnn_probs = np.stack([cnn_rows[pid]["probs"] for pid in patient_ids])
    vit_probs = np.stack([vit_rows[pid]["probs"] for pid in patient_ids])
    cnn_aligned = [cnn_rows[pid] for pid in patient_ids]
    vit_aligned = [vit_rows[pid] for pid in patient_ids]
    return patient_ids, folds, targets, cnn_preds, cnn_probs, vit_preds, vit_probs, cnn_aligned, vit_aligned


def load_original_grades(paths: list[Path]) -> dict[str, int]:
    grades_by_patient: dict[str, set[int]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = normalize_patient_id(row["ID"])
                grade = int(float(row["wm"]))
                grades_by_patient.setdefault(patient_id, set()).add(grade)

    inconsistent = {pid: values for pid, values in grades_by_patient.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"Inconsistent original grades: {list(inconsistent.items())[:5]}")
    return {pid: next(iter(values)) for pid, values in grades_by_patient.items()}


def binary_reliability_bins(targets: np.ndarray, probabilities: np.ndarray, n_bins: int) -> list[dict[str, float | int]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result: list[dict[str, float | int]] = []
    for index in range(n_bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (probabilities >= lower) & (probabilities < upper)
        if index == n_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        result.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "mean_confidence": float(np.mean(probabilities[mask])),
                "observed_frequency": float(np.mean(targets[mask])),
            }
        )
    return result


def top_label_ece(targets: np.ndarray, probs: np.ndarray, n_bins: int) -> float:
    preds = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = (preds == targets).astype(np.float64)
    bins = binary_reliability_bins(correct, confidence, n_bins=n_bins)
    total = len(targets)
    return float(
        sum(
            int(row["count"]) / total
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


def metric_bundle(targets: np.ndarray, probs: np.ndarray, ece_bins: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for class_id in range(probs.shape[1]):
        binary_targets = (targets == class_id).astype(np.int64)
        metrics[f"class{class_id}_roc_auc"] = float(roc_auc_score(binary_targets, probs[:, class_id]))
        metrics[f"class{class_id}_pr_auc_ap"] = float(
            average_precision_score(binary_targets, probs[:, class_id])
        )
        metrics[f"class{class_id}_brier"] = float(
            np.mean((probs[:, class_id] - binary_targets) ** 2)
        )
    metrics["nll"] = multiclass_nll(targets, probs)
    metrics["multiclass_brier"] = multiclass_brier(targets, probs)
    metrics["top_label_ece"] = top_label_ece(targets, probs, n_bins=ece_bins)
    return metrics


def percentile_ci(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def paired_bootstrap_metrics(
    targets: np.ndarray,
    cnn_probs: np.ndarray,
    vit_probs: np.ndarray,
    n_bootstrap: int,
    seed: int,
    ece_bins: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    if n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    point = {
        "cnn": metric_bundle(targets, cnn_probs, ece_bins=ece_bins),
        "vit": metric_bundle(targets, vit_probs, ece_bins=ece_bins),
    }
    metric_names = list(point["cnn"])
    distributions = {
        metric: {
            "cnn": np.empty(n_bootstrap, dtype=np.float64),
            "vit": np.empty(n_bootstrap, dtype=np.float64),
            "difference": np.empty(n_bootstrap, dtype=np.float64),
        }
        for metric in metric_names
    }
    class_indices = [np.flatnonzero(targets == class_id) for class_id in sorted(np.unique(targets))]
    rng = np.random.default_rng(seed)

    for iteration in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        rng.shuffle(sampled)
        sampled_targets = targets[sampled]
        cnn_values = metric_bundle(sampled_targets, cnn_probs[sampled], ece_bins=ece_bins)
        vit_values = metric_bundle(sampled_targets, vit_probs[sampled], ece_bins=ece_bins)
        for metric in metric_names:
            distributions[metric]["cnn"][iteration] = cnn_values[metric]
            distributions[metric]["vit"][iteration] = vit_values[metric]
            distributions[metric]["difference"][iteration] = vit_values[metric] - cnn_values[metric]

    results: list[dict[str, Any]] = []
    for metric in metric_names:
        difference_ci = percentile_ci(distributions[metric]["difference"])
        lower_is_better = metric in {"nll", "multiclass_brier", "top_label_ece"} or metric.endswith("_brier")
        difference = point["vit"][metric] - point["cnn"][metric]
        if difference_ci[0] > 0:
            favored = "cnn" if lower_is_better else "vit"
        elif difference_ci[1] < 0:
            favored = "vit" if lower_is_better else "cnn"
        else:
            favored = "inconclusive"
        results.append(
            {
                "metric": metric,
                "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                "cnn_estimate": point["cnn"][metric],
                "cnn_ci_95": percentile_ci(distributions[metric]["cnn"]),
                "vit_estimate": point["vit"][metric],
                "vit_ci_95": percentile_ci(distributions[metric]["vit"]),
                "difference_vit_minus_cnn": difference,
                "difference_ci_95": difference_ci,
                "difference_ci_excludes_zero": bool(difference_ci[0] > 0 or difference_ci[1] < 0),
                "favored_model": favored,
            }
        )
    return results, distributions


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def original_grade_analysis(
    original_grades: np.ndarray,
    targets: np.ndarray,
    cnn_preds: np.ndarray,
    vit_preds: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 97)
    for grade in sorted(np.unique(original_grades)):
        indices = np.flatnonzero(original_grades == grade)
        expected_target = min(int(grade), 2)
        if not np.all(targets[indices] == expected_target):
            raise ValueError(f"Merged target mismatch for original grade {grade}")
        cnn_correct = cnn_preds[indices] == expected_target
        vit_correct = vit_preds[indices] == expected_target
        differences = np.empty(n_bootstrap, dtype=np.float64)
        for iteration in range(n_bootstrap):
            sampled_local = rng.choice(len(indices), size=len(indices), replace=True)
            differences[iteration] = float(
                np.mean(vit_correct[sampled_local]) - np.mean(cnn_correct[sampled_local])
            )
        cnn_successes = int(np.count_nonzero(cnn_correct))
        vit_successes = int(np.count_nonzero(vit_correct))
        results.append(
            {
                "original_grade": int(grade),
                "merged_target": expected_target,
                "num_patients": len(indices),
                "cnn_detected": cnn_successes,
                "cnn_recall": cnn_successes / len(indices),
                "cnn_wilson_ci_95": wilson_interval(cnn_successes, len(indices)),
                "vit_detected": vit_successes,
                "vit_recall": vit_successes / len(indices),
                "vit_wilson_ci_95": wilson_interval(vit_successes, len(indices)),
                "difference_vit_minus_cnn": vit_successes / len(indices) - cnn_successes / len(indices),
                "difference_bootstrap_ci_95": percentile_ci(differences),
            }
        )
    return results


def determine_error_categories(
    targets: np.ndarray,
    cnn_preds: np.ndarray,
    vit_preds: np.ndarray,
) -> np.ndarray:
    cnn_correct = cnn_preds == targets
    vit_correct = vit_preds == targets
    categories = np.full(len(targets), "both_wrong", dtype=object)
    categories[cnn_correct & vit_correct] = "both_correct"
    categories[~cnn_correct & vit_correct] = "vit_only_correct"
    categories[cnn_correct & ~vit_correct] = "cnn_only_correct"
    return categories


def error_overlap_rows(
    categories: np.ndarray,
    targets: np.ndarray,
    original_grades: np.ndarray,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    subgroups: list[tuple[str, str, np.ndarray]] = [("all", "all", np.ones(len(targets), dtype=bool))]
    for class_id, label in enumerate(CLASS_LABELS):
        subgroups.append(("merged_target", label, targets == class_id))
    for grade in sorted(np.unique(original_grades)):
        subgroups.append(("original_grade", f"grade{grade}", original_grades == grade))

    for subgroup_type, subgroup, mask in subgroups:
        counts = Counter(categories[mask].tolist())
        row: dict[str, Any] = {
            "subgroup_type": subgroup_type,
            "subgroup": subgroup,
            "num_patients": int(np.count_nonzero(mask)),
        }
        for category in ERROR_CATEGORIES:
            row[category] = counts.get(category, 0)
        results.append(row)
    return results


def split_semicolon(value: str) -> list[str]:
    return [item for item in value.split(";") if item]


def selected_image_rows(
    model: str,
    aligned_rows: list[dict[str, Any]],
    targets: np.ndarray,
    original_grades: np.ndarray,
    categories: np.ndarray,
) -> list[dict[str, Any]]:
    model_correct = np.asarray([row["pred"] for row in aligned_rows]) == targets
    subgroups: list[tuple[str, str, np.ndarray]] = [("all", "all", np.ones(len(targets), dtype=bool))]
    for class_id, label in enumerate(CLASS_LABELS):
        subgroups.append(("merged_target", label, targets == class_id))
    for grade in sorted(np.unique(original_grades)):
        subgroups.append(("original_grade", f"grade{grade}", original_grades == grade))
    subgroups.extend(
        [
            ("model_correctness", "correct", model_correct),
            ("model_correctness", "wrong", ~model_correct),
        ]
    )
    for category in ERROR_CATEGORIES:
        subgroups.append(("paired_error_category", category, categories == category))

    output: list[dict[str, Any]] = []
    for subgroup_type, subgroup, mask in subgroups:
        modalities: list[str] = []
        axial_indices: list[int] = []
        for index in np.flatnonzero(mask):
            modalities.extend(split_semicolon(aligned_rows[int(index)]["selected_modalities"]))
            for value in split_semicolon(aligned_rows[int(index)]["selected_slice_indices"]):
                try:
                    axial_indices.append(int(value))
                except ValueError:
                    continue
        modality_counts = Counter(modalities)
        total_selected = len(modalities)
        output.append(
            {
                "model": model,
                "subgroup_type": subgroup_type,
                "subgroup": subgroup,
                "num_patients": int(np.count_nonzero(mask)),
                "num_selected_images": total_selected,
                "fl_count": modality_counts.get("FL", 0),
                "t1_count": modality_counts.get("T1", 0),
                "fl_fraction": modality_counts.get("FL", 0) / total_selected if total_selected else float("nan"),
                "known_axial_count": len(axial_indices),
                "axial_9_15_fraction": (
                    sum(9 <= value <= 15 for value in axial_indices) / len(axial_indices)
                    if axial_indices
                    else float("nan")
                ),
                "mean_axial": float(np.mean(axial_indices)) if axial_indices else float("nan"),
                "median_axial": float(np.median(axial_indices)) if axial_indices else float("nan"),
            }
        )
    return output


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized_row: dict[str, Any] = {}
        for key, value in row.items():
            normalized_row[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
        normalized.append(normalized_row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(normalized[0].keys()))
        writer.writeheader()
        writer.writerows(normalized)


def save_patient_errors(
    path: Path,
    patient_ids: list[str],
    folds: np.ndarray,
    targets: np.ndarray,
    original_grades: np.ndarray,
    categories: np.ndarray,
    cnn_preds: np.ndarray,
    cnn_probs: np.ndarray,
    vit_preds: np.ndarray,
    vit_probs: np.ndarray,
    cnn_rows: list[dict[str, Any]],
    vit_rows: list[dict[str, Any]],
) -> None:
    fields = [
        "patient_id", "cv_fold", "merged_target", "original_grade", "error_category",
        "cnn_pred", "cnn_prob_class0", "cnn_prob_class1", "cnn_prob_class2",
        "vit_pred", "vit_prob_class0", "vit_prob_class1", "vit_prob_class2",
        "cnn_selected_modalities", "cnn_selected_slice_indices",
        "vit_selected_modalities", "vit_selected_slice_indices",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index, patient_id in enumerate(patient_ids):
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "cv_fold": int(folds[index]),
                    "merged_target": int(targets[index]),
                    "original_grade": int(original_grades[index]),
                    "error_category": str(categories[index]),
                    "cnn_pred": int(cnn_preds[index]),
                    "cnn_prob_class0": cnn_probs[index, 0],
                    "cnn_prob_class1": cnn_probs[index, 1],
                    "cnn_prob_class2": cnn_probs[index, 2],
                    "vit_pred": int(vit_preds[index]),
                    "vit_prob_class0": vit_probs[index, 0],
                    "vit_prob_class1": vit_probs[index, 1],
                    "vit_prob_class2": vit_probs[index, 2],
                    "cnn_selected_modalities": cnn_rows[index]["selected_modalities"],
                    "cnn_selected_slice_indices": cnn_rows[index]["selected_slice_indices"],
                    "vit_selected_modalities": vit_rows[index]["selected_modalities"],
                    "vit_selected_slice_indices": vit_rows[index]["selected_slice_indices"],
                }
            )


def plot_discrimination_curves(
    targets: np.ndarray,
    cnn_probs: np.ndarray,
    vit_probs: np.ndarray,
    output_dir: Path,
) -> None:
    fig_roc, axes_roc = plt.subplots(1, 3, figsize=(14, 4.2))
    fig_pr, axes_pr = plt.subplots(1, 3, figsize=(14, 4.2))
    for class_id, class_label in enumerate(CLASS_LABELS):
        binary = (targets == class_id).astype(np.int64)
        prevalence = float(np.mean(binary))
        for model, probs in (("cnn", cnn_probs), ("vit", vit_probs)):
            fpr, tpr, _ = roc_curve(binary, probs[:, class_id])
            auc_value = roc_auc_score(binary, probs[:, class_id])
            axes_roc[class_id].plot(
                fpr,
                tpr,
                color=MODEL_COLORS[model],
                label=f"{MODEL_LABELS[model]} AUC={auc_value:.3f}",
            )
            precision, recall, _ = precision_recall_curve(binary, probs[:, class_id])
            ap = average_precision_score(binary, probs[:, class_id])
            axes_pr[class_id].plot(
                recall,
                precision,
                color=MODEL_COLORS[model],
                label=f"{MODEL_LABELS[model]} AP={ap:.3f}",
            )
        axes_roc[class_id].plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
        axes_roc[class_id].set_title(class_label)
        axes_roc[class_id].set_xlabel("False positive rate")
        axes_roc[class_id].set_ylabel("True positive rate")
        axes_roc[class_id].legend(fontsize=8)
        axes_roc[class_id].grid(alpha=0.2)

        axes_pr[class_id].axhline(prevalence, linestyle="--", color="#888888", linewidth=1)
        axes_pr[class_id].set_title(f"{class_label} (prevalence={prevalence:.3f})")
        axes_pr[class_id].set_xlabel("Recall")
        axes_pr[class_id].set_ylabel("Precision")
        axes_pr[class_id].legend(fontsize=8)
        axes_pr[class_id].grid(alpha=0.2)

    fig_roc.suptitle("One-vs-rest ROC curves: 5-fold OOF patients")
    fig_pr.suptitle("One-vs-rest precision-recall curves: 5-fold OOF patients")
    fig_roc.tight_layout(rect=(0, 0, 1, 0.95))
    fig_pr.tight_layout(rect=(0, 0, 1, 0.95))
    fig_roc.savefig(output_dir / "class_roc_curves.png", dpi=180, bbox_inches="tight")
    fig_pr.savefig(output_dir / "class_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig_roc)
    plt.close(fig_pr)


def plot_reliability(
    targets: np.ndarray,
    cnn_probs: np.ndarray,
    vit_probs: np.ndarray,
    n_bins: int,
    output_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6, 5.2))
    for model, probs in (("cnn", cnn_probs), ("vit", vit_probs)):
        preds = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        correct = (preds == targets).astype(np.float64)
        bins = binary_reliability_bins(correct, confidence, n_bins=n_bins)
        axis.plot(
            [float(row["mean_confidence"]) for row in bins],
            [float(row["observed_frequency"]) for row in bins],
            marker="o",
            color=MODEL_COLORS[model],
            label=f"{MODEL_LABELS[model]} (ECE={top_label_ece(targets, probs, n_bins):.3f})",
        )
    axis.plot([0, 1], [0, 1], "--", color="#666666")
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Observed accuracy")
    axis.set_title("Top-label reliability: 5-fold OOF patients")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "top_label_reliability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for class_id, class_label in enumerate(CLASS_LABELS):
        binary = (targets == class_id).astype(np.float64)
        for model, probs in (("cnn", cnn_probs), ("vit", vit_probs)):
            bins = binary_reliability_bins(binary, probs[:, class_id], n_bins=n_bins)
            axes[class_id].plot(
                [float(row["mean_confidence"]) for row in bins],
                [float(row["observed_frequency"]) for row in bins],
                marker="o",
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        axes[class_id].plot([0, 1], [0, 1], "--", color="#666666")
        axes[class_id].set_title(class_label)
        axes[class_id].set_xlabel("Predicted probability")
        axes[class_id].set_ylabel("Observed frequency")
        axes[class_id].legend(fontsize=8)
        axes[class_id].grid(alpha=0.2)
    fig.suptitle("Class-wise reliability: 5-fold OOF patients")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "class_reliability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_and_errors(
    targets: np.ndarray,
    cnn_preds: np.ndarray,
    vit_preds: np.ndarray,
    categories: np.ndarray,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(11.5, 4.6), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    colorbar_axis = fig.add_subplot(grid[0, 2])
    for axis, model, preds in ((axes[0], "cnn", cnn_preds), (axes[1], "vit", vit_preds)):
        cm = confusion_matrix(targets, preds, labels=[0, 1, 2])
        row_totals = cm.sum(axis=1, keepdims=True)
        normalized = cm / row_totals
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
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
        axis.set_title(MODEL_LABELS[model])
    fig.colorbar(image, cax=colorbar_axis, label="Row-normalized proportion")
    fig.suptitle("Patient-level OOF confusion matrices")
    fig.savefig(output_dir / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    counts = [int(np.count_nonzero(categories == category)) for category in ERROR_CATEGORIES]
    labels = ["Both correct", "ViT only", "CNN only", "Both wrong"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(labels, counts, color=["#70AD47", "#2E74B5", "#C55A11", "#A5A5A5"])
    axis.bar_label(bars)
    axis.set_ylabel("Patients")
    axis.set_title("Paired patient error overlap")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "error_overlap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def metric_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["metric"]: row for row in rows}


def format_estimate_ci(estimate: float, ci: list[float]) -> str:
    return f"{estimate:.4f} ({ci[0]:.4f}, {ci[1]:.4f})"


def write_report(
    path: Path,
    bootstrap_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    grade_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    n_patients: int,
    n_bootstrap: int,
    seed: int,
) -> None:
    metrics = metric_lookup(bootstrap_rows)
    lines = [
        "# 全axial・患者単位5-fold OOF 詳細解析",
        "",
        f"- 対象患者: {n_patients:,}人",
        "- 評価: 各患者につき検証予測1回のout-of-fold予測",
        "- モデル: DeiT-small 224 vs ResNet18 224",
        "- 集約: best-loss checkpoint・患者top-5（固定）",
        f"- paired bootstrap: {n_bootstrap:,}回、seed={seed}",
        "- 本解析では閾値・top-k・モデル条件を再調整していない",
        "",
        "## クラス別判別性能",
        "",
        "| クラス | 指標 | CNN (95% CI) | ViT (95% CI) | ViT−CNN (95% CI) | 判定 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for class_id, class_label in enumerate(CLASS_LABELS):
        for suffix, label in (("roc_auc", "ROC-AUC"), ("pr_auc_ap", "PR-AUC (AP)")):
            row = metrics[f"class{class_id}_{suffix}"]
            diff = row["difference_vit_minus_cnn"]
            diff_ci = row["difference_ci_95"]
            lines.append(
                f"| {class_label} | {label} | "
                f"{format_estimate_ci(row['cnn_estimate'], row['cnn_ci_95'])} | "
                f"{format_estimate_ci(row['vit_estimate'], row['vit_ci_95'])} | "
                f"{diff:+.4f} ({diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}) | "
                f"{row['favored_model']} |"
            )

    lines.extend(
        [
            "",
            "## キャリブレーション",
            "",
            "値が小さいほど良い。Brier scoreは3クラス誤差の合計平均。",
            "",
            "| 指標 | CNN (95% CI) | ViT (95% CI) | ViT−CNN (95% CI) | 判定 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for metric_name, label in (
        ("nll", "NLL"),
        ("multiclass_brier", "Multiclass Brier"),
        ("top_label_ece", "Top-label ECE"),
    ):
        row = metrics[metric_name]
        diff_ci = row["difference_ci_95"]
        lines.append(
            f"| {label} | {format_estimate_ci(row['cnn_estimate'], row['cnn_ci_95'])} | "
            f"{format_estimate_ci(row['vit_estimate'], row['vit_ci_95'])} | "
            f"{row['difference_vit_minus_cnn']:+.4f} ({diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}) | "
            f"{row['favored_model']} |"
        )

    overall_error = next(row for row in error_rows if row["subgroup_type"] == "all")
    lines.extend(
        [
            "",
            "## エラー重複",
            "",
            f"- 両モデル正解: {overall_error['both_correct']}人",
            f"- ViTのみ正解: {overall_error['vit_only_correct']}人",
            f"- CNNのみ正解: {overall_error['cnn_only_correct']}人",
            f"- 両モデル不正解: {overall_error['both_wrong']}人",
            "",
            "## 元grade別検出",
            "",
            "| 元grade | 患者数 | CNN | ViT | ViT−CNN bootstrap 95% CI |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in grade_rows:
        diff_ci = row["difference_bootstrap_ci_95"]
        lines.append(
            f"| {row['original_grade']} | {row['num_patients']} | "
            f"{row['cnn_detected']}/{row['num_patients']} ({row['cnn_recall']:.3f}) | "
            f"{row['vit_detected']}/{row['num_patients']} ({row['vit_recall']:.3f}) | "
            f"{row['difference_vit_minus_cnn']:+.3f} ({diff_ci[0]:+.3f}, {diff_ci[1]:+.3f}) |"
        )

    lines.extend(["", "## top-5選択画像監査", ""])
    all_selection = [
        row for row in selected_rows if row["subgroup_type"] == "all" and row["subgroup"] == "all"
    ]
    for row in all_selection:
        lines.append(
            f"- {MODEL_LABELS[row['model']]}: FL={row['fl_fraction']:.1%}、"
            f"axial 9–15={row['axial_9_15_fraction']:.1%}、"
            f"axial中央値={row['median_axial']:.1f}"
        )

    significant_discrimination = [
        row for row in bootstrap_rows
        if ("roc_auc" in row["metric"] or "pr_auc" in row["metric"])
        and row["difference_ci_excludes_zero"]
    ]
    significant_calibration = [
        metrics[name]
        for name in ("nll", "multiclass_brier", "top_label_ece")
        if metrics[name]["difference_ci_excludes_zero"]
    ]
    lines.extend(
        [
            "",
            "## 自動要約",
            "",
            "- 95%信頼区間が0をまたがないクラス別ROC/PR差: "
            + (", ".join(f"{row['metric']}→{row['favored_model']}" for row in significant_discrimination) or "なし"),
            "- 95%信頼区間が0をまたがないキャリブレーション差: "
            + (", ".join(f"{row['metric']}→{row['favored_model']}" for row in significant_calibration) or "なし"),
            "",
            "## 限界",
            "",
            "5-fold OOF予測は単一split依存を抑えるが、モデル・top-5は以前の探索後に固定されている。"
            "Bootstrap信頼区間は5個の学習済みfoldモデルを条件とした患者標本の不確実性であり、"
            "学習データ変動の全てを表さない。閾値最適化を行う場合は外側foldの予測を使わず、"
            "各training fold内のinner validationだけで決定する必要がある。外部施設検証が最終的に必要である。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cnn_source = load_oof_predictions(args.cnn_csv)
    vit_source = load_oof_predictions(args.vit_csv)
    (
        patient_ids,
        folds,
        targets,
        cnn_preds,
        cnn_probs,
        vit_preds,
        vit_probs,
        cnn_rows,
        vit_rows,
    ) = align_predictions(cnn_source, vit_source)
    if len(patient_ids) != 1154:
        raise ValueError(f"Expected 1,154 OOF patients, got {len(patient_ids)}")

    grade_map = load_original_grades(args.teacher_csvs)
    missing_grades = [patient_id for patient_id in patient_ids if patient_id not in grade_map]
    if missing_grades:
        raise ValueError(f"Missing original grade for OOF patients: {missing_grades[:5]}")
    original_grades = np.asarray([grade_map[patient_id] for patient_id in patient_ids], dtype=np.int64)
    merged_from_original = np.minimum(original_grades, 2)
    if not np.array_equal(merged_from_original, targets):
        raise ValueError("Original grades do not reproduce merged OOF targets")

    bootstrap_rows, _distributions = paired_bootstrap_metrics(
        targets=targets,
        cnn_probs=cnn_probs,
        vit_probs=vit_probs,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        ece_bins=args.ece_bins,
    )
    categories = determine_error_categories(targets, cnn_preds, vit_preds)
    error_rows = error_overlap_rows(categories, targets, original_grades)
    grade_rows = original_grade_analysis(
        original_grades=original_grades,
        targets=targets,
        cnn_preds=cnn_preds,
        vit_preds=vit_preds,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    selected_rows = [
        *selected_image_rows("cnn", cnn_rows, targets, original_grades, categories),
        *selected_image_rows("vit", vit_rows, targets, original_grades, categories),
    ]

    save_csv(bootstrap_rows, args.output_dir / "metric_bootstrap_summary.csv")
    save_csv(error_rows, args.output_dir / "error_overlap.csv")
    save_csv(grade_rows, args.output_dir / "original_grade_metrics.csv")
    save_csv(selected_rows, args.output_dir / "selected_image_summary.csv")
    save_patient_errors(
        args.output_dir / "patient_error_analysis.csv",
        patient_ids,
        folds,
        targets,
        original_grades,
        categories,
        cnn_preds,
        cnn_probs,
        vit_preds,
        vit_probs,
        cnn_rows,
        vit_rows,
    )

    plot_discrimination_curves(targets, cnn_probs, vit_probs, args.output_dir)
    plot_reliability(targets, cnn_probs, vit_probs, args.ece_bins, args.output_dir)
    plot_confusion_and_errors(targets, cnn_preds, vit_preds, categories, args.output_dir)

    results = {
        "analysis": {
            "protocol": "frozen_all_axial_patient_cv5_top5_oof_detailed_analysis",
            "num_patients": len(patient_ids),
            "folds": 5,
            "class_counts": {
                CLASS_LABELS[class_id]: int(np.count_nonzero(targets == class_id))
                for class_id in range(3)
            },
            "original_grade_counts": {
                f"grade{grade}": int(np.count_nonzero(original_grades == grade))
                for grade in sorted(np.unique(original_grades))
            },
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.seed,
            "ece_bins": args.ece_bins,
            "threshold_tuning_performed": False,
            "pr_auc_definition": "average precision",
        },
        "metric_bootstrap": bootstrap_rows,
        "error_overlap": error_rows,
        "original_grade_metrics": grade_rows,
        "selected_image_summary": selected_rows,
    }
    with (args.output_dir / "detailed_analysis_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, allow_nan=False)

    write_report(
        args.output_dir / "detailed_analysis_report.md",
        bootstrap_rows=bootstrap_rows,
        error_rows=error_rows,
        grade_rows=grade_rows,
        selected_rows=selected_rows,
        n_patients=len(patient_ids),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    metrics = metric_lookup(bootstrap_rows)
    print(f"Saved detailed OOF analysis to: {args.output_dir}")
    for class_id, label in enumerate(CLASS_LABELS):
        auc_row = metrics[f"class{class_id}_roc_auc"]
        ap_row = metrics[f"class{class_id}_pr_auc_ap"]
        print(
            f"{label}: ROC-AUC diff={auc_row['difference_vit_minus_cnn']:+.4f} "
            f"CI={auc_row['difference_ci_95']}; AP diff={ap_row['difference_vit_minus_cnn']:+.4f} "
            f"CI={ap_row['difference_ci_95']}"
        )
    for name in ("nll", "multiclass_brier", "top_label_ece"):
        row = metrics[name]
        print(
            f"{name}: CNN={row['cnn_estimate']:.4f}, ViT={row['vit_estimate']:.4f}, "
            f"diff={row['difference_vit_minus_cnn']:+.4f}, CI={row['difference_ci_95']}"
        )


if __name__ == "__main__":
    main()
