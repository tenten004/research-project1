"""Combine five out-of-fold patient predictions and run the final paired analysis."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/cv5_all_axial_3class"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/cv5_all_axial_3class"
MODEL_DIRS = {
    "vit": "deit_small_224_reg",
    "resnet18": "resnet18_224_reg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def fold_metrics_path(fold: int, model: str) -> Path:
    return OUTPUT_ROOT / MODEL_DIRS[model] / f"fold{fold}/metrics" / (
        f"{model}_eval_val_patient_top_k_confidence_k5_best_loss.json"
    )


def fold_patients_path(fold: int, model: str) -> Path:
    return OUTPUT_ROOT / MODEL_DIRS[model] / f"fold{fold}/metrics" / (
        f"{model}_eval_val_patient_top_k_confidence_k5_best_loss_patients.csv"
    )


def load_assignments() -> dict[str, dict[str, int]]:
    path = DATA_ROOT / "patient_fold_assignments.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    assignments: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            assignments[row["patient_id"]] = {
                "label": int(row["label"]),
                "validation_fold": int(row["validation_fold"]),
            }
    return assignments


def combine_model_predictions(
    model: str,
    assignments: dict[str, dict[str, int]],
    output_path: Path,
) -> tuple[set[str], list[dict[str, Any]]]:
    combined_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    seen_patients: set[str] = set()
    fold_metrics: list[dict[str, Any]] = []

    for fold in range(1, 6):
        patient_path = fold_patients_path(fold, model)
        metric_path = fold_metrics_path(fold, model)
        if not patient_path.exists() or not metric_path.exists():
            raise FileNotFoundError(
                f"Cross-validation is incomplete for fold{fold} {model}: "
                f"patients={patient_path.exists()}, metrics={metric_path.exists()}"
            )

        with patient_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Missing CSV header: {patient_path}")
            if fieldnames is None:
                fieldnames = ["cv_fold", *reader.fieldnames]
            elif fieldnames[1:] != reader.fieldnames:
                raise ValueError(f"Patient CSV schema differs in fold{fold} {model}")

            for row in reader:
                patient_id = row["patient_id"]
                if patient_id in seen_patients:
                    raise ValueError(f"Patient occurs in multiple validation folds for {model}: {patient_id}")
                if patient_id not in assignments:
                    raise ValueError(f"Unknown patient in {model} fold{fold}: {patient_id}")
                expected = assignments[patient_id]
                if expected["validation_fold"] != fold:
                    raise ValueError(
                        f"Patient {patient_id} expected in fold{expected['validation_fold']}, found in fold{fold}"
                    )
                if expected["label"] != int(row["target"]):
                    raise ValueError(f"Target mismatch for patient {patient_id} in {model}")
                combined_rows.append({"cv_fold": str(fold), **row})
                seen_patients.add(patient_id)

        with metric_path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        fold_metrics.append(
            {
                "fold": fold,
                "model": model,
                "num_patients": metrics["num_patients"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["f1"],
                "macro_roc_auc": metrics["roc_auc"],
                "grade2plus_recall": metrics["confusion_matrix"][2][2]
                / sum(metrics["confusion_matrix"][2]),
            }
        )

    if seen_patients != set(assignments):
        missing = sorted(set(assignments) - seen_patients)
        raise ValueError(f"Not every patient has an out-of-fold prediction for {model}: missing={missing[:5]}")
    if fieldnames is None:
        raise RuntimeError(f"No patient predictions found for {model}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)
    return seen_patients, fold_metrics


def write_fold_metrics(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    assignments = load_assignments()
    summary_root = OUTPUT_ROOT / "oof_summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    vit_csv = summary_root / "vit_oof_patients.csv"
    cnn_csv = summary_root / "resnet18_oof_patients.csv"
    vit_ids, vit_metrics = combine_model_predictions("vit", assignments, vit_csv)
    cnn_ids, cnn_metrics = combine_model_predictions("resnet18", assignments, cnn_csv)
    if vit_ids != cnn_ids:
        raise ValueError("ViT and CNN out-of-fold patient sets differ")

    write_fold_metrics([*vit_metrics, *cnn_metrics], summary_root / "fold_metrics.csv")

    bootstrap_output = summary_root / "paired_bootstrap"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/paired_patient_bootstrap.py"),
            "--cnn-csv",
            str(cnn_csv),
            "--vit-csv",
            str(vit_csv),
            "--output-dir",
            str(bootstrap_output),
            "--n-bootstrap",
            str(args.n_bootstrap),
            "--seed",
            str(args.seed),
            "--analysis-label",
            "All-axial top-5: patient-level 5-fold OOF paired bootstrap",
            "--data-scope",
            "FL+T1 all axial, patient-level 5-fold OOF, 3 classes",
            "--cv-folds",
            "5",
            "--limitation",
            (
                "The five-fold protocol was frozen after exploratory development on an earlier validation "
                "split. These intervals use out-of-fold predictions from all patients and quantify paired "
                "patient-sampling uncertainty conditional on the five fitted fold models. They do not remove "
                "all prior model/pooling selection bias or training-fold dependence; repeated cross-validation "
                "or external validation is still required for the strongest generalization claim."
            ),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    completion = {
        "protocol": "frozen_all_axial_patient_cv5_top5",
        "num_patients": len(assignments),
        "folds": 5,
        "models": ["deit_small_224_reg", "resnet18_224_reg"],
        "out_of_fold_predictions_verified": True,
        "each_patient_validation_count": 1,
        "paired_bootstrap_results": str(
            (bootstrap_output / "paired_bootstrap_results.json").relative_to(PROJECT_ROOT)
        ),
    }
    with (summary_root / "cv_complete.json").open("w", encoding="utf-8") as f:
        json.dump(completion, f, ensure_ascii=False, indent=2)
    print(f"Cross-validation summary saved to: {summary_root}")


if __name__ == "__main__":
    main()
