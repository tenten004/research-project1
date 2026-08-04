"""Run the frozen all-axial patient-level cross-validation protocol.

For every requested fold and model this script:
1. generates a resolved YAML config from the frozen fair-comparison config,
2. trains the model unless a complete result already exists,
3. evaluates the best-loss checkpoint with patient-level top-5 pooling, and
4. writes a resumable cross-validation status CSV.

Training itself is intentionally not checkpoint-resumable because ``src.train``
does not restore optimizer/scheduler state. An interrupted run must be restarted
explicitly with ``--restart-incomplete``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/cv5_all_axial_3class"
CONFIG_ROOT = PROJECT_ROOT / "config/generated_cv5_all_axial_3class"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/cv5_all_axial_3class"
BASE_CONFIGS = {
    "vit": PROJECT_ROOT / "config/config_repro_vit_all_axial_patient_split_3class_deit_small_224_reg.yaml",
    "resnet18": PROJECT_ROOT / "config/config_repro_cnn_all_axial_patient_split_3class_resnet18_224_reg.yaml",
}
MODEL_OUTPUT_NAMES = {
    "vit": "deit_small_224_reg",
    "resnet18": "resnet18_224_reg",
}
EXPECTED_AUGMENTATION = {
    "enabled": True,
    "rotation": 15,
    "hflip": 0.5,
    "vflip": 0.0,
    "brightness": 0.2,
    "contrast": 0.2,
    "gamma_min": 0.8,
    "gamma_max": 1.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["vit", "resnet18"],
        default=["vit", "resnet18"],
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate and validate configs without starting training.",
    )
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="Delete and restart an interrupted run; completed evaluations are still skipped.",
    )
    return parser.parse_args()


def relative_posix(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    return data


def assert_close(actual: Any, expected: float, name: str) -> None:
    if abs(float(actual) - expected) > 1e-12:
        raise ValueError(f"Frozen protocol mismatch for {name}: expected {expected}, got {actual}")


def validate_frozen_base_config(cfg: dict[str, Any], model: str) -> None:
    if int(cfg["seed"]) != 1:
        raise ValueError(f"Frozen protocol requires seed=1, got {cfg['seed']}")
    if int(cfg["data"]["image_size"]) != 224:
        raise ValueError("Frozen protocol requires image_size=224")
    assert_close(cfg["data"]["mean"], 0.5, "data.mean")
    assert_close(cfg["data"]["std"], 0.5, "data.std")
    if int(cfg["train"]["batch_size"]) != 16 or int(cfg["train"]["epochs"]) != 30:
        raise ValueError("Frozen protocol requires batch_size=16 and epochs=30")
    assert_close(cfg["train"]["lr"], 0.00003, "train.lr")
    assert_close(cfg["train"]["weight_decay"], 0.1, "train.weight_decay")
    if str(cfg["train"]["scheduler"]).lower() != "cosine":
        raise ValueError("Frozen protocol requires cosine scheduler")
    if str(cfg["train"]["best_metric"]).lower() != "loss":
        raise ValueError("Frozen protocol requires best_metric=loss")
    if str(cfg["train"]["optimizer"]["name"]).lower() != "adamw":
        raise ValueError("Frozen protocol requires AdamW")
    if str(cfg["loss"]["name"]).lower() != "cross_entropy":
        raise ValueError("Frozen protocol requires cross_entropy")
    assert_close(cfg["loss"]["label_smoothing"], 0.1, "loss.label_smoothing")
    if cfg.get("augmentation") != EXPECTED_AUGMENTATION:
        raise ValueError(
            "Frozen augmentation mismatch:\n"
            f"expected={EXPECTED_AUGMENTATION}\nactual={cfg.get('augmentation')}"
        )
    if int(cfg["model"]["num_classes"]) != 3:
        raise ValueError("Frozen protocol requires 3 classes")
    if str(cfg["model"]["vit_name"]) != "deit_small_patch16_224":
        raise ValueError("Frozen protocol requires deit_small_patch16_224 in the shared schema")
    if model == "vit":
        assert_close(cfg["model"].get("drop_rate"), 0.1, "model.drop_rate")
        assert_close(cfg["model"].get("drop_path_rate"), 0.1, "model.drop_path_rate")


def generated_config_path(fold: int, model: str) -> Path:
    return CONFIG_ROOT / f"fold{fold}_{model}.yaml"


def run_output_dir(fold: int, model: str) -> Path:
    return OUTPUT_ROOT / MODEL_OUTPUT_NAMES[model] / f"fold{fold}"


def evaluation_json_path(fold: int, model: str) -> Path:
    return run_output_dir(fold, model) / "metrics" / (
        f"{model}_eval_val_patient_top_k_confidence_k5_best_loss.json"
    )


def patient_csv_path(fold: int, model: str) -> Path:
    return run_output_dir(fold, model) / "metrics" / (
        f"{model}_eval_val_patient_top_k_confidence_k5_best_loss_patients.csv"
    )


def write_resolved_config(fold: int, model: str) -> Path:
    base_path = BASE_CONFIGS[model]
    cfg = load_yaml(base_path)
    validate_frozen_base_config(cfg, model)

    fold_data_dir = DATA_ROOT / f"fold{fold}"
    if not (fold_data_dir / "train").is_dir() or not (fold_data_dir / "val").is_dir():
        raise FileNotFoundError(f"Prepared fold dataset is missing: {fold_data_dir}")

    resolved = deepcopy(cfg)
    resolved["data"]["data_dir"] = relative_posix(fold_data_dir)
    resolved["output"]["output_dir"] = relative_posix(run_output_dir(fold, model))
    resolved["cv"] = {
        "protocol": "frozen_all_axial_patient_cv5_top5",
        "fold": fold,
        "n_splits": 5,
        "split_seed": 1,
        "split_unit": "patient",
        "primary_checkpoint": "loss",
        "primary_aggregate_level": "patient",
        "primary_pooling": "top_k_confidence",
        "top_k": 5,
        "base_config": relative_posix(base_path),
    }

    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    path = generated_config_path(fold, model)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, allow_unicode=True, sort_keys=False)
    return path


def latest_epoch(fold: int, model: str) -> int:
    path = run_output_dir(fold, model) / "logs" / f"{model}_epoch_log.csv"
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return max((int(row["epoch"]) for row in rows), default=0)


def training_complete(fold: int, model: str) -> bool:
    output_dir = run_output_dir(fold, model)
    return (
        latest_epoch(fold, model) >= 30
        and (output_dir / "models" / f"{model}_best_loss.pth").exists()
        and (output_dir / "metrics" / "summary.txt").exists()
    )


def run_command(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def train_and_evaluate(fold: int, model: str, config_path: Path, restart_incomplete: bool) -> None:
    eval_json = evaluation_json_path(fold, model)
    eval_csv = patient_csv_path(fold, model)
    if eval_json.exists() and eval_csv.exists():
        print(f"[SKIP] fold{fold} {model}: patient top-5 evaluation already complete", flush=True)
        return

    output_dir = run_output_dir(fold, model)
    complete = training_complete(fold, model)
    epoch = latest_epoch(fold, model)
    if not complete and output_dir.exists() and any(output_dir.iterdir()):
        if not restart_incomplete:
            raise RuntimeError(
                f"Incomplete run detected for fold{fold} {model} at epoch {epoch}. "
                "src.train cannot safely resume optimizer/scheduler state. "
                "Re-run with --restart-incomplete to delete and restart this run."
            )
        print(f"[RESTART] Removing incomplete fold{fold} {model} output at epoch {epoch}", flush=True)
        shutil.rmtree(output_dir)

    if not complete:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "fold": fold,
            "model": model,
            "status": "training",
            "started_at": datetime.now().astimezone().isoformat(),
            "config": relative_posix(config_path),
            "python": sys.executable,
        }
        with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        run_command(
            [
                sys.executable,
                "-u",
                "-m",
                "src.train",
                "--config",
                relative_posix(config_path),
                "--models",
                model,
            ]
        )

    run_command(
        [
            sys.executable,
            "-u",
            "-m",
            "src.evaluate",
            "--config",
            relative_posix(config_path),
            "--model",
            model,
            "--split",
            "val",
            "--checkpoint-metric",
            "loss",
            "--aggregate-level",
            "patient",
            "--pooling",
            "top_k_confidence",
            "--top-k",
            "5",
        ]
    )

    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "fold": fold,
        "model": model,
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "config": relative_posix(config_path),
        "evaluation_json": relative_posix(eval_json),
        "patient_csv": relative_posix(eval_csv),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def status_for(fold: int, model: str) -> dict[str, Any]:
    eval_json = evaluation_json_path(fold, model)
    eval_csv = patient_csv_path(fold, model)
    return {
        "fold": fold,
        "model": model,
        "latest_epoch": latest_epoch(fold, model),
        "training_complete": training_complete(fold, model),
        "evaluation_complete": eval_json.exists() and eval_csv.exists(),
        "config": relative_posix(generated_config_path(fold, model)),
        "output_dir": relative_posix(run_output_dir(fold, model)),
    }


def write_status(folds: list[int], models: list[str]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [status_for(fold, model) for fold in folds for model in models]
    with (OUTPUT_ROOT / "cv_status.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    folds = list(dict.fromkeys(args.folds))
    models = list(dict.fromkeys(args.models))
    if not folds or any(fold not in range(1, 6) for fold in folds):
        raise ValueError("--folds must contain values from 1 through 5")

    configs: dict[tuple[int, str], Path] = {}
    for fold in folds:
        for model in models:
            configs[(fold, model)] = write_resolved_config(fold, model)
    write_status(folds, models)
    print(f"Generated {len(configs)} frozen CV configs below: {CONFIG_ROOT}", flush=True)

    if args.prepare_only:
        print("Preparation-only mode: training was not started.", flush=True)
        return

    try:
        for fold in folds:
            for model in models:
                print(f"\n=== fold{fold}/5 | {model} ===", flush=True)
                train_and_evaluate(
                    fold=fold,
                    model=model,
                    config_path=configs[(fold, model)],
                    restart_incomplete=args.restart_incomplete,
                )
                write_status(folds, models)
    finally:
        write_status(folds, models)

    print("All requested cross-validation runs are complete.", flush=True)
    if set(folds) == set(range(1, 6)) and set(models) == {"vit", "resnet18"}:
        run_command(
            [
                sys.executable,
                "-u",
                "scripts/summarize_patient_cv.py",
                "--n-bootstrap",
                "10000",
                "--seed",
                "20260714",
            ]
        )


if __name__ == "__main__":
    main()
