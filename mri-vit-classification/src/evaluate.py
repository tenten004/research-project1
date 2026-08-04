import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torchvision import datasets

from src.dataset import build_dataloaders, build_transforms
from src.model import build_model
from src.utils import load_config, save_json


class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return sample, target, path


def _compute_metrics(targets_all: List[int], preds_all: List[int], probs_all: List[List[float]], num_classes: int) -> Dict[str, Any]:
    f1_average = "binary" if num_classes == 2 else "macro"

    labels = list(range(num_classes))
    cm = confusion_matrix(targets_all, preds_all, labels=labels)

    result: Dict[str, Any] = {
        "accuracy": accuracy_score(targets_all, preds_all),
        "f1": f1_score(targets_all, preds_all, average=f1_average, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
    }

    try:
        if num_classes == 2:
            positive_probs = [row[1] for row in probs_all]
            result["roc_auc"] = roc_auc_score(targets_all, positive_probs)
        else:
            result["roc_auc"] = roc_auc_score(targets_all, probs_all, multi_class="ovr", average="macro")
    except ValueError:
        result["roc_auc"] = float("nan")
    return result


def _extract_patient_id(path_str: str) -> str:
    filename = Path(path_str).name
    if "_" in filename:
        return filename.split("_", 1)[0]
    return Path(path_str).stem


def _extract_modality(path_str: str) -> str:
    parts = Path(path_str).name.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def _extract_slice_index(path_str: str) -> str:
    # Dataset preparation appends "-<random>" for name collision avoidance.
    # The original axial index is the last underscore-delimited token before it.
    original_stem = Path(path_str).stem.rsplit("-", 1)[0]
    axial_token = original_stem.rsplit("_", 1)[-1]
    return str(int(axial_token)) if axial_token.isdigit() else "unknown"


def _softmax_weights(scores: List[float], temperature: float) -> List[float]:
    safe_temp = max(temperature, 1e-6)
    scaled = [s / safe_temp for s in scores]
    max_scaled = max(scaled)
    exps = [math.exp(s - max_scaled) for s in scaled]
    denom = sum(exps)
    if denom <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [v / denom for v in exps]


def _top_confidence_indices(slice_probs: List[List[float]], top_k: int) -> List[int]:
    k = min(max(int(top_k), 1), len(slice_probs))
    return sorted(range(len(slice_probs)), key=lambda i: max(slice_probs[i]), reverse=True)[:k]


def _aggregate_probs(
    slice_probs: List[List[float]],
    pooling: str,
    temperature: float,
    top_k: int = 3,
) -> List[float]:
    if not slice_probs:
        return []
    num_classes = len(slice_probs[0])

    if pooling == "mean":
        n = len(slice_probs)
        return [sum(p[i] for p in slice_probs) / n for i in range(num_classes)]

    if pooling == "max_confidence":
        best = max(slice_probs, key=lambda p: max(p))
        return list(best)

    if pooling == "top_k_confidence":
        indices = _top_confidence_indices(slice_probs, top_k=top_k)
        return [sum(slice_probs[idx][i] for idx in indices) / len(indices) for i in range(num_classes)]

    if pooling == "attention_confidence":
        scores = [max(p) for p in slice_probs]
    elif pooling == "attention_entropy":
        eps = 1e-8
        scores = [sum(pi * math.log(max(pi, eps)) for pi in p) for p in slice_probs]
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")

    weights = _softmax_weights(scores, temperature=temperature)
    return [sum(w * p[i] for w, p in zip(weights, slice_probs)) for i in range(num_classes)]


def _save_patient_predictions(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_once(model, loader, device: torch.device, num_classes: int):
    # 学習済みモデルで1回評価し、主要指標を返す
    model.eval()
    preds_all, targets_all, probs_all = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds.detach().cpu().numpy().tolist())
            targets_all.extend(targets.detach().cpu().numpy().tolist())
            probs_all.extend(probs.detach().cpu().numpy().tolist())

    return _compute_metrics(targets_all=targets_all, preds_all=preds_all, probs_all=probs_all, num_classes=num_classes)


def evaluate_patient_level(
    model,
    loader,
    device: torch.device,
    num_classes: int,
    pooling: str,
    temperature: float,
    top_k: int = 3,
):
    model.eval()
    patient_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with torch.no_grad():
        for images, targets, paths in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy().tolist()
            targets_list = targets.detach().cpu().numpy().tolist()

            for target, prob, path in zip(targets_list, probs, paths):
                patient_id = _extract_patient_id(path)
                patient_to_rows[patient_id].append({"target": int(target), "probs": prob, "path": str(path)})

    patient_targets: List[int] = []
    patient_preds: List[int] = []
    patient_probs: List[List[float]] = []
    patient_rows: List[Dict[str, Any]] = []

    for patient_id in sorted(patient_to_rows.keys()):
        rows = patient_to_rows[patient_id]
        labels = [r["target"] for r in rows]
        target = Counter(labels).most_common(1)[0][0]
        probs = [r["probs"] for r in rows]
        agg_probs = _aggregate_probs(
            slice_probs=probs,
            pooling=pooling,
            temperature=temperature,
            top_k=top_k,
        )
        pred = int(max(range(len(agg_probs)), key=lambda i: agg_probs[i]))

        selected_indices: List[int] = []
        if pooling == "max_confidence":
            selected_indices = _top_confidence_indices(probs, top_k=1)
        elif pooling == "top_k_confidence":
            selected_indices = _top_confidence_indices(probs, top_k=top_k)

        selected_paths = [str(rows[idx]["path"]) for idx in selected_indices]

        patient_targets.append(target)
        patient_preds.append(pred)
        patient_probs.append(agg_probs)
        patient_rows.append(
            {
                "patient_id": patient_id,
                "target": target,
                "pred": pred,
                "num_slices": len(rows),
                "max_prob": max(agg_probs),
                **{f"prob_class{class_idx}": prob for class_idx, prob in enumerate(agg_probs)},
                "selected_paths": ";".join(selected_paths),
                "selected_modalities": ";".join(_extract_modality(path) for path in selected_paths),
                "selected_slice_indices": ";".join(_extract_slice_index(path) for path in selected_paths),
                "selected_confidences": ";".join(f"{max(probs[idx]):.8f}" for idx in selected_indices),
            }
        )

    metrics = _compute_metrics(
        targets_all=patient_targets,
        preds_all=patient_preds,
        probs_all=patient_probs,
        num_classes=num_classes,
    )
    metrics["num_patients"] = len(patient_rows)
    metrics["top_k"] = top_k if pooling == "top_k_confidence" else (1 if pooling == "max_confidence" else 0)
    return metrics, patient_rows


def main():
    # 設定・モデル・データを読み込み、指定 split で評価する
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, default="vit", choices=["vit", "resnet18"])
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="primary",
        choices=["primary", "loss", "accuracy", "f1", "roc_auc"],
        help="Which best checkpoint to evaluate.",
    )
    parser.add_argument(
        "--aggregate-level",
        type=str,
        default="slice",
        choices=["slice", "patient"],
        help="Evaluate per-slice (default) or aggregate per-patient.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max_confidence", "top_k_confidence", "attention_confidence", "attention_entropy"],
        help="Pooling method when --aggregate-level patient is used.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=3, help="Number of slices used by top_k_confidence pooling.")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders = None
    eval_loader = None
    if args.aggregate_level == "slice":
        dataloaders, _, _ = build_dataloaders(
            data_dir=cfg["data"]["data_dir"],
            image_size=cfg["data"]["image_size"],
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            mean=cfg["data"]["mean"],
            std=cfg["data"]["std"],
            augmentation=cfg.get("augmentation"),
            sampler={"type": "none"},
        )
    else:
        _train_tf, eval_tf = build_transforms(
            image_size=cfg["data"]["image_size"],
            mean=cfg["data"]["mean"],
            std=cfg["data"]["std"],
            augmentation=cfg.get("augmentation"),
        )
        split_root = Path(cfg["data"]["data_dir"]) / args.split
        if not split_root.exists():
            raise ValueError(f"Requested split '{args.split}' is not available at: {split_root}")

        eval_ds = ImageFolderWithPath(root=str(split_root), transform=eval_tf)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

    model = build_model(
        args.model,
        cfg["model"]["num_classes"],
        cfg["model"]["vit_name"],
        image_size=cfg["data"]["image_size"],
    ).to(device)
    model_dir = Path(cfg["output"]["output_dir"]) / "models"
    if args.checkpoint_metric == "primary":
        ckpt_path = model_dir / f"{args.model}_best.pth"
    else:
        ckpt_path = model_dir / f"{args.model}_best_{args.checkpoint_metric}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    criterion = nn.CrossEntropyLoss()
    _ = criterion  # reserved for extension if loss logging is needed

    # 評価結果を表示し、JSONでも保存して再利用しやすくする
    if args.aggregate_level == "slice":
        if dataloaders is None or args.split not in dataloaders:
            raise ValueError(
                f"Requested split '{args.split}' is not available. "
                "Create data/processed/test with class folders or use --split val."
            )
        metrics = evaluate_once(model, dataloaders[args.split], device, num_classes=cfg["model"]["num_classes"])
        patient_rows: List[Dict[str, Any]] = []
    else:
        if eval_loader is None:
            raise RuntimeError("Patient-level evaluation loader was not created.")
        metrics, patient_rows = evaluate_patient_level(
            model=model,
            loader=eval_loader,
            device=device,
            num_classes=cfg["model"]["num_classes"],
            pooling=args.pooling,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    metrics["aggregate_level"] = args.aggregate_level
    metrics["pooling"] = args.pooling if args.aggregate_level == "patient" else "none"
    metrics["checkpoint_metric"] = args.checkpoint_metric
    metrics["checkpoint_path"] = str(ckpt_path)
    print(metrics)

    suffix_parts = [args.model, "eval", args.split, args.aggregate_level]
    if args.aggregate_level == "patient":
        suffix_parts.append(args.pooling)
        if args.pooling == "top_k_confidence":
            suffix_parts.append(f"k{args.top_k}")
    if args.checkpoint_metric != "primary":
        suffix_parts.append(f"best_{args.checkpoint_metric}")
    suffix = "_".join(suffix_parts)

    out_path = Path(cfg["output"]["output_dir"]) / "metrics" / f"{suffix}.json"
    save_json(metrics, out_path)

    if args.aggregate_level == "patient":
        patient_csv = Path(cfg["output"]["output_dir"]) / "metrics" / f"{suffix}_patients.csv"
        _save_patient_predictions(patient_rows, patient_csv)


if __name__ == "__main__":
    main()
