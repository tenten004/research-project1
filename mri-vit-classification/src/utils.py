import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    # YAML 設定ファイルを辞書として読み込む
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(base_output_dir: str) -> None:
    # 出力で使うフォルダを事前作成（なければ作る）
    base = Path(base_output_dir)
    (base / "models").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / "metrics").mkdir(parents=True, exist_ok=True)
    (base / "figures").mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    # 再現性を高めるため乱数シードを固定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_counts(targets: List[int], num_classes: int) -> torch.Tensor:
    # クラスごとのサンプル数を集計
    if not targets:
        return torch.zeros(num_classes, dtype=torch.float)
    counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    return counts


def compute_class_weights_from_counts(
    counts: torch.Tensor,
    method: str = "inverse",
    effective_beta: float = 0.9999,
    power: float = 1.0,
) -> torch.Tensor:
    # クラスごとの重みを計算
    if method == "none":
        return torch.ones_like(counts)

    weights = torch.zeros_like(counts)
    valid = counts > 0

    if method == "inverse":
        weights[valid] = counts[valid].pow(-power)
    elif method == "effective":
        beta = effective_beta
        weights[valid] = (1.0 - beta) / (1.0 - torch.pow(beta, counts[valid]))
    else:
        raise ValueError(f"Unsupported class weight method: {method}")

    if weights.sum() > 0:
        weights = weights / weights.sum() * len(weights)
    return weights


def compute_class_weights(
    targets: List[int],
    num_classes: int,
    method: str = "inverse",
    effective_beta: float = 0.9999,
    power: float = 1.0,
) -> torch.Tensor:
    # ラベル一覧からクラス重みを計算
    counts = compute_class_counts(targets, num_classes)
    return compute_class_weights_from_counts(counts, method=method, effective_beta=effective_beta, power=power)


def compute_sample_weights(targets: List[int], class_weights: torch.Tensor) -> torch.Tensor:
    # サンプルごとの重みを作成
    weights = torch.tensor([class_weights[t].item() for t in targets], dtype=torch.double)
    return weights


def save_epoch_log(history: Dict[str, List[float]], output_csv: Path) -> None:
    # エポックごとの履歴を CSV に保存
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "val_roc_auc"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        n = len(history["train_loss"])
        for i in range(n):
            writer.writerow(
                [
                    i + 1,
                    history["train_loss"][i],
                    history["train_acc"][i],
                    history["val_loss"][i],
                    history["val_acc"][i],
                    history["val_f1"][i],
                    history["val_roc_auc"][i],
                ]
            )


def save_comparison_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    # モデル比較結果を1つの表にまとめる
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "model",
        "accuracy",
        "f1",
        "roc_auc",
        "best_val_loss",
        "best_metric",
        "best_metric_value",
        "best_epoch",
    ]

    extra_fields: List[str] = []
    known = set(base_fields)
    for row in rows:
        for key in row.keys():
            if key not in known:
                known.add(key)
                extra_fields.append(key)

    fields = base_fields + extra_fields

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary(rows: List[Dict[str, Any]], output_txt: Path) -> None:
    # 人が読みやすい要約をテキストで保存
    def _fmt_float(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _fmt_int(value: Any) -> str:
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as f:
        f.write("Model Comparison Results\n")
        f.write("=" * 32 + "\n")
        for row in rows:
            f.write(f"Model: {row['model']}\n")
            f.write(f"  Accuracy: {_fmt_float(row['accuracy'])}\n")
            f.write(f"  F1-score: {_fmt_float(row['f1'])}\n")
            f.write(f"  ROC-AUC: {_fmt_float(row['roc_auc'])}\n")
            f.write(f"  Best Val Loss: {_fmt_float(row['best_val_loss'])}\n")
            if row.get("best_metric") is not None:
                f.write(f"  Best Metric: {row['best_metric']}\n")
                if row.get("best_metric_value") is not None:
                    f.write(f"  Best Metric Value: {_fmt_float(row['best_metric_value'])}\n")
            f.write(f"  Best Epoch: {_fmt_int(row['best_epoch'])}\n")

            if row.get("peak_val_acc") is not None and row.get("peak_val_acc_epoch") is not None:
                f.write(
                    f"  Peak Val Accuracy: {_fmt_float(row['peak_val_acc'])} (epoch {_fmt_int(row['peak_val_acc_epoch'])})\n"
                )
            if row.get("peak_val_f1") is not None and row.get("peak_val_f1_epoch") is not None:
                f.write(f"  Peak Val F1: {_fmt_float(row['peak_val_f1'])} (epoch {_fmt_int(row['peak_val_f1_epoch'])})\n")
            if row.get("peak_val_roc_auc") is not None and row.get("peak_val_roc_auc_epoch") is not None:
                f.write(
                    f"  Peak Val ROC-AUC: {_fmt_float(row['peak_val_roc_auc'])}"
                    f" (epoch {_fmt_int(row['peak_val_roc_auc_epoch'])})\n"
                )
            if row.get("metric_checkpoint_csv"):
                f.write(f"  Metric Checkpoint CSV: {row['metric_checkpoint_csv']}\n")

            f.write("-" * 32 + "\n")


def save_json(data: Dict[str, Any], output_json: Path) -> None:
    # 評価指標などをJSONで保存
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
