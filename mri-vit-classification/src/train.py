import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from src.dataset import build_dataloaders
from src.losses import FocalLoss
from src.model import build_model
from src.utils import (
    compute_class_counts,
    compute_class_weights,
    ensure_dirs,
    load_config,
    save_comparison_csv,
    save_epoch_log,
    save_summary,
    set_seed,
)
from src.visualize import plot_learning_curves, save_vit_attention_map


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device):
    # 1エポック分の学習（順伝播 -> 誤差計算 -> 逆伝播 -> 重み更新）
    model.train()
    running_loss = 0.0
    preds_all, targets_all = [], []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        # 直前バッチの勾配を初期化
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        preds_all.extend(preds.detach().cpu().numpy().tolist())
        targets_all.extend(targets.detach().cpu().numpy().tolist())

    return running_loss / len(loader.dataset), accuracy_score(targets_all, preds_all)


def evaluate(model, loader, criterion, device: torch.device, num_classes: int) -> Dict[str, Any]:
    # 検証処理（重み更新せず、指標だけ計算）
    model.eval()
    running_loss = 0.0
    preds_all, targets_all, probs_all = [], [], []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            preds_all.extend(preds.detach().cpu().numpy().tolist())
            targets_all.extend(targets.detach().cpu().numpy().tolist())
            probs_all.extend(probs.detach().cpu().numpy().tolist())

    f1_average = "binary" if num_classes == 2 else "macro"

    labels = list(range(num_classes))
    cm = confusion_matrix(targets_all, preds_all, labels=labels)

    metrics = {
        "loss": running_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets_all, preds_all),
        "f1": f1_score(targets_all, preds_all, average=f1_average, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
    }

    try:
        if num_classes == 2:
            positive_probs = [row[1] for row in probs_all]
            metrics["roc_auc"] = roc_auc_score(targets_all, positive_probs)
        else:
            metrics["roc_auc"] = roc_auc_score(targets_all, probs_all, multi_class="ovr", average="macro")
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def save_confusion_matrix_csv(cm: List[List[int]], labels: List[int], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [str(label) for label in labels])
        for label, row in zip(labels, cm):
            writer.writerow([str(label)] + row)


def run_training(model_name: str, cfg: Dict[str, Any], dataloaders, device: torch.device) -> Dict[str, Any]:
    # 設定からモデルを構築して実行デバイスへ配置
    model = build_model(
        model_name=model_name,
        num_classes=cfg["model"]["num_classes"],
        vit_name=cfg["model"]["vit_name"],
        image_size=cfg["data"]["image_size"],
    ).to(device)

    # 損失関数と最適化手法を定義
    loss_cfg = cfg.get("loss", {})
    loss_name = loss_cfg.get("name", "cross_entropy")
    class_weighting = loss_cfg.get("class_weighting", "none")
    effective_beta = float(loss_cfg.get("effective_beta", 0.9999))
    power = float(loss_cfg.get("power", 1.0))
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))

    class_weights = None
    manual_weights = loss_cfg.get("class_weights")
    if manual_weights is not None:
        if len(manual_weights) != cfg["model"]["num_classes"]:
            raise ValueError("class_weights length must match num_classes")
        class_weights = torch.tensor(manual_weights, dtype=torch.float, device=device)
    elif class_weighting != "none":
        train_targets = dataloaders["train"].dataset.targets
        class_weights = compute_class_weights(
            train_targets,
            cfg["model"]["num_classes"],
            method=class_weighting,
            effective_beta=effective_beta,
            power=power,
        ).to(device)

    if loss_name == "focal":
        focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
        criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
    elif loss_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unsupported loss name: {loss_name}")
    optimizer_cfg = cfg.get("train", {}).get("optimizer", {})
    optimizer_name = str(optimizer_cfg.get("name", "adamw")).lower()
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(optimizer_cfg.get("momentum", 0.0)),
            alpha=float(optimizer_cfg.get("alpha", 0.99)),
            eps=float(optimizer_cfg.get("eps", 1e-8)),
            centered=bool(optimizer_cfg.get("centered", False)),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    lr_decay = float(cfg.get("train", {}).get("lr_decay", 0.0))
    scheduler = None
    if lr_decay > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0 / (1.0 + lr_decay * epoch),
        )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_roc_auc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_metric_name = cfg.get("train", {}).get("best_metric", "loss")
    best_metric_value = float("inf") if best_metric_name == "loss" else -float("inf")
    save_cm = bool(cfg.get("train", {}).get("save_confusion_matrix", False))
    out_dir = Path(cfg["output"]["output_dir"])
    best_path = out_dir / "models" / f"{model_name}_best.pth"

    # エポックごとに学習・検証を繰り返す
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val = evaluate(model, dataloaders["val"], criterion, device, num_classes=cfg["model"]["num_classes"])

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["val_f1"].append(val["f1"])
        history["val_roc_auc"].append(val["roc_auc"])

        print(
            f"[{model_name}] Epoch {epoch:02d}/{cfg['train']['epochs']} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} val_f1={val['f1']:.4f} val_auc={val['roc_auc']:.4f}"
        )

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]

        if best_metric_name == "loss":
            current_metric = val["loss"]
            is_better = current_metric < best_metric_value
        elif best_metric_name == "accuracy":
            current_metric = val["accuracy"]
            is_better = current_metric > best_metric_value
        elif best_metric_name == "f1":
            current_metric = val["f1"]
            is_better = current_metric > best_metric_value
        elif best_metric_name == "roc_auc":
            current_metric = val["roc_auc"]
            is_better = current_metric > best_metric_value
        else:
            raise ValueError(f"Unsupported best_metric: {best_metric_name}")

        if is_better:
            best_metric_value = current_metric
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        if save_cm:
            cm_path = out_dir / "metrics" / f"{model_name}_confusion_matrix_epoch{epoch:02d}.csv"
            save_confusion_matrix_csv(val["confusion_matrix"], val["confusion_matrix_labels"], cm_path)

        if scheduler is not None:
            scheduler.step()

    save_epoch_log(history, out_dir / "logs" / f"{model_name}_epoch_log.csv")
    plot_learning_curves(history, title=model_name, output_path=out_dir / "figures" / f"{model_name}_learning_curves.png")

    # ベスト重みを再読み込みして最終検証指標を計算
    model.load_state_dict(torch.load(best_path, map_location=device))
    final_metrics = evaluate(model, dataloaders["val"], criterion, device, num_classes=cfg["model"]["num_classes"])

    # ViT は注意マップ画像も保存（モデル構造に依存するため失敗時はスキップ）
    if model_name == "vit":
        images, _ = next(iter(dataloaders["val"]))
        _ = save_vit_attention_map(
            model=model,
            image_batch=images[:1].to(device),
            output_path=out_dir / "figures" / "vit_attention_map.png",
            mean=cfg["data"]["mean"],
            std=cfg["data"]["std"],
        )

    return {
        "model": model_name,
        "accuracy": final_metrics["accuracy"],
        "f1": final_metrics["f1"],
        "roc_auc": final_metrics["roc_auc"],
        "best_val_loss": best_val_loss,
        "best_metric": best_metric_name,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
    }


def main():
    # 1) 設定読み込み 2) データ準備 3) 学習実行 4) 結果保存
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--models", nargs="+", default=["vit", "resnet18"], choices=["vit", "resnet18"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dirs(cfg["output"]["output_dir"])

    # GPU が使える場合は CUDA を利用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataloaders, _, class_to_idx = build_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
        augmentation=cfg.get("augmentation"),
        sampler=cfg.get("train", {}).get("sampler"),
    )
    print("class_to_idx:", class_to_idx)

    train_counts = compute_class_counts(dataloaders["train"].dataset.targets, cfg["model"]["num_classes"])
    print("class_counts:", train_counts.tolist())

    rows: List[Dict[str, Any]] = []
    for model_name in args.models:
        rows.append(run_training(model_name, cfg, dataloaders, device))

    # モデル比較結果を CSV / テキストで出力
    out_dir = Path(cfg["output"]["output_dir"])
    save_comparison_csv(rows, out_dir / "metrics" / "comparison_metrics.csv")
    save_summary(rows, out_dir / "metrics" / "summary.txt")


if __name__ == "__main__":
    main()
