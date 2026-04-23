import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import ensure_dirs, load_config, save_comparison_csv, save_epoch_log, save_summary, set_seed
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


def evaluate(model, loader, criterion, device: torch.device, num_classes: int) -> Dict[str, float]:
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

    metrics = {
        "loss": running_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets_all, preds_all),
        "f1": f1_score(targets_all, preds_all, average=f1_average, zero_division=0),
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


def run_training(model_name: str, cfg: Dict[str, Any], dataloaders, device: torch.device) -> Dict[str, Any]:
    # 設定からモデルを構築して実行デバイスへ配置
    model = build_model(
        model_name=model_name,
        num_classes=cfg["model"]["num_classes"],
        vit_name=cfg["model"]["vit_name"],
    ).to(device)

    # 損失関数と最適化手法を定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_roc_auc": []}
    best_val_loss = float("inf")
    best_epoch = 0
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

        # 検証損失が最小の重みだけを保存
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

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
    )
    print("class_to_idx:", class_to_idx)

    rows: List[Dict[str, Any]] = []
    for model_name in args.models:
        rows.append(run_training(model_name, cfg, dataloaders, device))

    # モデル比較結果を CSV / テキストで出力
    out_dir = Path(cfg["output"]["output_dir"])
    save_comparison_csv(rows, out_dir / "metrics" / "comparison_metrics.csv")
    save_summary(rows, out_dir / "metrics" / "summary.txt")


if __name__ == "__main__":
    main()
