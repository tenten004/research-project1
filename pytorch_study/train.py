import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from dataset import build_dataloaders
from model import build_model
from utils import (
    evaluate,
    plot_learning_curves,
    save_comparison_csv,
    save_epoch_log,
    save_text_summary,
    set_seed,
    train_one_epoch,
    visualize_vit_attention_map,
)


def train_model(model_name: str, cfg: Config, dataloaders, device: torch.device) -> Dict[str, Any]:
    # モデルを作成し、CPU/GPU の実行デバイスに配置
    model = build_model(
        model_name=model_name,
        num_classes=cfg.num_classes,
        vit_model_name=cfg.vit_name,
    ).to(device)

    # 分類タスクで一般的な損失関数と最適化手法
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # エポックごとの学習履歴（後でCSV保存・可視化に利用）
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_roc_auc": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_path = Path(cfg.output_dir) / "models" / f"{model_name}_best.pth"

    # 1エポックごとに「学習 -> 検証」を実施
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])

        print(
            f"[{model_name}] Epoch {epoch:02d}/{cfg.epochs} "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['roc_auc']:.4f}"
        )

        # 検証損失が最も良かった重みのみ保存
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

    epoch_log_path = Path(cfg.output_dir) / "logs" / f"{model_name}_epoch_log.csv"
    save_epoch_log(history=history, output_csv_path=epoch_log_path)

    plot_path = Path(cfg.output_dir) / "plots" / f"{model_name}_learning_curves.png"
    plot_learning_curves(history=history, title=model_name, output_path=plot_path)

    # ベスト重みを読み直して最終評価値を計算
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_metrics = evaluate(
        model=model,
        dataloader=dataloaders["val"],
        criterion=criterion,
        device=device,
    )

    if model_name == "vit":
        sample_images, _ = next(iter(dataloaders["val"]))
        sample_images = sample_images.to(device)
        attention_path = Path(cfg.output_dir) / "attention" / "vit_attention_map.png"
        ok = visualize_vit_attention_map(
            model=model,
            image_batch=sample_images[:1],
            output_path=attention_path,
            mean=cfg.mean,
            std=cfg.std,
        )
        if not ok:
            print("[vit] Attention map visualization was skipped (model internals not compatible).")

    result = {
        "model": model_name,
        "accuracy": final_metrics["accuracy"],
        "f1": final_metrics["f1"],
        "roc_auc": final_metrics["roc_auc"],
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    return result


def parse_args() -> argparse.Namespace:
    # コマンドライン引数から実験条件を受け取る
    parser = argparse.ArgumentParser(description="Train ViT and ResNet18 for binary MRI classification")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["vit", "resnet18"],
        choices=["vit", "resnet18"],
        help="Models to train under the same setting",
    )
    return parser.parse_args()


def main() -> None:
    # 全体の流れ: 引数解析 -> 設定作成 -> データ準備 -> 学習 -> 結果保存
    args = parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    cfg.ensure_output_dirs()
    set_seed(cfg.seed)

    # GPU が利用可能なら CUDA を使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, _ = build_dataloaders(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        mean=cfg.mean,
        std=cfg.std,
    )

    results = []
    for model_name in args.models:
        print(f"\n=== Training: {model_name} ===")
        result = train_model(model_name=model_name, cfg=cfg, dataloaders=dataloaders, device=device)
        results.append(result)

    # 比較結果をCSVとテキストに保存
    comparison_csv_path = Path(cfg.output_dir) / "metrics" / "comparison_metrics.csv"
    save_comparison_csv(results=results, output_csv_path=comparison_csv_path)

    summary_txt_path = Path(cfg.output_dir) / "metrics" / "summary.txt"
    save_text_summary(results=results, output_txt_path=summary_txt_path)

    print("\nTraining finished. Artifacts saved under:", cfg.output_dir)


if __name__ == "__main__":
    main()
