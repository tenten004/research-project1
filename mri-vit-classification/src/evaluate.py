import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import load_config, save_json


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

    f1_average = "binary" if num_classes == 2 else "macro"

    result = {
        "accuracy": accuracy_score(targets_all, preds_all),
        "f1": f1_score(targets_all, preds_all, average=f1_average, zero_division=0),
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


def main():
    # 設定・モデル・データを読み込み、指定 split で評価する
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, default="vit", choices=["vit", "resnet18"])
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, _, _ = build_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
    )

    model = build_model(args.model, cfg["model"]["num_classes"], cfg["model"]["vit_name"]).to(device)
    ckpt_path = Path(cfg["output"]["output_dir"]) / "models" / f"{args.model}_best.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    criterion = nn.CrossEntropyLoss()
    _ = criterion  # reserved for extension if loss logging is needed

    if args.split not in dataloaders:
        raise ValueError(
            f"Requested split '{args.split}' is not available. "
            "Create data/processed/test with class folders or use --split val."
        )

    # 評価結果を表示し、JSONでも保存して再利用しやすくする
    metrics = evaluate_once(model, dataloaders[args.split], device, num_classes=cfg["model"]["num_classes"])
    print(metrics)

    out_path = Path(cfg["output"]["output_dir"]) / "metrics" / f"{args.model}_eval_{args.split}.json"
    save_json(metrics, out_path)


if __name__ == "__main__":
    main()
