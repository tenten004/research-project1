import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import load_config, save_json


def evaluate_once(model, loader, device: torch.device):
    model.eval()
    preds_all, targets_all, probs_all = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds.detach().cpu().numpy().tolist())
            targets_all.extend(targets.detach().cpu().numpy().tolist())
            probs_all.extend(probs.detach().cpu().numpy().tolist())

    result = {
        "accuracy": accuracy_score(targets_all, preds_all),
        "f1": f1_score(targets_all, preds_all, zero_division=0),
    }
    try:
        result["roc_auc"] = roc_auc_score(targets_all, probs_all)
    except ValueError:
        result["roc_auc"] = float("nan")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, default="vit", choices=["vit", "resnet18"])
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

    metrics = evaluate_once(model, dataloaders["val"], device)
    print(metrics)

    out_path = Path(cfg["output"]["output_dir"]) / "metrics" / f"{args.model}_eval.json"
    save_json(metrics, out_path)


if __name__ == "__main__":
    main()
