import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())

    loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "loss": loss,
        "accuracy": acc,
        "f1": f1,
        "roc_auc": auc,
    }


def save_epoch_log(history: Dict[str, List[float]], output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "val_roc_auc"]

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        total_epochs = len(history["train_loss"])
        for i in range(total_epochs):
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


def save_comparison_csv(results: List[Dict[str, Any]], output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["model", "accuracy", "f1", "roc_auc", "best_val_loss", "best_epoch"]

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def plot_learning_curves(history: Dict[str, List[float]], title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_text_summary(results: List[Dict[str, Any]], output_txt_path: Path) -> None:
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Model Comparison Results\n", "=" * 40 + "\n"]
    for row in results:
        lines.append(f"Model: {row['model']}\n")
        lines.append(f"  Accuracy: {row['accuracy']:.4f}\n")
        lines.append(f"  F1-score: {row['f1']:.4f}\n")
        lines.append(f"  ROC-AUC: {row['roc_auc']:.4f}\n")
        lines.append(f"  Best Val Loss: {row['best_val_loss']:.4f}\n")
        lines.append(f"  Best Epoch: {row['best_epoch']}\n")
        lines.append("-" * 40 + "\n")

    winner = sorted(results, key=lambda x: (x["roc_auc"], x["f1"], x["accuracy"]), reverse=True)[0]
    lines.append(f"Best overall (by ROC-AUC/F1/Acc): {winner['model']}\n")

    with output_txt_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def visualize_vit_attention_map(
    model,
    image_batch: torch.Tensor,
    output_path: Path,
    mean: float = 0.5,
    std: float = 0.5,
) -> bool:
    """
    Save an approximate attention map from the last ViT block.
    Returns False if the expected timm ViT internals are not found.
    """
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        return False

    attn_container = {}

    def hook_fn(_module, _inputs, output):
        attn_container["attn"] = output.detach().cpu()

    try:
        handle = model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)
    except AttributeError:
        return False

    model.eval()
    with torch.no_grad():
        _ = model(image_batch)
    handle.remove()

    if "attn" not in attn_container:
        return False

    attn = attn_container["attn"]
    # [B, heads, tokens, tokens] -> first sample
    attn = attn[0].mean(dim=0)
    cls_to_patch = attn[0, 1:]

    num_patches = cls_to_patch.shape[0]
    side = int(np.sqrt(num_patches))
    if side * side != num_patches:
        return False

    heatmap = cls_to_patch.reshape(side, side).numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    image = image_batch[0].detach().cpu().permute(1, 2, 0).numpy()
    image = image * std + mean
    image = np.clip(image, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.5, extent=[0, image.shape[1], image.shape[0], 0])
    axes[1].set_title("ViT Attention (CLS -> Patch)")
    axes[1].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True
