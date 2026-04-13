from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_learning_curves(history: Dict[str, List[float]], title: str, output_path: Path) -> None:
    # 学習曲線（Loss / Accuracy）を画像として保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_title(f"{title} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_vit_attention_map(
    model,
    image_batch: torch.Tensor,
    output_path: Path,
    mean: float,
    std: float,
) -> bool:
    """ViT 最終ブロックの CLS->Patch 注意をヒートマップとして保存する。"""
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        return False

    bucket = {}

    def hook_fn(_module, _inputs, output):
        bucket["attn"] = output.detach().cpu()

    try:
        handle = model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)
    except AttributeError:
        return False

    model.eval()
    with torch.no_grad():
        _ = model(image_batch)
    handle.remove()

    if "attn" not in bucket:
        return False

    attn = bucket["attn"]
    # 複数ヘッドを平均して、1枚の注意マップに集約
    attn = attn[0].mean(dim=0)
    cls_to_patch = attn[0, 1:]
    n = cls_to_patch.shape[0]
    side = int(np.sqrt(n))
    if side * side != n:
        return False

    heatmap = cls_to_patch.reshape(side, side).numpy()
    # 描画しやすいように 0-1 に正規化
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    img = image_batch[0].detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img * std + mean, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(img)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.5, extent=[0, img.shape[1], img.shape[0], 0])
    axes[1].set_title("ViT Attention")
    axes[1].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True
