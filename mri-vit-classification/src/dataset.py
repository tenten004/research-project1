from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int, mean: float, std: float) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std]),
        ]
    )
    return train_transform, val_transform


def build_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    mean: float,
    std: float,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], Dict[str, int]]:
    """
    Expected structure:
      data_dir/
        train/class0, class1
        val/class0, class1
    """
    train_tf, val_tf = build_transforms(image_size=image_size, mean=mean, std=std)

    train_root = Path(data_dir) / "train"
    val_root = Path(data_dir) / "val"

    train_ds = datasets.ImageFolder(root=str(train_root), transform=train_tf)
    val_ds = datasets.ImageFolder(root=str(val_root), transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dataloaders = {"train": train_loader, "val": val_loader}
    sizes = {"train": len(train_ds), "val": len(val_ds)}
    class_to_idx = train_ds.class_to_idx
    return dataloaders, sizes, class_to_idx
