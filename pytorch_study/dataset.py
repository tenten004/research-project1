from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int = 224, mean: float = 0.5, std: float = 0.5) -> Tuple[transforms.Compose, transforms.Compose]:
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
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    mean: float = 0.5,
    std: float = 0.5,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    """
    Expected structure:
        dataset/
          train/
            class0/
            class1/
          val/
            class0/
            class1/
    """
    train_transform, val_transform = build_transforms(image_size=image_size, mean=mean, std=std)

    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    return dataloaders, dataset_sizes
