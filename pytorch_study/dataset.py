from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int = 224, mean: float = 0.5, std: float = 0.5) -> Tuple[transforms.Compose, transforms.Compose]:
    # 学習時は軽いデータ拡張を入れて汎化性能を向上
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std]),
        ]
    )

    # 検証時は拡張なしで、純粋な性能を評価
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
        train/val 用の DataLoader を作成する。

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

    # ImageFolder はクラスごとのフォルダ構造から自動でラベルを付与する
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
    # データ件数は損失平均やログ確認に使用
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    return dataloaders, dataset_sizes
