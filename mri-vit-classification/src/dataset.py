from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _has_class_subdirs(root: Path) -> bool:
    # test などが存在しない場合に安全に分岐するための判定
    return root.exists() and any(p.is_dir() for p in root.iterdir())


def build_transforms(image_size: int, mean: float, std: float) -> Tuple[transforms.Compose, transforms.Compose]:
    # 学習時は軽い拡張を入れて汎化性能を高める
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std]),
        ]
    )
    # 検証時は拡張なしで、モデルの純粋な性能を測る
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
                train/val（必要ならtest）の DataLoader を作成する。

        Expected structure:
      data_dir/
        train/class0, class1
        val/class0, class1
                test/class0, class1 (optional)
    """
    train_tf, val_tf = build_transforms(image_size=image_size, mean=mean, std=std)

    # ImageFolder は「クラス名フォルダ配下の画像」を自動でラベル化して読み込む
    train_root = Path(data_dir) / "train"
    val_root = Path(data_dir) / "val"
    test_root = Path(data_dir) / "test"

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
    # データ件数は平均損失計算やログ表示に使う
    sizes = {"train": len(train_ds), "val": len(val_ds)}

    # test は存在する場合のみ有効化
    if _has_class_subdirs(test_root):
        test_ds = datasets.ImageFolder(root=str(test_root), transform=val_tf)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        dataloaders["test"] = test_loader
        sizes["test"] = len(test_ds)

    # クラス名と数値ラベルの対応（例: {"class0": 0, "class1": 1}）
    class_to_idx = train_ds.class_to_idx
    return dataloaders, sizes, class_to_idx
