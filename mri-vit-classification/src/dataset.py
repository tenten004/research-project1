import random
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from src.utils import compute_class_weights, compute_sample_weights


def _has_class_subdirs(root: Path) -> bool:
    # test などが存在しない場合に安全に分岐するための判定
    return root.exists() and any(p.is_dir() for p in root.iterdir())


class ImbalanceAwareImageFolder(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        base_transform: transforms.Compose | None,
        minority_transform: transforms.Compose | None,
        minority_classes: set[int],
    ) -> None:
        super().__init__(root=root, transform=None)
        self.base_transform = base_transform
        self.minority_transform = minority_transform
        self.minority_classes = minority_classes

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if target in self.minority_classes and self.minority_transform is not None:
            sample = self.minority_transform(sample)
        elif self.base_transform is not None:
            sample = self.base_transform(sample)
        return sample, target


class RandomGammaAdjust:
    def __init__(self, gamma_min: float, gamma_max: float) -> None:
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, img):
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return TF.adjust_gamma(img, gamma)


def _build_train_transform(image_size: int, mean: float, std: float, aug_cfg: Dict[str, Any]) -> transforms.Compose:
    enabled = aug_cfg.get("enabled", True)
    if not enabled:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std]),
            ]
        )

    rotation = float(aug_cfg.get("rotation", 10))
    hflip = float(aug_cfg.get("hflip", 0.5))
    vflip = float(aug_cfg.get("vflip", 0.0))
    brightness = float(aug_cfg.get("brightness", 0.0))
    contrast = float(aug_cfg.get("contrast", 0.0))
    gamma_min = float(aug_cfg.get("gamma_min", 1.0))
    gamma_max = float(aug_cfg.get("gamma_max", 1.0))

    if gamma_min > gamma_max:
        gamma_min, gamma_max = gamma_max, gamma_min

    ops = [transforms.Resize((image_size, image_size))]
    if rotation > 0:
        ops.append(transforms.RandomRotation(degrees=rotation))
    if hflip > 0:
        ops.append(transforms.RandomHorizontalFlip(p=hflip))
    if vflip > 0:
        ops.append(transforms.RandomVerticalFlip(p=vflip))
    if brightness > 0 or contrast > 0:
        ops.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))
    if gamma_min != 1.0 or gamma_max != 1.0:
        ops.append(RandomGammaAdjust(gamma_min, gamma_max))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std]),
        ]
    )
    return transforms.Compose(ops)


def _derive_minority_aug(base_aug: Dict[str, Any], minority_cfg: Dict[str, Any]) -> Dict[str, Any]:
    multiplier = float(minority_cfg.get("multiplier", 1.0))
    rotation = float(base_aug.get("rotation", 10)) * multiplier
    hflip = float(base_aug.get("hflip", 0.5))
    vflip = float(base_aug.get("vflip", 0.0))
    brightness = float(base_aug.get("brightness", 0.0)) * multiplier
    contrast = float(base_aug.get("contrast", 0.0)) * multiplier
    gamma_min = float(base_aug.get("gamma_min", 1.0))
    gamma_max = float(base_aug.get("gamma_max", 1.0))
    gamma_min = 1.0 - (1.0 - gamma_min) * multiplier
    gamma_max = 1.0 + (gamma_max - 1.0) * multiplier

    return {
        "enabled": base_aug.get("enabled", True),
        "rotation": minority_cfg.get("rotation", rotation),
        "hflip": minority_cfg.get("hflip", hflip),
        "vflip": minority_cfg.get("vflip", vflip),
        "brightness": minority_cfg.get("brightness", brightness),
        "contrast": minority_cfg.get("contrast", contrast),
        "gamma_min": minority_cfg.get("gamma_min", gamma_min),
        "gamma_max": minority_cfg.get("gamma_max", gamma_max),
    }


def build_transforms(
    image_size: int,
    mean: float,
    std: float,
    augmentation: Dict[str, Any] | None = None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    aug_cfg = augmentation or {}
    # 学習時は拡張を入れて汎化性能を高める
    train_transform = _build_train_transform(image_size=image_size, mean=mean, std=std, aug_cfg=aug_cfg)
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
    augmentation: Dict[str, Any] | None = None,
    sampler: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], Dict[str, int]]:
    """
                train/val（必要ならtest）の DataLoader を作成する。

        Expected structure:
      data_dir/
        train/class0, class1
        val/class0, class1
                test/class0, class1 (optional)
    """
    train_tf, val_tf = build_transforms(image_size=image_size, mean=mean, std=std, augmentation=augmentation)

    # ImageFolder は「クラス名フォルダ配下の画像」を自動でラベル化して読み込む
    train_root = Path(data_dir) / "train"
    val_root = Path(data_dir) / "val"
    test_root = Path(data_dir) / "test"

    aug_cfg = augmentation or {}
    minority_cfg = aug_cfg.get("minority", {})
    minority_classes = {int(c) for c in minority_cfg.get("classes", [])}

    if minority_classes:
        minority_aug = _derive_minority_aug(aug_cfg, minority_cfg)
        minority_tf = _build_train_transform(
            image_size=image_size,
            mean=mean,
            std=std,
            aug_cfg=minority_aug,
        )
        train_ds = ImbalanceAwareImageFolder(
            root=str(train_root),
            base_transform=train_tf,
            minority_transform=minority_tf,
            minority_classes=minority_classes,
        )
    else:
        train_ds = datasets.ImageFolder(root=str(train_root), transform=train_tf)
    val_ds = datasets.ImageFolder(root=str(val_root), transform=val_tf)

    sampler_cfg = sampler or {}
    sampler_type = sampler_cfg.get("type", "none")
    sampler_obj = None
    shuffle = True
    if sampler_type == "weighted":
        num_classes = len(train_ds.classes)
        method = sampler_cfg.get("method", "inverse")
        effective_beta = float(sampler_cfg.get("effective_beta", 0.9999))
        power = float(sampler_cfg.get("power", 1.0))
        class_weights = compute_class_weights(
            train_ds.targets,
            num_classes,
            method=method,
            effective_beta=effective_beta,
            power=power,
        )
        sample_weights = compute_sample_weights(train_ds.targets, class_weights)
        sampler_obj = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
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
