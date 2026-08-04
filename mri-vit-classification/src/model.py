from __future__ import annotations

from typing import Literal

import timm
import torch.nn as nn
from torchvision import models

ModelName = Literal["vit", "resnet18"]


def build_model(
    model_name: ModelName,
    num_classes: int,
    vit_name: str,
    image_size: int | None = None,
    drop_rate: float | None = None,
    drop_path_rate: float | None = None,
) -> nn.Module:
    # ViT: 事前学習済み重みを利用し、最終分類ヘッドのクラス数のみ合わせる
    if model_name == "vit":
        kwargs = {"pretrained": True, "num_classes": num_classes}
        if image_size is not None:
            kwargs["img_size"] = image_size
        # 過学習対策: dropout / stochastic depth (drop path) を有効化できるようにする
        if drop_rate is not None:
            kwargs["drop_rate"] = float(drop_rate)
        if drop_path_rate is not None:
            kwargs["drop_path_rate"] = float(drop_path_rate)
        return timm.create_model(vit_name, **kwargs)

    # ResNet18: 最終全結合層を2クラス分類用に差し替える
    if model_name == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # 指定ミスを早めに検出するためのガード
    raise ValueError(f"Unsupported model_name: {model_name}")
