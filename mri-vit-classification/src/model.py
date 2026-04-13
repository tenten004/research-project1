from typing import Literal

import timm
import torch.nn as nn
from torchvision import models

ModelName = Literal["vit", "resnet18"]


def build_model(model_name: ModelName, num_classes: int, vit_name: str) -> nn.Module:
    # ViT: 事前学習済み重みを利用し、最終分類ヘッドのクラス数のみ合わせる
    if model_name == "vit":
        return timm.create_model(vit_name, pretrained=True, num_classes=num_classes)

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
