from typing import Literal

import timm
import torch.nn as nn
from torchvision import models


ModelName = Literal["vit", "resnet18"]


def build_model(model_name: ModelName, num_classes: int = 2, vit_model_name: str = "vit_base_patch16_224") -> nn.Module:
    # ViT: 事前学習済みモデルを使い、最終分類層のクラス数を合わせる
    if model_name == "vit":
        model = timm.create_model(vit_model_name, pretrained=True, num_classes=num_classes)
        return model

    # ResNet18: 最終全結合層を2クラス分類用に差し替える
    if model_name == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=True)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # サポート対象外のモデル名は早めにエラーで通知
    raise ValueError(f"Unsupported model_name: {model_name}")
