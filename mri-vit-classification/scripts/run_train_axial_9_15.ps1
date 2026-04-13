$ErrorActionPreference = "Stop"

# 卒研用: axial 9-15（7枚固定）で ViT と CNN(ResNet18) を同条件比較
python -m src.train --config config/config_axial_9_15.yaml --models vit resnet18
