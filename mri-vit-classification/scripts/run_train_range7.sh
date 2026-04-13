#!/usr/bin/env bash
set -euo pipefail

# 卒研用: axial range=7 データで ViT と CNN(ResNet18) を同条件比較
python -m src.train --config config/config_axial_range7.yaml --models vit resnet18
