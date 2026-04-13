#!/usr/bin/env bash
set -euo pipefail

# 卒研用: axial 9-15 設定で保存した重みを評価
python -m src.evaluate --config config/config_axial_9_15.yaml --model vit --split val
python -m src.evaluate --config config/config_axial_9_15.yaml --model resnet18 --split val
