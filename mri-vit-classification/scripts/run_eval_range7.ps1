$ErrorActionPreference = "Stop"

# 卒研用: axial range=7 設定で保存した重みを評価
python -m src.evaluate --config config/config_axial_range7.yaml --model vit --split val
python -m src.evaluate --config config/config_axial_range7.yaml --model resnet18 --split val
