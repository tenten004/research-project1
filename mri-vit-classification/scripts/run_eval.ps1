$ErrorActionPreference = "Stop"
python -m src.evaluate --config config/config.yaml --model vit
python -m src.evaluate --config config/config.yaml --model resnet18
