#!/usr/bin/env bash
set -euo pipefail

python -m src.evaluate --config config/config.yaml --model vit
python -m src.evaluate --config config/config.yaml --model resnet18
