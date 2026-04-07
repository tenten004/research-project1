# MRI ViT Classification

Brain MRI 2-class classification project (0: normal, 1: abnormal) using Vision Transformer and ResNet18 baseline.

## Project Structure

- config/: experiment configuration
- data/: raw/processed/sample datasets
- src/: training/evaluation/model code
- notebooks/: exploratory analysis notebook
- outputs/: models, logs, metrics, figures
- scripts/: helper scripts

## Dataset Layout

Expected image-folder layout for training/validation:

```text
data/processed/
  train/
    class0/
    class1/
  val/
    class0/
    class1/
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start (Windows PowerShell)

```powershell
./scripts/run_train.ps1
./scripts/run_eval.ps1
```

## Train (ViT + ResNet18 comparison)

```bash
python -m src.train --config config/config.yaml
```

## Evaluate

```bash
python -m src.evaluate --config config/config.yaml --model vit
python -m src.evaluate --config config/config.yaml --model resnet18
```

## Outputs

- models: `outputs/models/*.pth`
- logs: `outputs/logs/*_epoch_log.csv`
- metrics: `outputs/metrics/comparison_metrics.csv`, `outputs/metrics/summary.txt`
- figures: `outputs/figures/*`

## CI

GitHub Actions workflow is added at repository root:

- `.github/workflows/python-ci.yml`

This workflow installs dependencies, runs syntax checks, and verifies CLI entry points.
