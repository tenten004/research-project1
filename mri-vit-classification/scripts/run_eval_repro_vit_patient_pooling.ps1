param(
  [string]$Config = "config/config_repro_vit_all_axial_patient_split.yaml",
  [string]$Split = "val",
  [string]$CheckpointMetric = "accuracy",
  [double]$Temperature = 1.0
)

$ErrorActionPreference = "Stop"

# Evaluate patient-level aggregation without manual axial-range selection.
python -m src.evaluate --config $Config --model vit --split $Split --aggregate-level patient --pooling mean --checkpoint-metric $CheckpointMetric --temperature $Temperature
python -m src.evaluate --config $Config --model vit --split $Split --aggregate-level patient --pooling max_confidence --checkpoint-metric $CheckpointMetric --temperature $Temperature
python -m src.evaluate --config $Config --model vit --split $Split --aggregate-level patient --pooling attention_confidence --checkpoint-metric $CheckpointMetric --temperature $Temperature
python -m src.evaluate --config $Config --model vit --split $Split --aggregate-level patient --pooling attention_entropy --checkpoint-metric $CheckpointMetric --temperature $Temperature
