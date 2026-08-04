param(
  [string]$Config = "config/config_repro_vit_all_axial_patient_split.yaml",
  [string]$Split = "val"
)

$ErrorActionPreference = "Stop"

# Compare slice-level performance across metric-selected checkpoints.
python -m src.evaluate --config $Config --model vit --split $Split --aggregate-level slice --checkpoint-metric primary
python -m src.evaluate --config $Config --model vit --split $Split --aggregate-level slice --checkpoint-metric loss
python -m src.evaluate --config $Config --model vit --split $Split --aggregate-level slice --checkpoint-metric accuracy
python -m src.evaluate --config $Config --model vit --split $Split --aggregate-level slice --checkpoint-metric f1
python -m src.evaluate --config $Config --model vit --split $Split --aggregate-level slice --checkpoint-metric roc_auc
