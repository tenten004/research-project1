$ErrorActionPreference = "Stop"

# Train ViT and CNN on patient-level split dataset for fair all-axial comparison.
python -m src.train --config config/config_repro_vit_all_axial_patient_split.yaml --models vit
python -m src.train --config config/config_repro_cnn_all_axial_patient_split.yaml --models resnet18
