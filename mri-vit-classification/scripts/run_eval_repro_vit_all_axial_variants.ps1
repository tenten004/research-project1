$ErrorActionPreference = "Stop"

# Evaluate alternative ViT backbone experiments on validation split
python -m src.evaluate --config config/config_repro_vit_all_axial_vit_small_patch16.yaml --model vit --split val
python -m src.evaluate --config config/config_repro_vit_all_axial_deit_small_patch16.yaml --model vit --split val
