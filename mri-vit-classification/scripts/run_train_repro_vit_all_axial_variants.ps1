$ErrorActionPreference = "Stop"

# Run alternative ViT backbone experiments (baseline output is untouched)
python -m src.train --config config/config_repro_vit_all_axial_vit_small_patch16.yaml --models vit
python -m src.train --config config/config_repro_vit_all_axial_deit_small_patch16.yaml --models vit
