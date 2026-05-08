$ErrorActionPreference = "Stop"

# Evaluate on validation split for axial 9-15 datasets
python -m src.evaluate --config config/config_grade_vit_fl_axial_9_15.yaml --model vit --split val
python -m src.evaluate --config config/config_grade_vit_t1_axial_9_15.yaml --model vit --split val
python -m src.evaluate --config config/config_grade_vit_t2_axial_9_15.yaml --model vit --split val
