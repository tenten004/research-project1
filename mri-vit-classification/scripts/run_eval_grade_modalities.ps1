$ErrorActionPreference = "Stop"

# Evaluate on validation split (and test split if available)
python -m src.evaluate --config config/config_grade_vit_fl.yaml --model vit --split val
python -m src.evaluate --config config/config_grade_vit_t1.yaml --model vit --split val
python -m src.evaluate --config config/config_grade_vit_t2.yaml --model vit --split val
