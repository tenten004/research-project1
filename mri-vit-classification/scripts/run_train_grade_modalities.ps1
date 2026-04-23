$ErrorActionPreference = "Stop"

# ViT only: compare grade classification performance by modality (FL/T1/T2)
python -m src.train --config config/config_grade_vit_fl.yaml --models vit
python -m src.train --config config/config_grade_vit_t1.yaml --models vit
python -m src.train --config config/config_grade_vit_t2.yaml --models vit
