from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: str = "dataset"
    output_dir: str = "outputs"

    image_size: int = 224
    mean: float = 0.5
    std: float = 0.5

    batch_size: int = 16
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    seed: int = 42

    vit_name: str = "vit_base_patch16_224"
    num_classes: int = 2

    def ensure_output_dirs(self) -> None:
        base = Path(self.output_dir)
        (base / "models").mkdir(parents=True, exist_ok=True)
        (base / "logs").mkdir(parents=True, exist_ok=True)
        (base / "metrics").mkdir(parents=True, exist_ok=True)
        (base / "plots").mkdir(parents=True, exist_ok=True)
        (base / "attention").mkdir(parents=True, exist_ok=True)
