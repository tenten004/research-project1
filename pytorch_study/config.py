from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # データセットと出力先の基本パス
    data_dir: str = "dataset"
    output_dir: str = "outputs"

    # 画像前処理の設定
    image_size: int = 224
    mean: float = 0.5
    std: float = 0.5

    # 学習ハイパーパラメータ
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    seed: int = 42

    # モデル設定
    vit_name: str = "vit_base_patch16_224"
    num_classes: int = 2

    def ensure_output_dirs(self) -> None:
        # 学習結果を保存するフォルダを事前に作成
        base = Path(self.output_dir)
        (base / "models").mkdir(parents=True, exist_ok=True)
        (base / "logs").mkdir(parents=True, exist_ok=True)
        (base / "metrics").mkdir(parents=True, exist_ok=True)
        (base / "plots").mkdir(parents=True, exist_ok=True)
        (base / "attention").mkdir(parents=True, exist_ok=True)
