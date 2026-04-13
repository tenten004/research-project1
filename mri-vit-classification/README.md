# MRI ViT Classification

Vision Transformer と ResNet18 を用いて、脳 MRI 画像を 2 クラス（0: 正常、1: 異常）に分類するプロジェクトです。

## プロジェクト構成

- config/: 実験設定ファイル
- data/: 生データ・前処理済みデータ・サンプルデータ
- src/: 学習・評価・モデル定義コード
- notebooks/: 探索的解析ノートブック
- outputs/: モデル、ログ、評価指標、図の出力先
- scripts/: 補助スクリプト

## データセット配置

学習・検証（および任意でテスト）用の画像フォルダは、次の構成を想定しています。

```text
data/processed/
  train/
    class0/
    class1/
  val/
    class0/
    class1/
  test/        # 任意（最終レポート時は推奨）
    class0/
    class1/
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 学習（ViT + ResNet18 の比較）

```bash
python -m src.train --config config/config.yaml
python -m src.train --config config/config.yaml --resume
```

`--resume` を指定すると、`outputs/models/<model>_last.ckpt` が存在する場合に各モデルを再読み込みします。

## 評価

```bash
python -m src.evaluate --config config/config.yaml --model vit
python -m src.evaluate --config config/config.yaml --model resnet18
python -m src.evaluate --config config/config.yaml --model vit --split test
```

## 出力ファイル

- models: `outputs/models/*.pth`
- logs: `outputs/logs/*_epoch_log.csv`
- metrics: `outputs/metrics/comparison_metrics.csv`, `outputs/metrics/summary.txt`
- figures: `outputs/figures/*`
