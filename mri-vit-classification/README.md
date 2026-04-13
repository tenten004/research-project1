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

## 卒研向け比較条件（Axial 9-15）

現時点の最適条件として、同じアキシャルスライス範囲（9枚目から15枚目の計7枚）で CNN と ViT を比較する場合は、以下を利用してください。

- CNN は `resnet18`（2D CNN ベースライン）として比較します。
- データは `data/processed_axial_9_15` に、通常と同じフォルダ構成（train/val/test）で配置します。
- 設定ファイルは `config/config_axial_9_15.yaml` を使います。

```text
data/processed_axial_9_15/
  train/
    class0/
    class1/
  val/
    class0/
    class1/
  test/        # 任意
    class0/
    class1/
```

実行例（PowerShell）:

```powershell
./scripts/run_train_axial_9_15.ps1
./scripts/run_eval_axial_9_15.ps1
```

実行例（bash）:

```bash
./scripts/run_train_axial_9_15.sh
./scripts/run_eval_axial_9_15.sh
```

出力先は `outputs/axial_9_15` 配下になります。

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
