# MRI ViT Classification

この README は、上から順に実行すれば
「FL/T1/T2 ごとの grade 判別 (0-4)」まで完了する構成です。

## 0. このプロジェクトでやること

- 主目的: ViT で grade 判別 (0-4) を行い、撮像法 (FL/T1/T2) ごとの性能を比較
- 補足: `config/config.yaml` は 2 クラス分類 (0: 正常, 1: 異常) 用

## 1. 事前準備

### 1-1. 作業ディレクトリへ移動

プロジェクトルートで次を実行してください。

```powershell
cd mri-vit-classification
```

### 1-2. 依存パッケージをインストール

```powershell
pip install -r requirements.txt
```

## 2. 入力データの前提を確認

次を満たしていればこのまま進められます。

- CSV に `name`, `ID`, `wm` 列がある
- `wm` は grade ラベル (0-4)
- 画像は `-ImageRoot` で指定したフォルダ配下にある
- 画像名は `FL_`, `T1_`, `T2_` の接頭辞を持つ (単一CSVの場合)

## 3. grade データセットを作成

まずはこのコマンドを実行してください (PowerShell)。

```powershell
./scripts/run_prepare_grade_dataset.ps1 -ImageRoot "C:/Users/ishii/takemura/labeled_image" -CleanOutput
```

上記は `C:/Users/ishii/takemura/1_CNN/3_教師データ` にある以下の 3 CSV を自動検出します。

- `labeled_image_list_FL_preprocess.csv`
- `labeled_image_list_T1_preprocess.csv`
- `labeled_image_list_T2_preprocess.csv`

自動検出を使わない場合は、どちらか 1 つの方式で指定します。

- 単一CSVを使う:

```powershell
./scripts/run_prepare_grade_dataset.ps1 -CsvPath "C:/path/to/labels.csv" -ImageRoot "C:/path/to/images" -CleanOutput
```

- モダリティ別CSVを使う:

```powershell
./scripts/run_prepare_grade_dataset.ps1 `
  -ImageRoot "C:/path/to/images" `
  -CsvPathFL "C:/path/to/labeled_image_list_FL_preprocess.csv" `
  -CsvPathT1 "C:/path/to/labeled_image_list_T1_preprocess.csv" `
  -CsvPathT2 "C:/path/to/labeled_image_list_T2_preprocess.csv" `
  -CleanOutput
```

成功すると次が生成されます。

```text
data/grade_by_modality/
  FL/
    train/grade0 ... grade4
    val/grade0 ... grade4
    test/grade0 ... grade4
  T1/
    train/grade0 ... grade4
    val/grade0 ... grade4
    test/grade0 ... grade4
  T2/
    train/grade0 ... grade4
    val/grade0 ... grade4
    test/grade0 ... grade4
  summary_counts.csv
```

## 4. 学習を実行 (ViT, FL/T1/T2)

```powershell
./scripts/run_train_grade_modalities.ps1
```

内部で以下の設定を順に使います。

- `config/config_grade_vit_fl.yaml`
- `config/config_grade_vit_t1.yaml`
- `config/config_grade_vit_t2.yaml`

## 5. 評価を実行

```powershell
./scripts/run_eval_grade_modalities.ps1
```

## 6. 結果を確認

以下の 3 ファイルを比較してください。

- `outputs/grade_vit/fl/metrics/vit_eval_val.json`
- `outputs/grade_vit/t1/metrics/vit_eval_val.json`
- `outputs/grade_vit/t2/metrics/vit_eval_val.json`

主に `accuracy`, `f1`, `roc_auc` を比較します。

## 7. よくある詰まりポイント

- `--resume` は現在未対応です
- CSV の列名が違うとデータ生成に失敗します (`name`, `ID`, `wm` を確認)
- 画像ファイルが見つからない場合は、`-ImageRoot` とファイル名の一致を確認

## 8. オプション: 2クラス分類を実行したい場合

grade 判別ではなく、2クラス分類 (正常/異常) を回す場合の手順です。

```powershell
./scripts/run_train.ps1
./scripts/run_eval.ps1
```

または直接実行:

```powershell
python -m src.train --config config/config.yaml --models vit resnet18
python -m src.evaluate --config config/config.yaml --model vit --split val
python -m src.evaluate --config config/config.yaml --model resnet18 --split val
```

## 9. オプション: Axial 9-15 比較

```powershell
./scripts/run_train_axial_9_15.ps1
./scripts/run_eval_axial_9_15.ps1
```
