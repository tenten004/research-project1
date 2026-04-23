$ErrorActionPreference = "Stop"

param(
  [string]$ImageRoot = "C:/Users/ishii/takemura/labeled_image",
  [string]$CsvPath = "",
  [string]$CsvPathFL = "",
  [string]$CsvPathT1 = "",
  [string]$CsvPathT2 = "",
  [string]$CsvDir = "C:/Users/ishii/takemura/1_CNN/3_教師データ",
  [string]$OutputRoot = "data/grade_by_modality",
  [double]$TrainRatio = 0.8,
  [double]$ValRatio = 0.1,
  [switch]$CleanOutput
)

if (-not (Test-Path $ImageRoot)) {
  throw "Image root not found: $ImageRoot"
}

$hasSeparateCsv =
  (-not [string]::IsNullOrWhiteSpace($CsvPathFL)) -or
  (-not [string]::IsNullOrWhiteSpace($CsvPathT1)) -or
  (-not [string]::IsNullOrWhiteSpace($CsvPathT2))

if (-not $hasSeparateCsv -and [string]::IsNullOrWhiteSpace($CsvPath)) {
  $autoFL = Join-Path $CsvDir "labeled_image_list_FL_preprocess.csv"
  $autoT1 = Join-Path $CsvDir "labeled_image_list_T1_preprocess.csv"
  $autoT2 = Join-Path $CsvDir "labeled_image_list_T2_preprocess.csv"

  if (Test-Path $autoFL) { $CsvPathFL = $autoFL }
  if (Test-Path $autoT1) { $CsvPathT1 = $autoT1 }
  if (Test-Path $autoT2) { $CsvPathT2 = $autoT2 }

  $hasSeparateCsv =
    (-not [string]::IsNullOrWhiteSpace($CsvPathFL)) -or
    (-not [string]::IsNullOrWhiteSpace($CsvPathT1)) -or
    (-not [string]::IsNullOrWhiteSpace($CsvPathT2))
}

if (-not $hasSeparateCsv -and [string]::IsNullOrWhiteSpace($CsvPath)) {
  throw "Specify -CsvPath (combined CSV) or one/more of -CsvPathFL/-CsvPathT1/-CsvPathT2"
}

if ($CleanOutput -and (Test-Path $OutputRoot)) {
  Remove-Item -Path $OutputRoot -Recurse -Force
}

function Invoke-PrepareForModality {
  param(
    [string]$Csv,
    [string]$Modality
  )

  if ([string]::IsNullOrWhiteSpace($Csv)) {
    return
  }
  if (-not (Test-Path $Csv)) {
    throw "CSV not found for ${Modality}: $Csv"
  }

  python -m src.prepare_grade_dataset `
    --csv-path $Csv `
    --image-root $ImageRoot `
    --output-root $OutputRoot `
    --name-col name `
    --id-col ID `
    --label-col wm `
    --fixed-modality $Modality `
    --include-modalities FL T1 T2 `
    --train-ratio $TrainRatio `
    --val-ratio $ValRatio `
    --copy-mode copy
}

if ($hasSeparateCsv) {
  Write-Host "Use separate modality CSVs:"
  if (-not [string]::IsNullOrWhiteSpace($CsvPathFL)) { Write-Host "  FL: $CsvPathFL" }
  if (-not [string]::IsNullOrWhiteSpace($CsvPathT1)) { Write-Host "  T1: $CsvPathT1" }
  if (-not [string]::IsNullOrWhiteSpace($CsvPathT2)) { Write-Host "  T2: $CsvPathT2" }

  Invoke-PrepareForModality -Csv $CsvPathFL -Modality "FL"
  Invoke-PrepareForModality -Csv $CsvPathT1 -Modality "T1"
  Invoke-PrepareForModality -Csv $CsvPathT2 -Modality "T2"
}
else {
  if (-not (Test-Path $CsvPath)) {
    throw "CSV not found: $CsvPath"
  }

  if ($CleanOutput) {
    python -m src.prepare_grade_dataset `
      --csv-path $CsvPath `
      --image-root $ImageRoot `
      --output-root $OutputRoot `
      --name-col name `
      --id-col ID `
      --label-col wm `
      --modality-source filename_prefix `
      --include-modalities FL T1 T2 `
      --train-ratio $TrainRatio `
      --val-ratio $ValRatio `
      --copy-mode copy `
      --clean-output
  }
  else {
    python -m src.prepare_grade_dataset `
      --csv-path $CsvPath `
      --image-root $ImageRoot `
      --output-root $OutputRoot `
      --name-col name `
      --id-col ID `
      --label-col wm `
      --modality-source filename_prefix `
      --include-modalities FL T1 T2 `
      --train-ratio $TrainRatio `
      --val-ratio $ValRatio `
      --copy-mode copy
  }
}
