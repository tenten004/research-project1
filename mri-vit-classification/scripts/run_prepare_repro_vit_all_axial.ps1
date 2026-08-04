param(
  [string]$ImageRoot = "",
  [string]$CsvPathFL = "",
  [string]$CsvPathT1 = "",
  [string]$OutputRoot = "data/repro_fl_t1_all_axial_patient_split",
  [double]$ValRatio = 0.25,
  [string]$SplitMode = "patient",
  [Nullable[int]]$AxialMin = $null,
  [Nullable[int]]$AxialMax = $null,
  [int]$Seed = 1,
  [switch]$CleanOutput
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ImageRoot)) {
  throw "Specify -ImageRoot"
}
if ([string]::IsNullOrWhiteSpace($CsvPathFL)) {
  throw "Specify -CsvPathFL"
}
if ([string]::IsNullOrWhiteSpace($CsvPathT1)) {
  throw "Specify -CsvPathT1"
}
if (-not (Test-Path $ImageRoot)) {
  throw "Image root not found: $ImageRoot"
}
if (-not (Test-Path $CsvPathFL)) {
  throw "CSV not found (FL): $CsvPathFL"
}
if (-not (Test-Path $CsvPathT1)) {
  throw "CSV not found (T1): $CsvPathT1"
}
if ($SplitMode -ne "image" -and $SplitMode -ne "patient") {
  throw "SplitMode must be image or patient"
}

$prepareArgs = @(
  "--csv-paths", $CsvPathFL, $CsvPathT1,
  "--image-root", $ImageRoot,
  "--output-root", $OutputRoot,
  "--name-col", "name",
  "--id-col", "ID",
  "--label-col", "wm",
  "--axial-col", "axial",
  "--val-ratio", "$ValRatio",
  "--split-mode", $SplitMode,
  "--seed", "$Seed"
)

if ($AxialMin -ne $null) {
  $prepareArgs += @("--axial-min", "$AxialMin")
}
if ($AxialMax -ne $null) {
  $prepareArgs += @("--axial-max", "$AxialMax")
}
if ($CleanOutput) {
  $prepareArgs += "--clean-output"
}

python -m src.prepare_repro_vit_dataset @prepareArgs
