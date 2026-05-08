$ErrorActionPreference = "Stop"

# ViT only: compare grade classification performance by modality (FL/T1/T2)
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe") -ErrorAction SilentlyContinue
if (-not $venvPython) {
	throw "Virtual environment not found. Expected .venv under the repository root."
}

Set-Location $repoRoot

& $venvPython -m src.train --config config/config_grade_vit_fl.yaml --models vit
& $venvPython -m src.train --config config/config_grade_vit_t1.yaml --models vit
& $venvPython -m src.train --config config/config_grade_vit_t2.yaml --models vit
