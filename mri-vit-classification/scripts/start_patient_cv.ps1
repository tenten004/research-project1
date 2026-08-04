param(
    [switch]$RestartIncomplete,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe") -ErrorAction SilentlyContinue
if (-not $venvPython) {
    throw "Virtual environment not found. Expected .venv under the repository root."
}

$dataRoot = Join-Path $projectRoot "data\cv5_all_axial_3class"
if (-not (Test-Path (Join-Path $dataRoot "metadata.json"))) {
    throw "Prepared CV folds not found. Run src.prepare_patient_cv_folds first."
}

$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*scripts/run_patient_cv.py*" }
if ($existing) {
    $ids = ($existing | Select-Object -ExpandProperty ProcessId) -join ", "
    throw "A patient CV runner is already active (PID: $ids)."
}

Set-Location $projectRoot
$arguments = @("-u", "scripts/run_patient_cv.py")
if ($RestartIncomplete) {
    $arguments += "--restart-incomplete"
}

if ($Foreground) {
    & $venvPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Patient CV failed with exit code $LASTEXITCODE."
    }
    exit 0
}

$logRoot = Join-Path $projectRoot "outputs\cv5_all_axial_3class"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logRoot "cv_runner_${timestamp}_stdout.log"
$stderr = Join-Path $logRoot "cv_runner_${timestamp}_stderr.log"
$process = Start-Process `
    -FilePath $venvPython `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

[PSCustomObject]@{
    ProcessId = $process.Id
    StartedAt = $process.StartTime
    Stdout = $stdout
    Stderr = $stderr
}
