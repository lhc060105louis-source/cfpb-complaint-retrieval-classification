[CmdletBinding()]
param(
    [int]$From = 0,
    [int]$To = 9,
    [switch]$Help
)

if ($Help) {
@"
Usage: run_all.ps1 [-From <int>] [-To <int>] [-Help]

Options:
  -From <int>    Stage index to start from (0-9). Default: 0.
  -To   <int>    Stage index to stop after (0-9). Default: 9.
  -Help          Show this help.

Stages:
  0 check_header
  1 eda
  2 eda_report
  3 make_subset                  (skipped by default; only run when explicit)
  4 make_split
  5 bm25_demo
  6 tfidf_logreg_baseline
  7 retrieval_augmented_classification
  8 hybrid_retrieval_classifier
  9 compare_results

Environment variables forwarded to every stage:
  CS410_DATA_PATH, CS410_OUT_DIR, CS410_FIG_DIR,
  CS410_SPLIT_NAME, CS410_SPLIT_PATH, CS410_SPLIT_COUNTS_NAME,
  CS410_MAX_ROWS, CS410_OVR_N_JOBS,
  CS410_SVD_DIM, CS410_IVF_NLIST, CS410_IVF_NPROBE, CS410_IVF_TRAIN_SAMPLES,
  CS410_BM25_TOPK, CS410_BM25_NUM_QUERIES,
  CS410_SRC_PATH, CS410_SUBSET_PATH, CS410_N_LINES.

Tip: data\complaints.csv must exist before running. Use .\download_data.ps1
     to fetch the official CFPB dump if you do not already have it.
"@ | Write-Host
    exit 0
}

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir

$DataPath = if ($env:CS410_DATA_PATH) { $env:CS410_DATA_PATH } else { "data\complaints.csv" }
if (-not (Test-Path $DataPath)) {
    Write-Error @"
Data file not found at '$DataPath'.
Run .\download_data.ps1 first, or set CS410_DATA_PATH to point at an
existing CFPB complaints CSV.
"@
    exit 3
}

$OutDir = if ($env:CS410_OUT_DIR) { $env:CS410_OUT_DIR } else { "outputs" }
$FigDir = if ($env:CS410_FIG_DIR) { $env:CS410_FIG_DIR } else { "figs" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $FigDir | Out-Null

$Stages = @(
    "00_check_header.py",
    "01_eda.py",
    "02_eda_report.py",
    "03_make_subset.py",
    "04_make_split.py",
    "05_bm25_demo.py",
    "06_tfidf_logreg_baseline.py",
    "07_retrieval_augmented_classification.py",
    "08_hybrid_retrieval_classifier.py",
    "09_compare_results.py"
)

for ($idx = $From; $idx -le $To; $idx++) {
    $script = $Stages[$idx]
    if ($idx -eq 3 -and $From -ne 3 -and $To -ne 3) {
        Write-Host "==== [stage $idx] $script (skipped; pass -From 3 to run) ===="
        continue
    }
    Write-Host "==== [stage $idx] $script ===="
    & python -u (Join-Path "pipeline" $script)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Stage $idx ($script) failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

Write-Host "==== all requested stages finished ===="
