[CmdletBinding()]
param(
    [string]$DataDir = "",
    [switch]$KeepZip,
    [switch]$Help
)

if ($Help) {
@"
Usage: download_data.ps1 [-DataDir DIR] [-KeepZip] [-Help]

Downloads the official CFPB consumer complaint dump (~1.8 GB zipped,
~8.6 GB extracted) and unpacks it into the target directory.

Options:
  -DataDir DIR     Destination directory. Default: .\data
                   (or $env:CS410_DATA_DIR if set)
  -KeepZip         Do not delete complaints.csv.zip after extraction.
  -Help            Show this help.

The download URL is hard-coded to:
  https://files.consumerfinance.gov/ccdb/complaints.csv.zip
"@ | Write-Host
    exit 0
}

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir

if (-not $DataDir) {
    if ($env:CS410_DATA_DIR) { $DataDir = $env:CS410_DATA_DIR } else { $DataDir = "data" }
}

$Url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
$ZipPath = Join-Path $DataDir "complaints.csv.zip"
$CsvPath = Join-Path $DataDir "complaints.csv"

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

if (Test-Path $CsvPath) {
    $size = (Get-Item $CsvPath).Length
    Write-Host "Existing complaints.csv found at $CsvPath ($size bytes); skipping download."
    exit 0
}

Write-Host "Downloading $Url"
Write-Host "  -> $ZipPath"

$progressPreference = "Continue"
Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing

Write-Host "Extracting into $DataDir"
Expand-Archive -Path $ZipPath -DestinationPath $DataDir -Force

if (-not $KeepZip) {
    Remove-Item -Force $ZipPath
    Write-Host "Removed $ZipPath (pass -KeepZip to keep it)."
}

Write-Host "Done. CSV at $CsvPath"
