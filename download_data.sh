#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: download_data.sh [--data-dir DIR] [--keep-zip] [-h|--help]

Downloads the official CFPB consumer complaint dump (~1.8 GB zipped,
~8.6 GB extracted) and unpacks it into the target directory.

Options:
  --data-dir DIR   Destination directory. Default: ./data
                   (or $CS410_DATA_DIR if set in the environment)
  --keep-zip       Do not delete complaints.csv.zip after extraction.
  -h, --help       Show this help.

The download URL is hard-coded to:
  https://files.consumerfinance.gov/ccdb/complaints.csv.zip
EOF
}

DEST_DIR="${CS410_DATA_DIR:-data}"
KEEP_ZIP=0
URL="https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DEST_DIR="$2"; shift 2 ;;
        --keep-zip) KEEP_ZIP=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$DEST_DIR"
ZIP_PATH="$DEST_DIR/complaints.csv.zip"
CSV_PATH="$DEST_DIR/complaints.csv"

if [[ -f "$CSV_PATH" ]]; then
    SIZE=$(stat -Lc%s "$CSV_PATH" 2>/dev/null || stat -Lf%z "$CSV_PATH")
    echo "Existing complaints.csv found at $CSV_PATH (${SIZE} bytes); skipping download."
    exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required but not installed." >&2
    exit 4
fi
if ! command -v unzip >/dev/null 2>&1; then
    echo "ERROR: unzip is required but not installed." >&2
    exit 4
fi

echo "Downloading $URL"
echo "  -> $ZIP_PATH"
curl -L --fail --progress-bar -o "$ZIP_PATH" "$URL"

echo "Extracting into $DEST_DIR"
unzip -o "$ZIP_PATH" -d "$DEST_DIR"

if [[ "$KEEP_ZIP" -eq 0 ]]; then
    rm -f "$ZIP_PATH"
    echo "Removed $ZIP_PATH (pass --keep-zip to keep it)."
fi

echo "Done. CSV at $CSV_PATH"
