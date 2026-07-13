#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_all.sh [--from STAGE] [--to STAGE] [-h|--help]

Options:
  --from STAGE    Stage index to start from (0-9). Default: 0.
  --to   STAGE    Stage index to stop after (0-9). Default: 9.
  -h, --help      Show this help.

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

Tip: data/complaints.csv must exist before running. Use ./download_data.sh
     to fetch the official CFPB dump if you do not already have it.
EOF
}

START=0
END=9

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) START="$2"; shift 2 ;;
        --to)   END="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="${CS410_DATA_PATH:-data/complaints.csv}"
if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: data file not found at '$DATA_PATH'." >&2
    echo "Run ./download_data.sh first, or set CS410_DATA_PATH to point at" >&2
    echo "an existing CFPB complaints CSV." >&2
    exit 3
fi

mkdir -p "${CS410_OUT_DIR:-outputs}" "${CS410_FIG_DIR:-figs}"

declare -a STAGES=(
    "00_check_header.py"
    "01_eda.py"
    "02_eda_report.py"
    "03_make_subset.py"
    "04_make_split.py"
    "05_bm25_demo.py"
    "06_tfidf_logreg_baseline.py"
    "07_retrieval_augmented_classification.py"
    "08_hybrid_retrieval_classifier.py"
    "09_compare_results.py"
)

run_stage() {
    local idx="$1"
    local script="${STAGES[$idx]}"
    if [[ "$idx" -eq 3 ]] && [[ "$START" -ne 3 ]] && [[ "$END" -ne 3 ]]; then
        echo "==== [stage $idx] $script (skipped; pass --from 3 to run) ===="
        return 0
    fi
    echo "==== [stage $idx] $script ===="
    python3 -u "pipeline/$script"
}

for idx in $(seq "$START" "$END"); do
    run_stage "$idx"
done

echo "==== all requested stages finished ===="
