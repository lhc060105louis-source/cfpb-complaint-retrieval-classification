# CFPB Consumer Complaint Similar-Case Retrieval and Issue Classification

End-to-end text-mining pipeline over the full Consumer Financial Protection
Bureau (CFPB) consumer complaint dump (~8.6 GB CSV, ~14.8 M raw rows,
~3.77 M usable narratives). The pipeline performs exploratory analysis, builds
a time-based train/val/test split, runs a BM25 similar-case retrieval demo,
and benchmarks three Issue classifiers:

1. TF-IDF + One-vs-Rest Logistic Regression (baseline).
2. Retrieval-Augmented Classification (RAC): TF-IDF -> TruncatedSVD(256) ->
   FAISS IVFFlat to retrieve the top-3 most similar training narratives for
   every input, append them as evidence, and re-train a TF-IDF + OvR LogReg
   classifier on the augmented text.
3. Conservative Hybrid: keep the baseline prediction unless the top-3 RAC
   neighbours unanimously agree on a label, in which case the retrieval vote
   overrides.

## Directory layout

```
cs410_release/
├── README.md                      # this file
├── requirements.txt               # Python dependencies
├── download_data.sh               # Linux/macOS data downloader
├── download_data.ps1              # Windows PowerShell data downloader
├── run_all.sh                     # Linux/macOS batch driver
├── run_all.ps1                    # Windows PowerShell batch driver
└── pipeline/
    ├── __init__.py                # re-exports common helpers
    ├── common.py                  # Paths, RuntimeLimits, logger, streaming reader,
    │                              # filtering, split helpers
    ├── 00_check_header.py         # print column names of complaints.csv
    ├── 01_eda.py                  # length distributions, top labels, monthly counts,
    │                              # 4 figures
    ├── 02_eda_report.py           # render a paper-ready EDA summary text
    ├── 03_make_subset.py          # (optional) materialize a head-N row subset
    ├── 04_make_split.py           # time-based 70/15/15 split, write IDs to JSON
    ├── 05_bm25_demo.py            # BM25Okapi index over train, top-K demo on test
    ├── 06_tfidf_logreg_baseline.py  # TF-IDF (1+2grams) + OvR LogReg (liblinear)
    ├── 07_retrieval_augmented_classification.py
    │                              # SVD(256)+FAISS IVFFlat retrieval, augmented-text
    │                              # classifier, parquet checkpoint between phases
    ├── 08_hybrid_retrieval_classifier.py
    │                              # baseline preds + top-3 unanimous override
    └── 09_compare_results.py      # 3-model comparison CSV + figure + summary text
```

## Hardware and runtime estimates

The full pipeline was developed on a workstation with 96 CPU cores and 1 TB of
RAM. End-to-end full-data runtime on this hardware:

| Stage | Runtime |
| --- | --- |
| 00 check_header | < 1 s |
| 01 EDA | ~4 min |
| 02 EDA report | < 1 s |
| 04 split | ~2 min |
| 05 BM25 demo | ~25 min |
| 06 baseline (TF-IDF + OvR LogReg, n_jobs=-1) | ~73 min |
| 07 RAC (SVD + FAISS + classifier, OvR n_jobs=16) | ~5 h 44 min |
| 08 hybrid | ~18 min |
| 09 compare | < 2 s |
| **Total full-data run** | **~7.5 h** |

You will also need disk space for:

- `data/complaints.csv` -- ~8.6 GB.
- `outputs/rac_augmented_df.parquet` -- ~5 GB checkpoint.
- `outputs/rac_retrieval_evidence.csv` -- ~4.5 GB.
- `outputs/*_predictions.csv` -- ~1.3 GB each (3 files).

Plan for **~25 GB** of free space under the output directory at full scale.

## Setup

### 1. Create the Python environment

```bash
conda create -y -n cs410 python=3.10 pip
conda activate cs410
pip install -r requirements.txt
```

`faiss-cpu` ships pre-built wheels with AVX2/AVX512 BLAS bundled, so no system
package or `sudo` is required.

### 2. Download the data

The pipeline expects `data/complaints.csv`, the official CFPB dump
(~1.8 GB compressed, ~8.6 GB extracted; takes 1-3 minutes on a fast link):

```bash
# Linux/macOS
./download_data.sh

# Windows PowerShell
./download_data.ps1
```

Both scripts default to `./data/`; override with `--data-dir DIR`
(`-DataDir DIR` on PowerShell) or set `CS410_DATA_DIR`. They skip the
download if `complaints.csv` already exists in the destination.

## How to run

```bash
# Linux/macOS
./run_all.sh

# Windows PowerShell
./run_all.ps1
```

This runs every stage with default paths (`data/complaints.csv`, `outputs/`,
`figs/`). To run a subset of stages use `--from N --to M` (or `-From N -To M`
on PowerShell).

### Resuming the RAC stage after a crash

`07_retrieval_augmented_classification.py` writes a parquet checkpoint
(`outputs/rac_augmented_df.parquet`, ~5 GB) right after the retrieval phase
completes. If the classifier-training phase fails (e.g. OOM), simply re-running
the script will detect the checkpoint and skip directly back to vectorization
and classifier training.

To force a clean re-run of retrieval, delete the checkpoint:

```bash
rm outputs/rac_augmented_df.parquet
```

## Environment variables

All scripts read configuration from environment variables. None are required
for a default full-data run; the table below lists every variable the pipeline
honours.

| Variable | Default | Used by | Purpose |
| --- | --- | --- | --- |
| `CS410_DATA_PATH` | `data/complaints.csv` | all read-CSV stages | Path to the CFPB CSV. |
| `CS410_OUT_DIR` | `outputs` | all stages | Directory for CSV/JSON/parquet outputs. |
| `CS410_FIG_DIR` | `figs` | EDA, compare | Directory for PNG figures. |
| `CS410_SPLIT_NAME` | `split_full.json` | `04`, used by `Paths.from_env` | File name (under `OUT_DIR`) for the split JSON. |
| `CS410_SPLIT_PATH` | `<OUT_DIR>/<SPLIT_NAME>` | every classifier stage | Override the full split path. |
| `CS410_SPLIT_COUNTS_NAME` | `split_counts_full.csv` | `04` | File name for the split row-count summary. |
| `CS410_MAX_ROWS` | unset (read all rows) | every read-CSV stage | Truncate CSV reads to the first N rows for fast iteration. |
| `CS410_OVR_N_JOBS` | `16` (RAC) | `07` | OvR parallelism inside RAC; lower this if memory is tight. |
| `CS410_SVD_DIM` | `256` | `07` | TruncatedSVD output dimension feeding FAISS. |
| `CS410_IVF_NLIST` | `1024` | `07` | FAISS IVF cluster count (auto-capped to sqrt(N)). |
| `CS410_IVF_NPROBE` | `16` | `07` | FAISS IVF probes per query. |
| `CS410_IVF_TRAIN_SAMPLES` | `262144` | `07` | Sample size used to train the IVF quantizer. |
| `CS410_BM25_TOPK` | `5` | `05` | BM25 demo top-K. |
| `CS410_BM25_NUM_QUERIES` | `2` | `05` | Number of test queries to showcase. |
| `CS410_SRC_PATH` | `data/complaints.csv` | `03` | Subset source CSV. |
| `CS410_SUBSET_PATH` | `data/complaints_subset.csv` | `03` | Subset destination CSV. |
| `CS410_N_LINES` | `200001` | `03` | Lines to copy into the subset (header + 200k rows). |

## Outputs

After a full run, `outputs/` contains:

- EDA: `eda_summary.csv`, `top_products.csv`, `top_issues.csv`,
  `month_counts.csv`, `eda_report_ready.txt`.
- Split: `split_full.json` (~43 MB), `split_counts_full.csv`.
- BM25 demo: `retrieval_demo.json`, `retrieval_demo_table.csv`.
- Baseline classifier: `tfidf_logreg_issue_metrics.csv`,
  `tfidf_logreg_issue_classification_report.csv`,
  `tfidf_logreg_issue_predictions.csv`.
- RAC: `rac_issue_metrics.csv`, `rac_issue_classification_report.csv`,
  `rac_issue_predictions.csv`, `rac_retrieval_evidence.csv` (~4.5 GB),
  `rac_augmented_df.parquet` (~5 GB checkpoint).
- Hybrid: `hybrid_retrieval_issue_metrics.csv`,
  `hybrid_retrieval_issue_predictions.csv`.
- Comparison: `classification_model_comparison.csv`,
  `classification_report_ready.txt`.

`figs/` contains:

- `top_products.png`, `top_issues.png`,
  `text_len_words_hist.png`, `month_counts.png`.
- `classification_model_comparison.png`.

## Reference results (held-out test split, full data)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- |
| TF-IDF + LogReg (baseline) | 0.479 | 0.174 | **0.491** |
| RAC (SVD + FAISS + augmented text) | 0.443 | 0.140 | 0.455 |
| Conservative Hybrid retrieval | 0.467 | 0.173 | 0.478 |

The plain TF-IDF baseline is the strongest model on the held-out test split.
RAC and Hybrid retrieve useful similar cases (good for explanations) but do
not outperform the baseline on classification accuracy.

## Engineering notes

- **CSV reading.** All read-CSV stages use `engine="c"`, `dtype=str`,
  `chunksize=200_000`, with a `tqdm` row-level progress bar whose total is
  estimated by sampling the first 50 MB of the file. Reading 8.6 GB takes
  about one minute end-to-end.
- **Empty-narrative filtering.** Because `dtype=str` converts pandas NaN to
  the literal string `"nan"`, every filter explicitly rejects rows whose
  text column lower-cases to `"nan"` in addition to the standard empty/
  whitespace checks.
- **RAC retriever.** The original brute-force sklearn nearest-neighbour
  search on 60k-d sparse TF-IDF vectors is intractable on full data
  (~5+ days). The released code instead fits TruncatedSVD(256) on the
  TF-IDF matrix, L2-normalizes the dense embeddings, and uses a FAISS
  IndexIVFFlat with inner product (= cosine on normalized vectors) to do
  approximate top-K search. Retrieval over 2.6 M training documents finishes
  in roughly half an hour.
- **OOM-safe RAC training.** OvR LogReg with `n_jobs=-1` (96 workers) on
  the augmented text matrix exceeds 1 TB of RAM. The released code limits
  the OvR worker count via `CS410_OVR_N_JOBS` (default 16), and the
  retrieval phase persists its augmented dataframe to a parquet checkpoint
  so the long retrieval pipeline does not need to be re-run if the
  classifier crashes.
