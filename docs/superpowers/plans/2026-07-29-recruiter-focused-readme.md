# Recruiter-Focused README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current README with an accurate, recruiter-focused English portfolio page that communicates the project's scale, technical depth, workflow, results, and reproducibility.

**Architecture:** Keep the documentation in one canonical `README.md`. Organize it in two layers: a portfolio-oriented first half for rapid evaluation and a reproducibility-oriented second half for developers.

**Tech Stack:** Markdown, Mermaid, Shields.io badges, Python, pandas, scikit-learn, BM25, TruncatedSVD, FAISS.

## Global Constraints

- The README must remain entirely in English.
- The primary audience is technical recruiters and hiring managers.
- Preserve the repository's documented dataset scale, runtime estimates, disk estimates, model configurations, reference metrics, commands, output paths, and environment variables.
- Describe the repository as an offline experimental pipeline, not a deployed application.
- State that the TF-IDF baseline outperformed both retrieval-based variants on the held-out test split.
- Do not invent tests, deployments, APIs, licenses, or capabilities.
- Keep Linux/macOS and Windows execution instructions.

---

### Task 1: Rewrite and Verify the README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Existing scripts in `pipeline/`, `run_all.sh`, `run_all.ps1`, `download_data.sh`, `download_data.ps1`, and `requirements.txt`.
- Produces: A standalone GitHub landing page whose commands and references match those files.

- [ ] **Step 1: Replace the opening with recruiter-oriented positioning**

Add the title, badges, concise value proposition, navigation, project overview,
highlights, and a technology summary. Mention the approximately 14.8 million
raw rows, 3.77 million usable narratives, chronological evaluation, and
retrieval evidence without claiming deployment.

- [ ] **Step 2: Add the pipeline architecture and model explanations**

Add a valid Mermaid flowchart covering data download, preprocessing/EDA,
chronological splitting, BM25 demonstration, TF-IDF baseline, RAC, hybrid
override, and evaluation. Explain each of the three classifiers and how
Top-3 retrieved evidence is constructed.

- [ ] **Step 3: Present results and engineering interpretation**

Reproduce the exact test metrics:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.479 | 0.174 | 0.491 |
| Retrieval-Augmented Classification | 0.443 | 0.140 | 0.455 |
| Conservative Hybrid | 0.467 | 0.173 | 0.478 |

Explain that the baseline performed best, retrieval remained valuable for
case-level evidence, and low Macro-F1 relative to Weighted-F1 is consistent
with a long-tailed label distribution.

- [ ] **Step 4: Preserve reproducibility documentation**

Document repository structure, Python 3.10 environment creation, dependency
installation, data download, cross-platform execution, stage selection,
small-scale iteration with `CS410_MAX_ROWS`, RAC checkpoint recovery, output
artifacts, environment variables, hardware/runtime estimates, and disk
requirements.

- [ ] **Step 5: Add engineering decisions and limitations**

Describe chunked CSV ingestion, invalid narrative filtering, chronological
splitting, SVD compression, cosine retrieval through normalized inner product,
FAISS IVF approximate search, bounded OvR parallelism, and checkpointing.
Explicitly list the lack of a serving API, model persistence, retrieval metrics,
automated tests/CI, systematic tuning, and a repository license.

- [ ] **Step 6: Run documentation verification**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

readme = Path("README.md").read_text(encoding="utf-8")
required = [
    "14.8 million",
    "3.77 million",
    "0.479",
    "0.174",
    "0.491",
    "download_data.sh",
    "run_all.ps1",
    "CS410_MAX_ROWS",
    "rac_augmented_df.parquet",
    "```mermaid",
]
missing = [item for item in required if item not in readme]
assert not missing, f"README is missing: {missing}"
assert "TBD" not in readme and "TODO" not in readme
print("README content checks passed")
PY
git diff --check
```

Expected: `README content checks passed`, followed by no output from
`git diff --check`.

- [ ] **Step 7: Review the final documentation diff**

Run:

```bash
git diff -- README.md
git status --short
```

Expected: `README.md` is the only implementation file modified; the committed
design and plan documents may also appear in repository history.

- [ ] **Step 8: Commit**

```bash
git add README.md docs/superpowers/plans/2026-07-29-recruiter-focused-readme.md
git commit -m "docs: rewrite README for portfolio presentation"
```
