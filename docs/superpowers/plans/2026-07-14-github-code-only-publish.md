# GitHub Code-Only Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a clean, reproducible, code-only version of the CFPB complaint retrieval and classification project to `lhc060105louis-source/cfpb-complaint-retrieval-classification`.

**Architecture:** Promote the release package contents to the repository root, protect local data and generated artifacts with a root `.gitignore`, validate source syntax without requiring the multi-gigabyte dataset, then explicitly stage and push the intended files. Preserve the existing pipeline behavior and documentation while excluding the report and presentation media.

**Tech Stack:** Git, GitHub CLI, PowerShell, Bash, Python 3.10+, pandas, scikit-learn, FAISS, BM25.

## Global Constraints

- Publish source code and necessary project documentation only.
- Do not track `CS 410 Project Final Report.pdf` or `Cs410 pre video.mp4`.
- Do not track downloaded data, generated outputs, figures, checkpoints, logs, caches, or virtual environments.
- Do not change model behavior or retrain any model.
- Publish to the empty public repository `lhc060105louis-source/cfpb-complaint-retrieval-classification` on `main`.

---

### Task 1: Promote the release package to the repository root

**Files:**
- Move: `cs410_release/README.md` to `README.md`
- Move: `cs410_release/requirements.txt` to `requirements.txt`
- Move: `cs410_release/download_data.ps1` to `download_data.ps1`
- Move: `cs410_release/download_data.sh` to `download_data.sh`
- Move: `cs410_release/run_all.ps1` to `run_all.ps1`
- Move: `cs410_release/run_all.sh` to `run_all.sh`
- Move: `cs410_release/pipeline/` to `pipeline/`

**Interfaces:**
- Consumes: the existing release package under `cs410_release/`.
- Produces: a repository-root CLI layout matching every path already documented in `README.md`.

- [ ] **Step 1: Verify every source and destination resolves inside the workspace**

Run:

```powershell
$root = (Resolve-Path -LiteralPath '.').Path
$source = (Resolve-Path -LiteralPath 'cs410_release').Path
if (-not $source.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Source escaped workspace' }
if (Test-Path -LiteralPath 'pipeline') { throw 'Destination pipeline already exists' }
```

Expected: no output and exit code 0.

- [ ] **Step 2: Move each intended release item to the root**

Run:

```powershell
Move-Item -LiteralPath 'cs410_release\README.md' -Destination 'README.md'
Move-Item -LiteralPath 'cs410_release\requirements.txt' -Destination 'requirements.txt'
Move-Item -LiteralPath 'cs410_release\download_data.ps1' -Destination 'download_data.ps1'
Move-Item -LiteralPath 'cs410_release\download_data.sh' -Destination 'download_data.sh'
Move-Item -LiteralPath 'cs410_release\run_all.ps1' -Destination 'run_all.ps1'
Move-Item -LiteralPath 'cs410_release\run_all.sh' -Destination 'run_all.sh'
Move-Item -LiteralPath 'cs410_release\pipeline' -Destination 'pipeline'
```

Expected: the seven destinations exist at the repository root.

- [ ] **Step 3: Confirm the old release directory is empty, then remove it**

Run:

```powershell
$remaining = Get-ChildItem -LiteralPath 'cs410_release' -Force
if ($remaining) { throw "Unexpected files remain in cs410_release: $($remaining.Name -join ', ')" }
Remove-Item -LiteralPath 'cs410_release'
```

Expected: `cs410_release/` no longer exists.

### Task 2: Add repository exclusions

**Files:**
- Create: `.gitignore`

**Interfaces:**
- Consumes: local runtime paths documented by the pipeline README.
- Produces: Git exclusion rules that keep source visible and large/local artifacts untracked.

- [ ] **Step 1: Create `.gitignore` with exact code-only exclusions**

Create `.gitignore` with:

```gitignore
# Local project media (code-only publication)
/CS 410 Project Final Report.pdf
/Cs410 pre video.mp4

# Downloaded data and generated artifacts
/data/
/outputs/
/figs/
/tmp/
*.parquet
*.log

# Python caches and environments
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
env/

# Editor and operating-system files
.vscode/
.idea/
.DS_Store
Thumbs.db
```

- [ ] **Step 2: Verify media and runtime artifacts are ignored**

Run:

```powershell
git check-ignore -v -- 'CS 410 Project Final Report.pdf' 'Cs410 pre video.mp4' 'tmp/pdfs/final-report-page1.png'
```

Expected: one matching `.gitignore` rule for each path.

### Task 3: Validate the code-only repository

**Files:**
- Inspect: `README.md`
- Inspect: `requirements.txt`
- Test: `pipeline/*.py`, `*.ps1`, and `*.sh`

**Interfaces:**
- Consumes: the root-level source layout from Tasks 1-2.
- Produces: evidence that source files parse and the staged scope contains no binary media or runtime data.

- [ ] **Step 1: Parse every Python source file**

Run:

```powershell
python -m compileall -q pipeline
```

Expected: no output and exit code 0.

- [ ] **Step 2: Parse the PowerShell scripts without executing them**

Run:

```powershell
$errors = @()
Get-ChildItem -Filter '*.ps1' | ForEach-Object {
  $tokens = $null
  $parseErrors = $null
  [System.Management.Automation.Language.Parser]::ParseFile($_.FullName, [ref]$tokens, [ref]$parseErrors) | Out-Null
  $errors += $parseErrors
}
if ($errors.Count -gt 0) { $errors | Format-List; exit 1 }
```

Expected: no output and exit code 0.

- [ ] **Step 3: Parse the Bash scripts when Bash is available**

Run:

```powershell
$bash = Get-Command bash -ErrorAction SilentlyContinue
if ($bash) { & $bash.Source -n download_data.sh run_all.sh; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
```

Expected: no output and exit code 0; absence of Bash is reported as a skipped platform check.

- [ ] **Step 4: Remove generated Python bytecode and confirm repository scope**

Run:

```powershell
$root = (Resolve-Path -LiteralPath '.').Path
$cacheDirs = Get-ChildItem -LiteralPath 'pipeline' -Directory -Filter '__pycache__' -Recurse
foreach ($cacheDir in $cacheDirs) {
  $resolved = (Resolve-Path -LiteralPath $cacheDir.FullName).Path
  if (-not $resolved.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Cache path escaped workspace' }
  Remove-Item -LiteralPath $resolved -Recurse -Force
}
git status --short --ignored
```

Expected: source, README, requirements, scripts, `.gitignore`, design, and plan are visible; PDF, video, and `tmp/` are ignored.

### Task 4: Commit the intended source publication

**Files:**
- Stage: `.gitignore`, `README.md`, `requirements.txt`, `download_data.ps1`, `download_data.sh`, `run_all.ps1`, `run_all.sh`, `pipeline/`, `docs/superpowers/`

**Interfaces:**
- Consumes: validated repository contents.
- Produces: a local commit containing only the agreed code-only publication.

- [ ] **Step 1: Stage only the intended paths**

Run:

```powershell
git add -- '.gitignore' 'README.md' 'requirements.txt' 'download_data.ps1' 'download_data.sh' 'run_all.ps1' 'run_all.sh' 'pipeline' 'docs/superpowers'
git status --short
```

Expected: no PDF, MP4, `data/`, `outputs/`, `figs/`, `tmp/`, cache, or environment path is staged.

- [ ] **Step 2: Inspect the staged file list and diff summary**

Run:

```powershell
git diff --cached --name-status
git diff --cached --stat
```

Expected: only the explicit paths from Step 1 appear.

- [ ] **Step 3: Commit the source publication**

Run:

```powershell
git commit -m "Publish CFPB complaint retrieval and classification pipeline"
```

Expected: a new commit on `main` containing the code-only repository.

### Task 5: Connect and publish to GitHub

**Files:**
- Modify: `.git/config` through Git remote commands.

**Interfaces:**
- Consumes: local `main` commits and authenticated GitHub CLI session.
- Produces: `origin/main` at the requested public GitHub repository.

- [ ] **Step 1: Configure and verify the remote**

Run:

```powershell
git remote add origin 'https://github.com/lhc060105louis-source/cfpb-complaint-retrieval-classification.git'
git remote -v
```

Expected: fetch and push URLs both match the requested repository.

- [ ] **Step 2: Push `main` with upstream tracking**

Run:

```powershell
git push -u origin main
```

Expected: `main` is created on GitHub and tracks `origin/main`.

- [ ] **Step 3: Verify the published repository and latest commit**

Run:

```powershell
gh repo view lhc060105louis-source/cfpb-complaint-retrieval-classification --json url,defaultBranchRef,isEmpty
gh api repos/lhc060105louis-source/cfpb-complaint-retrieval-classification/commits/main --jq '.sha + " " + .commit.message'
git status -sb
```

Expected: the repository is non-empty, its default branch is `main`, the latest remote commit matches the local source-publication commit, and the working tree has no unignored changes.
