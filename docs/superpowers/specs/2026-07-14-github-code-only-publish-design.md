# GitHub Code-Only Publication Design

## Objective

Publish the CFPB complaint retrieval and classification project to
`lhc060105louis-source/cfpb-complaint-retrieval-classification` as a clean,
reproducible, code-only repository.

## Repository contents

The repository root will contain the current contents of `cs410_release/`:

- `README.md`
- `requirements.txt`
- cross-platform data download and pipeline runner scripts
- the `pipeline/` Python package and numbered pipeline stages

The existing README will remain the main project documentation, with path and
formatting corrections if needed after the files are placed at the root.

## Excluded contents

The following local artifacts will not be committed:

- `CS 410 Project Final Report.pdf`
- `Cs410 pre video.mp4`
- downloaded CFPB data and generated subsets
- model outputs, generated figures, checkpoints, logs, caches, and virtual
  environments
- temporary review files under `tmp/`

A root `.gitignore` will encode these exclusions, including the large runtime
artifacts documented in the README.

## Git and GitHub strategy

The local project directory will be initialized as a Git repository with
`main` as its initial branch. The empty public GitHub repository supplied by
the user will be configured as `origin`. Intended files will be staged
explicitly, committed with a concise initial commit message, and pushed to
`origin/main`.

Because the remote repository is empty, the initial publication will be pushed
directly to `main`; no pull request is necessary for this bootstrap commit.

## Validation

Before publication:

1. Confirm the staged file list contains source and project documentation only.
2. Confirm neither the PDF nor video is tracked.
3. Compile all Python modules with `python -m compileall`.
4. Run lightweight CLI/header checks that do not require the multi-gigabyte
   CFPB dataset.
5. Confirm the remote and branch, push, then verify the GitHub repository URL
   and latest commit.

## Non-goals

This publication does not change model behavior, retrain models, upload the
CFPB dataset, add generated results, or publish the report and presentation
media.
