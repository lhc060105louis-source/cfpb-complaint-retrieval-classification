from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import Paths, get_logger


def list_columns(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    head = pd.read_csv(
        path,
        nrows=0,
        engine="c",
        on_bad_lines="skip",
        encoding_errors="ignore",
    )
    return list(head.columns)


def main() -> int:
    log = get_logger("00_check_header")
    paths = Paths.from_env().ensure()
    cols = list_columns(paths.data_path)
    log.info("Columns in %s:", paths.data_path)
    for idx, name in enumerate(cols):
        log.info("  %2d %r", idx, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
