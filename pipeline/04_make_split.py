from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import (
    DATE_COL,
    ID_COL,
    TEXT_COL,
    Paths,
    RuntimeLimits,
    dump_json,
    filter_usable,
    get_logger,
    stage,
    stream_csv,
)


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.70
    val: float = 0.15

    def boundaries(self, n: int) -> tuple[int, int]:
        n_train = int(n * self.train)
        n_val = int(n * self.val)
        return n_train, n_train + n_val


@dataclass
class SplitArtifacts:
    splits: dict[str, list[str]]
    stats: pd.DataFrame


def build_splits(df: pd.DataFrame, fractions: SplitFractions) -> SplitArtifacts:
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise ValueError("no rows available to split")
    train_end, val_end = fractions.boundaries(n)
    parts = {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:],
    }
    splits = {name: part[ID_COL].astype(str).tolist() for name, part in parts.items()}
    stats = pd.DataFrame(
        [
            {
                "split": name,
                "n": len(part),
                "date_min": str(part[DATE_COL].min()) if len(part) else None,
                "date_max": str(part[DATE_COL].max()) if len(part) else None,
            }
            for name, part in parts.items()
        ]
    )
    return SplitArtifacts(splits=splits, stats=stats)


def main() -> int:
    log = get_logger("04_make_split")
    paths = Paths.from_env().ensure()
    limits = RuntimeLimits.from_env()
    counts_name = os.environ.get("CS410_SPLIT_COUNTS_NAME", "split_counts_full.csv")

    with stage(log, "read_csv"):
        df = stream_csv(
            paths.data_path,
            usecols=(DATE_COL, TEXT_COL, ID_COL),
            chunksize=limits.csv_chunksize,
            max_rows=limits.max_rows,
            sample_bytes=limits.sample_bytes_for_estimate,
        )
        log.info("rows after csv read: %d", len(df))

    with stage(log, "filter_and_parse_date"):
        df = filter_usable(df, text_col=TEXT_COL, label_col=None)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL])
        log.info("rows after filter: %d", len(df))

    with stage(log, "build_splits"):
        artifacts = build_splits(df, SplitFractions())

    dump_json(artifacts.splits, paths.split_path)
    artifacts.stats.to_csv(paths.out_dir / counts_name, index=False)

    log.info("split file: %s", paths.split_path)
    log.info("counts file: %s", paths.out_dir / counts_name)
    log.info("\n%s", artifacts.stats.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
