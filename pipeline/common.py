from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd
from tqdm import tqdm


TEXT_COL = "Consumer complaint narrative"
LABEL_COL = "Issue"
ID_COL = "Complaint ID"
DATE_COL = "Date received"
PRODUCT_COL = "Product"


@dataclass(frozen=True)
class Paths:
    data_path: Path
    out_dir: Path
    fig_dir: Path
    split_path: Path

    @classmethod
    def from_env(cls) -> "Paths":
        out_dir = Path(os.environ.get("CS410_OUT_DIR", "outputs"))
        fig_dir = Path(os.environ.get("CS410_FIG_DIR", "figs"))
        split_name = os.environ.get("CS410_SPLIT_NAME", "split_full.json")
        return cls(
            data_path=Path(os.environ.get("CS410_DATA_PATH", "data/complaints.csv")),
            out_dir=out_dir,
            fig_dir=fig_dir,
            split_path=Path(os.environ.get("CS410_SPLIT_PATH", str(out_dir / split_name))),
        )

    def ensure(self) -> "Paths":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class RuntimeLimits:
    max_rows: int | None = None
    csv_chunksize: int = 200_000
    sample_bytes_for_estimate: int = 50 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "RuntimeLimits":
        raw = os.environ.get("CS410_MAX_ROWS", "").strip()
        return cls(max_rows=int(raw) if raw else None)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@contextmanager
def stage(logger: logging.Logger, label: str) -> Iterator[None]:
    logger.info("[%s] start", label)
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        logger.info("[%s] done in %.2fs", label, elapsed)


def estimate_total_rows(path: Path, sample_bytes: int) -> int:
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        sample = fh.read(sample_bytes)
    sample_rows = max(sample.count(b"\n"), 1)
    return int(file_size / (sample_bytes / sample_rows))


def stream_csv(
    path: Path,
    usecols: Sequence[str] | None = None,
    chunksize: int = 200_000,
    max_rows: int | None = None,
    sample_bytes: int = 50 * 1024 * 1024,
    desc: str = "Reading CSV",
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    estimated = estimate_total_rows(path, sample_bytes)
    if max_rows is not None:
        estimated = min(estimated, max_rows)
    reader = pd.read_csv(
        path,
        engine="c",
        on_bad_lines="skip",
        encoding_errors="ignore",
        usecols=list(usecols) if usecols else None,
        dtype=str,
        chunksize=chunksize,
    )
    chunks: list[pd.DataFrame] = []
    rows_kept = 0
    with tqdm(total=estimated, unit=" rows", unit_scale=True, desc=desc) as bar:
        for chunk in reader:
            if max_rows is not None and rows_kept + len(chunk) >= max_rows:
                chunk = chunk.iloc[: max_rows - rows_kept]
                chunks.append(chunk)
                rows_kept += len(chunk)
                bar.update(len(chunk))
                break
            chunks.append(chunk)
            rows_kept += len(chunk)
            bar.update(len(chunk))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=list(usecols or []))


def filter_usable(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
    label_col: str | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].notna()]
    df = df[df[text_col].str.strip() != ""]
    df = df[df[text_col].str.lower() != "nan"]
    if label_col is not None and label_col in df.columns:
        df[label_col] = df[label_col].astype(str)
        df = df[df[label_col].str.strip() != ""]
        df = df[df[label_col].str.lower() != "nan"]
    return df


def load_split(split_path: Path) -> dict[str, list[str]]:
    with split_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def attach_split(df: pd.DataFrame, split_path: Path, id_col: str = ID_COL) -> pd.DataFrame:
    splits = load_split(split_path)
    id_to_split: dict[str, str] = {}
    for split_name, ids in splits.items():
        id_to_split.update({str(cid): split_name for cid in ids})
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    df["split"] = df[id_col].map(id_to_split)
    return df[df["split"].notna()]


def shorten(text: Any, max_chars: int) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= max_chars else s[:max_chars] + "..."


def dump_json(obj: Any, path: Path, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)


@dataclass
class StageReport:
    name: str
    elapsed: float
    extra: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        out = {"stage": self.name, "elapsed_s": round(self.elapsed, 3)}
        out.update(self.extra)
        return out
