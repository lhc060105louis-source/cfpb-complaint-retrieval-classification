from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import Paths, get_logger


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value, default: int = 0) -> int:
    f = _to_float(value)
    return int(f) if f is not None else default


def _fmt(value, ndigits: int = 1) -> str:
    return "NA" if value is None else f"{value:.{ndigits}f}"


@dataclass
class EdaSnapshot:
    rows: int
    date_min: str
    date_max: str
    words_p10: float | None
    words_p50: float | None
    words_p90: float | None
    chars_p10: float | None
    chars_p50: float | None
    chars_p90: float | None
    n_products: int
    n_issues: int
    top_issue_coverage: float
    top_product_coverage: float

    def render(self) -> list[str]:
        return [
            f"Usable narratives (after robust parsing + filtering): N = {self.rows}",
            f"Time coverage: {self.date_min} to {self.date_max}",
            f"Narrative length (words): P10={_fmt(self.words_p10)}, P50={_fmt(self.words_p50)}, P90={_fmt(self.words_p90)}",
            f"Narrative length (chars): P10={_fmt(self.chars_p10)}, P50={_fmt(self.chars_p50)}, P90={_fmt(self.chars_p90)}",
            f"Label space size: #Products={self.n_products}, #Issues={self.n_issues}",
            f"Top-10 Issue coverage: {self.top_issue_coverage * 100:.2f}% of narratives",
            f"Top-10 Product coverage: {self.top_product_coverage * 100:.2f}% of narratives",
        ]


def _read_summary(path: Path) -> pd.Series:
    return pd.read_csv(path, header=None, index_col=0).squeeze("columns")


def _read_label_counts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["label", "count"])
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    return df


def build_snapshot(out_dir: Path) -> EdaSnapshot:
    summary = _read_summary(out_dir / "eda_summary.csv")
    issues = _read_label_counts(out_dir / "top_issues.csv")
    products = _read_label_counts(out_dir / "top_products.csv")
    rows = _to_int(summary.get("rows_loaded_after_skip", summary.get("rows_in_subset_file", 0)))
    return EdaSnapshot(
        rows=rows,
        date_min=str(summary.get("date_min", "NA")),
        date_max=str(summary.get("date_max", "NA")),
        words_p10=_to_float(summary.get("text_words_p10")),
        words_p50=_to_float(summary.get("text_words_p50")),
        words_p90=_to_float(summary.get("text_words_p90")),
        chars_p10=_to_float(summary.get("text_chars_p10")),
        chars_p50=_to_float(summary.get("text_chars_p50")),
        chars_p90=_to_float(summary.get("text_chars_p90")),
        n_products=_to_int(summary.get("num_unique_products", 0)),
        n_issues=_to_int(summary.get("num_unique_issues", 0)),
        top_issue_coverage=(issues["count"].head(10).sum() / rows) if rows else 0.0,
        top_product_coverage=(products["count"].head(10).sum() / rows) if rows else 0.0,
    )


def main() -> int:
    log = get_logger("02_eda_report")
    paths = Paths.from_env().ensure()
    snapshot = build_snapshot(paths.out_dir)
    text = "\n".join(snapshot.render()) + "\n"
    target = paths.out_dir / "eda_report_ready.txt"
    target.write_text(text, encoding="utf-8")
    for line in snapshot.render():
        log.info(line)
    log.info("wrote %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
