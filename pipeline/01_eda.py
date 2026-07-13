from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import (
    DATE_COL,
    ID_COL,
    LABEL_COL,
    PRODUCT_COL,
    TEXT_COL,
    Paths,
    RuntimeLimits,
    filter_usable,
    get_logger,
    stage,
    stream_csv,
)


ISSUE_COL = LABEL_COL
USECOLS = (DATE_COL, PRODUCT_COL, ISSUE_COL, TEXT_COL, ID_COL)


@dataclass
class LabelDistribution:
    column: str
    top: pd.Series
    total_rows: int

    def percent(self) -> pd.Series:
        return (self.top / max(self.total_rows, 1) * 100.0).sort_values()


@dataclass
class TextLengthStats:
    chars_p10: float
    chars_p50: float
    chars_p90: float
    words_p10: float
    words_p50: float
    words_p90: float

    @classmethod
    def from_series(cls, char_lengths: pd.Series, word_lengths: pd.Series) -> "TextLengthStats":
        return cls(
            chars_p10=float(char_lengths.quantile(0.10)),
            chars_p50=float(char_lengths.quantile(0.50)),
            chars_p90=float(char_lengths.quantile(0.90)),
            words_p10=float(word_lengths.quantile(0.10)),
            words_p50=float(word_lengths.quantile(0.50)),
            words_p90=float(word_lengths.quantile(0.90)),
        )


def wrap_labels(labels: Iterable[str], width: int = 38) -> list[str]:
    return ["\n".join(textwrap.wrap(str(value), width=width)) for value in labels]


def save_figure(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, dpi=300, bbox_inches="tight")


def plot_label_distribution(
    dist: LabelDistribution,
    title: str,
    target: Path,
    width: int,
    figsize: tuple[float, float],
) -> None:
    pct = dist.percent()
    labels = wrap_labels(pct.index.tolist(), width=width)
    positions = list(range(len(pct)))
    plt.figure(figsize=figsize)
    plt.barh(positions, pct.values, height=0.55)
    plt.yticks(positions, labels)
    plt.xlabel("Percent of narratives (%)")
    plt.title(title)
    plt.grid(axis="x", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    save_figure(target)
    plt.close()


def plot_word_length_histogram(word_lengths: pd.Series, target: Path) -> None:
    plt.figure(figsize=(7.0, 3.8))
    word_lengths.clip(upper=word_lengths.quantile(0.99)).hist(bins=50)
    plt.title("Narrative Length (words), clipped at 99th percentile")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.tight_layout()
    save_figure(target)
    plt.close()


def plot_monthly_counts(monthly: pd.Series, target: Path) -> None:
    plt.figure(figsize=(7.0, 3.4))
    monthly.plot(linewidth=1.0)
    plt.title("Complaints by Month")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.tight_layout()
    save_figure(target)
    plt.close()


def main() -> int:
    log = get_logger("01_eda")
    paths = Paths.from_env().ensure()
    limits = RuntimeLimits.from_env()

    with stage(log, "read_csv"):
        df = stream_csv(
            paths.data_path,
            usecols=USECOLS,
            chunksize=limits.csv_chunksize,
            max_rows=limits.max_rows,
            sample_bytes=limits.sample_bytes_for_estimate,
        )
        log.info("rows after csv read: %d", len(df))

    with stage(log, "filter_usable"):
        df = filter_usable(df, text_col=TEXT_COL, label_col=None)
        log.info("rows after usable-text filter: %d", len(df))

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["text_len_chars"] = df[TEXT_COL].str.len()
    df["text_len_words"] = df[TEXT_COL].str.split().str.len()

    text_stats = TextLengthStats.from_series(df["text_len_chars"], df["text_len_words"])
    products = LabelDistribution(PRODUCT_COL, df[PRODUCT_COL].value_counts(dropna=False).head(10), len(df))
    issues = LabelDistribution(ISSUE_COL, df[ISSUE_COL].value_counts(dropna=False).head(10), len(df))

    df_time = df.dropna(subset=[DATE_COL]).copy()
    df_time["month"] = df_time[DATE_COL].dt.to_period("M").astype(str)
    monthly = df_time["month"].value_counts().sort_index()

    summary = {
        "rows_loaded_after_skip": int(len(df)),
        "date_min": str(df_time[DATE_COL].min()) if len(df_time) else None,
        "date_max": str(df_time[DATE_COL].max()) if len(df_time) else None,
        "text_chars_p10": text_stats.chars_p10,
        "text_chars_p50": text_stats.chars_p50,
        "text_chars_p90": text_stats.chars_p90,
        "text_words_p10": text_stats.words_p10,
        "text_words_p50": text_stats.words_p50,
        "text_words_p90": text_stats.words_p90,
        "num_unique_products": int(df[PRODUCT_COL].nunique(dropna=True)),
        "num_unique_issues": int(df[ISSUE_COL].nunique(dropna=True)),
    }
    pd.Series(summary).to_csv(paths.out_dir / "eda_summary.csv")
    products.top.to_csv(paths.out_dir / "top_products.csv")
    issues.top.to_csv(paths.out_dir / "top_issues.csv")
    monthly.to_csv(paths.out_dir / "month_counts.csv")

    plt.rcParams["ytick.major.pad"] = 10

    with stage(log, "plot_products"):
        plot_label_distribution(
            products,
            "Top 10 Products (share of narratives)",
            paths.fig_dir / "top_products.png",
            width=40,
            figsize=(7.5, 5.2),
        )
    with stage(log, "plot_issues"):
        plot_label_distribution(
            issues,
            "Top 10 Issues (share of narratives)",
            paths.fig_dir / "top_issues.png",
            width=42,
            figsize=(7.5, 5.8),
        )
    with stage(log, "plot_word_lengths"):
        plot_word_length_histogram(df["text_len_words"], paths.fig_dir / "text_len_words_hist.png")
    with stage(log, "plot_monthly"):
        plot_monthly_counts(monthly, paths.fig_dir / "month_counts.png")

    log.info("EDA done. Figures in %s, tables in %s", paths.fig_dir, paths.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
