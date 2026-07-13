from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import Paths, get_logger


@dataclass(frozen=True)
class ModelMetricSource:
    label: str
    path: Path


def collect(sources: Iterable[ModelMetricSource]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for src in sources:
        df = pd.read_csv(src.path)
        df["model"] = src.label
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    cols = ["model", "split", "n", "accuracy", "macro_f1", "weighted_f1"]
    if "retrieval_override_rate" in df.columns:
        cols.append("retrieval_override_rate")
    return df[[c for c in cols if c in df.columns]]


def plot_test_comparison(test_df: pd.DataFrame, target: Path) -> None:
    test_df = test_df.sort_values("weighted_f1", ascending=True)
    plt.figure(figsize=(7.2, 3.8))
    plt.barh(test_df["model"], test_df["weighted_f1"])
    plt.xlabel("Weighted F1 on test")
    plt.title("Issue Classification Model Comparison")
    plt.xlim(0, max(0.60, test_df["weighted_f1"].max() + 0.05))
    plt.grid(axis="x", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, dpi=300, bbox_inches="tight")
    plt.close()


def render_summary(test_df: pd.DataFrame) -> list[str]:
    rows = {row["model"]: row for _, row in test_df.iterrows()}
    baseline = rows.get("TF-IDF + LogReg")
    rac = rows.get("RAC: augmented text")
    hybrid = rows.get("Hybrid retrieval")
    best = test_df.sort_values("weighted_f1", ascending=False).iloc[0]
    lines = ["Issue classification results on the held-out test split:"]
    if baseline is not None:
        lines.append(
            f"- TF-IDF + Logistic Regression baseline: accuracy={baseline['accuracy']:.3f}, "
            f"macro-F1={baseline['macro_f1']:.3f}, weighted-F1={baseline['weighted_f1']:.3f}."
        )
    if rac is not None:
        lines.append(
            f"- Retrieval-augmented text classifier: accuracy={rac['accuracy']:.3f}, "
            f"macro-F1={rac['macro_f1']:.3f}, weighted-F1={rac['weighted_f1']:.3f}."
        )
    if hybrid is not None:
        lines.append(
            f"- Conservative hybrid retrieval classifier: accuracy={hybrid['accuracy']:.3f}, "
            f"macro-F1={hybrid['macro_f1']:.3f}, weighted-F1={hybrid['weighted_f1']:.3f}."
        )
    lines.append(f"Best test weighted-F1 in the current run: {best['model']} ({best['weighted_f1']:.3f}).")
    lines.append(
        "Interpretation: the plain TF-IDF baseline is currently strongest on test; the retrieval variants "
        "are implemented and provide evidence, but naive retrieval augmentation does not yet improve final "
        "held-out performance."
    )
    return lines


def main() -> int:
    log = get_logger("09_compare")
    paths = Paths.from_env().ensure()

    sources = [
        ModelMetricSource("TF-IDF + LogReg", paths.out_dir / "tfidf_logreg_issue_metrics.csv"),
        ModelMetricSource("RAC: augmented text", paths.out_dir / "rac_issue_metrics.csv"),
        ModelMetricSource("Hybrid retrieval", paths.out_dir / "hybrid_retrieval_issue_metrics.csv"),
    ]

    comparison = collect(sources)
    out_table = paths.out_dir / "classification_model_comparison.csv"
    comparison.to_csv(out_table, index=False)

    test_df = comparison[comparison["split"] == "test"].copy()
    out_fig = paths.fig_dir / "classification_model_comparison.png"
    plot_test_comparison(test_df, out_fig)

    out_text = paths.out_dir / "classification_report_ready.txt"
    summary_lines = render_summary(test_df)
    out_text.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    log.info("Saved: %s", out_table)
    log.info("Saved: %s", out_text)
    log.info("Saved: %s", out_fig)
    log.info("\n%s", comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
