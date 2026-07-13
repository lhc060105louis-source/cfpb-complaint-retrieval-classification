from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import (
    ID_COL,
    LABEL_COL,
    TEXT_COL,
    Paths,
    RuntimeLimits,
    attach_split,
    filter_usable,
    get_logger,
    stage,
    stream_csv,
)


@dataclass(frozen=True)
class VectorizerConfig:
    min_df: int = 3
    max_df: float = 0.90
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 100_000
    sublinear_tf: bool = True


@dataclass(frozen=True)
class ClassifierConfig:
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    solver: str = "liblinear"
    random_state: int = 410
    n_jobs: int = -1
    verbose: int = 10


def build_pipeline(vec: VectorizerConfig, clf: ClassifierConfig) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    min_df=vec.min_df,
                    max_df=vec.max_df,
                    ngram_range=vec.ngram_range,
                    max_features=vec.max_features,
                    sublinear_tf=vec.sublinear_tf,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=clf.max_iter,
                        class_weight=clf.class_weight,
                        solver=clf.solver,
                        random_state=clf.random_state,
                    ),
                    n_jobs=clf.n_jobs,
                    verbose=clf.verbose,
                ),
            ),
        ]
    )


@dataclass
class SplitMetrics:
    split: str
    n: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    predictions: pd.Series = field(default_factory=lambda: pd.Series(dtype=str))

    def to_record(self) -> dict[str, float | int | str]:
        return {
            "split": self.split,
            "n": self.n,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
        }


def evaluate(model, df: pd.DataFrame, split_name: str) -> SplitMetrics:
    part = df[df["split"] == split_name]
    y_true = part[LABEL_COL]
    y_pred = pd.Series(model.predict(part[TEXT_COL]), index=part.index)
    return SplitMetrics(
        split=split_name,
        n=int(len(part)),
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        predictions=y_pred,
    )


def main() -> int:
    log = get_logger("06_baseline")
    paths = Paths.from_env().ensure()
    limits = RuntimeLimits.from_env()

    out_metrics = paths.out_dir / "tfidf_logreg_issue_metrics.csv"
    out_report = paths.out_dir / "tfidf_logreg_issue_classification_report.csv"
    out_preds = paths.out_dir / "tfidf_logreg_issue_predictions.csv"

    with stage(log, "load"):
        df = stream_csv(
            paths.data_path,
            usecols=(ID_COL, TEXT_COL, LABEL_COL),
            chunksize=limits.csv_chunksize,
            max_rows=limits.max_rows,
            sample_bytes=limits.sample_bytes_for_estimate,
        )
        df = filter_usable(df, text_col=TEXT_COL, label_col=LABEL_COL)
        df = attach_split(df, paths.split_path, id_col=ID_COL)
        log.info("rows after split attach: %d", len(df))

    train_df = df[df["split"] == "train"]
    log.info("train rows=%d, distinct issues=%d", len(train_df), train_df[LABEL_COL].nunique())

    model = build_pipeline(VectorizerConfig(), ClassifierConfig())

    log.info("[fit] starting OvR LogReg...")
    started = time.perf_counter()
    model.fit(train_df[TEXT_COL], train_df[LABEL_COL])
    log.info("[fit] done in %.1fs", time.perf_counter() - started)

    metric_records: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    for split_name in ("val", "test"):
        with stage(log, f"eval_{split_name}"):
            m = evaluate(model, df, split_name)
            metric_records.append(m.to_record())
            part = df[df["split"] == split_name].copy()
            part["pred_issue"] = m.predictions
            pred_frames.append(part[[ID_COL, "split", LABEL_COL, "pred_issue", TEXT_COL]])

    metrics_df = pd.DataFrame(metric_records)
    metrics_df.to_csv(out_metrics, index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(out_preds, index=False)

    test_df = df[df["split"] == "test"]
    test_pred = model.predict(test_df[TEXT_COL])
    report = classification_report(test_df[LABEL_COL], test_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_report)

    log.info("Saved: %s", out_metrics)
    log.info("Saved: %s", out_report)
    log.info("Saved: %s", out_preds)
    log.info("\n%s", metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
