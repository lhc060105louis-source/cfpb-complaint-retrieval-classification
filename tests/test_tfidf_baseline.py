from __future__ import annotations

import importlib

import pandas as pd

from pipeline.common import LABEL_COL, TEXT_COL


baseline = importlib.import_module("pipeline.06_tfidf_logreg_baseline")


def test_pipeline_fits_and_evaluates_held_out_examples() -> None:
    frame = pd.DataFrame(
        {
            TEXT_COL: [
                "charged an unexpected card fee",
                "card fee appeared on my statement",
                "charged another annual card fee",
                "credit report contains a wrong account",
                "wrong account remains on credit report",
                "credit bureau lists incorrect account",
                "unexpected fee charged to my card",
                "incorrect account on my credit report",
            ],
            LABEL_COL: ["Fees"] * 3 + ["Credit reporting"] * 3 + ["Fees", "Credit reporting"],
            "split": ["train"] * 6 + ["test"] * 2,
        }
    )
    model = baseline.build_pipeline(
        baseline.VectorizerConfig(min_df=1, max_df=1.0, max_features=100),
        baseline.ClassifierConfig(max_iter=100, n_jobs=1, verbose=0),
    )
    train = frame[frame["split"] == "train"]
    model.fit(train[TEXT_COL], train[LABEL_COL])

    metrics = baseline.evaluate(model, frame, "test")

    assert metrics.n == 2
    assert metrics.predictions.tolist() == ["Fees", "Credit reporting"]
    assert metrics.accuracy == 1.0
