from __future__ import annotations

import importlib

import pandas as pd

from pipeline.common import DATE_COL, ID_COL


make_split = importlib.import_module("pipeline.04_make_split")


def test_build_splits_orders_rows_chronologically() -> None:
    frame = pd.DataFrame(
        {
            ID_COL: ["late", "early", "middle", "later"],
            DATE_COL: pd.to_datetime(
                ["2024-04-01", "2024-01-01", "2024-02-01", "2024-03-01"]
            ),
        }
    )

    artifacts = make_split.build_splits(
        frame,
        make_split.SplitFractions(train=0.50, val=0.25),
    )

    assert artifacts.splits == {
        "train": ["early", "middle"],
        "val": ["later"],
        "test": ["late"],
    }
