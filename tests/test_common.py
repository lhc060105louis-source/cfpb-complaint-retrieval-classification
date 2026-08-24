from __future__ import annotations

import pandas as pd

from pipeline.common import LABEL_COL, TEXT_COL, filter_usable


def test_filter_usable_removes_missing_text_and_labels() -> None:
    frame = pd.DataFrame(
        {
            TEXT_COL: ["valid complaint", None, "   ", "nan", "second valid complaint"],
            LABEL_COL: ["Fees", "Fees", "Fees", "Fees", None],
        }
    )

    filtered = filter_usable(frame, text_col=TEXT_COL, label_col=LABEL_COL)

    assert filtered[TEXT_COL].tolist() == ["valid complaint"]
    assert filtered[LABEL_COL].tolist() == ["Fees"]
