from __future__ import annotations

import importlib

import pandas as pd


hybrid = importlib.import_module("pipeline.08_hybrid_retrieval_classifier")


def test_incomplete_neighbor_set_is_not_unanimous_top_k() -> None:
    evidence = pd.DataFrame(
        [
            {
                "query_id": "query-1",
                "rank": 1,
                "hit_issue": "Incorrect information on your report",
            }
        ]
    )

    votes = hybrid.aggregate_evidence(
        evidence,
        hybrid.HybridConfig(require_unanimous_topk=3),
    )

    assert votes.loc[0, "top3_all_agree"] == False  # noqa: E712


def test_three_matching_neighbors_are_unanimous_top_k() -> None:
    evidence = pd.DataFrame(
        [
            {"query_id": "query-1", "rank": rank, "hit_issue": "Fees"}
            for rank in (1, 2, 3)
        ]
    )

    votes = hybrid.aggregate_evidence(
        evidence,
        hybrid.HybridConfig(require_unanimous_topk=3),
    )

    assert votes.loc[0, "top3_all_agree"] == True  # noqa: E712
    assert votes.loc[0, "retrieval_vote_issue"] == "Fees"
