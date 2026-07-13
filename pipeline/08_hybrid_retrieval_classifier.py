from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import ID_COL, LABEL_COL, Paths, get_logger, stage


@dataclass
class HybridConfig:
    require_unanimous_topk: int = 3


def majority_vote(labels: list[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


def aggregate_evidence(evidence: pd.DataFrame, config: HybridConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for query_id, group in evidence.groupby("query_id", sort=False):
        labels = group.sort_values("rank")["hit_issue"].astype(str).tolist()
        rows.append(
            {
                ID_COL: query_id,
                "retrieval_top1_issue": labels[0],
                "retrieval_vote_issue": majority_vote(labels),
                "top3_all_agree": len(set(labels[: config.require_unanimous_topk])) == 1,
            }
        )
    return pd.DataFrame(rows)


def merge_and_override(baseline_preds: pd.DataFrame, votes: pd.DataFrame) -> pd.DataFrame:
    merged = baseline_preds.merge(votes, on=ID_COL, how="left")
    merged["pred_issue"] = merged["pred_issue"].astype(str)
    merged["hybrid_pred_issue"] = merged["pred_issue"]
    mask = merged["top3_all_agree"].fillna(False)
    merged.loc[mask, "hybrid_pred_issue"] = merged.loc[mask, "retrieval_vote_issue"]
    return merged


def compute_split_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for split_name, part in merged.groupby("split"):
        y_true = part[LABEL_COL].astype(str)
        y_pred = part["hybrid_pred_issue"].astype(str)
        rows.append(
            {
                "split": split_name,
                "n": int(len(part)),
                "retrieval_override_rate": float(part["top3_all_agree"].fillna(False).mean()),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            }
        )
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


def main() -> int:
    log = get_logger("08_hybrid")
    paths = Paths.from_env().ensure()

    baseline_path = paths.out_dir / "tfidf_logreg_issue_predictions.csv"
    evidence_path = paths.out_dir / "rac_retrieval_evidence.csv"
    out_metrics = paths.out_dir / "hybrid_retrieval_issue_metrics.csv"
    out_preds = paths.out_dir / "hybrid_retrieval_issue_predictions.csv"

    with stage(log, "read_inputs"):
        baseline_preds = pd.read_csv(baseline_path, dtype={ID_COL: str})
        evidence = pd.read_csv(evidence_path, dtype={"query_id": str, "hit_id": str})
        log.info("baseline rows=%d, evidence rows=%d", len(baseline_preds), len(evidence))

    config = HybridConfig()
    with stage(log, "aggregate_evidence"):
        votes = aggregate_evidence(evidence, config)
    with stage(log, "merge_override"):
        merged = merge_and_override(baseline_preds, votes)
    with stage(log, "compute_metrics"):
        metrics_df = compute_split_metrics(merged)

    metrics_df.to_csv(out_metrics, index=False)
    merged.to_csv(out_preds, index=False)
    log.info("Saved: %s", out_metrics)
    log.info("Saved: %s", out_preds)
    log.info("\n%s", metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
