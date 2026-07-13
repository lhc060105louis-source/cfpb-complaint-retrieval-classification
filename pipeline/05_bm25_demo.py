from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import (
    ID_COL,
    TEXT_COL,
    Paths,
    RuntimeLimits,
    dump_json,
    filter_usable,
    get_logger,
    load_split,
    shorten,
    stage,
    stream_csv,
)


_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    no_punct = _NON_ALNUM.sub(" ", lowered)
    collapsed = _WHITESPACE.sub(" ", no_punct).strip()
    return collapsed.split() if collapsed else []


@dataclass
class DemoConfig:
    topk: int = 5
    num_queries: int = 2
    snippet_chars: int = 220


@dataclass
class BM25Index:
    bm25: BM25Okapi
    train_ids: list[str]

    @classmethod
    def build(cls, train_ids: Iterable[str], train_texts: Iterable[str]) -> "BM25Index":
        ids = list(train_ids)
        tokenized = [tokenize(text) for text in tqdm(train_texts, desc="Tokenize train")]
        return cls(bm25=BM25Okapi(tokenized), train_ids=ids)

    def topk(self, query_text: str, k: int) -> list[tuple[int, float, str]]:
        scores = self.bm25.get_scores(tokenize(query_text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(idx, float(scores[idx]), self.train_ids[idx]) for idx in ranked]


def main() -> int:
    log = get_logger("05_bm25_demo")
    paths = Paths.from_env().ensure()
    limits = RuntimeLimits.from_env()
    config = DemoConfig(
        topk=int(os.environ.get("CS410_BM25_TOPK", "5")),
        num_queries=int(os.environ.get("CS410_BM25_NUM_QUERIES", "2")),
    )

    with stage(log, "read_csv"):
        df = stream_csv(
            paths.data_path,
            usecols=(ID_COL, TEXT_COL),
            chunksize=limits.csv_chunksize,
            max_rows=limits.max_rows,
            sample_bytes=limits.sample_bytes_for_estimate,
        )
        log.info("rows after csv read: %d", len(df))

    df = filter_usable(df, text_col=TEXT_COL, label_col=None)
    df[ID_COL] = df[ID_COL].astype(str)
    id_to_text = dict(zip(df[ID_COL], df[TEXT_COL]))

    splits = load_split(paths.split_path)
    train_ids = [cid for cid in splits["train"] if cid in id_to_text]
    test_ids = [cid for cid in splits["test"] if cid in id_to_text]

    with stage(log, "build_bm25"):
        index = BM25Index.build(train_ids, (id_to_text[cid] for cid in train_ids))

    query_ids = test_ids[: config.num_queries]
    rows: list[dict] = []
    json_blocks: list[dict] = []
    for qid in tqdm(query_ids, desc="BM25 queries"):
        qtext = id_to_text[qid]
        hits_block: list[dict] = []
        for rank, (idx, score, hit_id) in enumerate(index.topk(qtext, config.topk), start=1):
            snippet = shorten(id_to_text[hit_id], config.snippet_chars)
            hits_block.append({"rank": rank, "hit_id": hit_id, "score": score, "hit_snippet": snippet})
            rows.append(
                {
                    "query_id": qid,
                    "query_snippet": shorten(qtext, config.snippet_chars),
                    "rank": rank,
                    "hit_id": hit_id,
                    "bm25_score": score,
                    "hit_snippet": snippet,
                }
            )
        json_blocks.append(
            {"query_id": qid, "query_snippet": shorten(qtext, config.snippet_chars), "topk": hits_block}
        )

    table_path = paths.out_dir / "retrieval_demo_table.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    json_path = paths.out_dir / "retrieval_demo.json"
    dump_json(json_blocks, json_path, indent=2)

    log.info("Saved: %s", table_path)
    log.info("Saved: %s", json_path)
    log.info("Preview (first 5 rows):\n%s", pd.DataFrame(rows).head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
