from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import faiss
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

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
    shorten,
    stage,
    stream_csv,
)


@dataclass(frozen=True)
class RetrieverConfig:
    svd_dim: int = 256
    ivf_nlist: int = 1024
    ivf_nprobe: int = 16
    ivf_train_samples: int = 262_144
    topk: int = 3
    snippet_chars: int = 350
    encode_batch: int = 50_000
    add_batch: int = 200_000
    search_batch: int = 8192

    @classmethod
    def from_env(cls) -> "RetrieverConfig":
        return cls(
            svd_dim=int(os.environ.get("CS410_SVD_DIM", "256")),
            ivf_nlist=int(os.environ.get("CS410_IVF_NLIST", "1024")),
            ivf_nprobe=int(os.environ.get("CS410_IVF_NPROBE", "16")),
            ivf_train_samples=int(os.environ.get("CS410_IVF_TRAIN_SAMPLES", "262144")),
        )


@dataclass(frozen=True)
class ClassifierConfig:
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    solver: str = "liblinear"
    random_state: int = 410
    n_jobs: int = int(os.environ.get("CS410_OVR_N_JOBS", "16"))
    verbose: int = 10
    vec_min_df: int = 3
    vec_max_df: float = 0.90
    vec_ngram: tuple[int, int] = (1, 2)
    vec_max_features: int = 120_000


class DenseRetrieverVectorizer:
    def __init__(self, tfidf: TfidfVectorizer, svd: TruncatedSVD) -> None:
        self.tfidf = tfidf
        self.svd = svd

    def transform(self, texts: Sequence[str], batch_size: int, desc: str) -> np.ndarray:
        out = np.empty((len(texts), self.svd.n_components), dtype=np.float32)
        for s in tqdm(range(0, len(texts), batch_size), desc=desc):
            e = min(s + batch_size, len(texts))
            block = self.svd.transform(self.tfidf.transform(texts[s:e])).astype(np.float32, copy=False)
            faiss.normalize_L2(block)
            out[s:e] = block
        return out


class FaissCosineRetriever:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._quantizer = faiss.IndexFlatIP(dim)
        self._index: faiss.IndexIVFFlat | None = None

    def build(self, vectors: np.ndarray, nlist: int, train_samples: int, nprobe: int) -> None:
        n = vectors.shape[0]
        nlist_actual = min(nlist, max(1, int(math.sqrt(n))))
        self._index = faiss.IndexIVFFlat(self._quantizer, self.dim, nlist_actual, faiss.METRIC_INNER_PRODUCT)
        sample_ids = np.random.default_rng(410).choice(n, size=min(train_samples, n), replace=False)
        self._index.train(np.ascontiguousarray(vectors[sample_ids]))
        for s in tqdm(range(0, n, 200_000), desc="faiss add"):
            self._index.add(np.ascontiguousarray(vectors[s : s + 200_000]))
        self._index.nprobe = nprobe

    def search(self, queries: np.ndarray, k: int, batch_size: int, desc: str) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("FaissCosineRetriever.build was not called")
        n = queries.shape[0]
        distances = np.empty((n, k), dtype=np.float32)
        indices = np.empty((n, k), dtype=np.int64)
        for s in tqdm(range(0, n, batch_size), desc=desc):
            e = min(s + batch_size, n)
            sims, idx = self._index.search(np.ascontiguousarray(queries[s:e]), k)
            distances[s:e] = 1.0 - sims
            indices[s:e] = idx
        return distances, indices


@dataclass
class RetrievalArtifacts:
    vectorizer: DenseRetrieverVectorizer
    retriever: FaissCosineRetriever
    train_dense: np.ndarray
    train_df: pd.DataFrame


def fit_retriever(train_df: pd.DataFrame, config: RetrieverConfig, log) -> RetrievalArtifacts:
    train_texts = train_df[TEXT_COL].tolist()

    tfidf = TfidfVectorizer(
        lowercase=True,
        min_df=3,
        max_df=0.90,
        ngram_range=(1, 2),
        max_features=60_000,
        sublinear_tf=True,
    )
    log.info("[retriever] fit TF-IDF on %d docs", len(train_texts))
    started = time.perf_counter()
    train_sparse = tfidf.fit_transform(train_texts)
    log.info(
        "[retriever] TF-IDF shape=%s nnz=%d in %.1fs",
        train_sparse.shape,
        train_sparse.nnz,
        time.perf_counter() - started,
    )

    svd = TruncatedSVD(
        n_components=min(config.svd_dim, min(train_sparse.shape) - 1),
        random_state=410,
        algorithm="randomized",
        n_iter=5,
    )
    log.info("[retriever] fit TruncatedSVD to %d dims", svd.n_components)
    started = time.perf_counter()
    svd.fit(train_sparse)
    log.info(
        "[retriever] SVD done in %.1fs, explained_variance_sum=%.4f",
        time.perf_counter() - started,
        float(svd.explained_variance_ratio_.sum()),
    )

    vectorizer = DenseRetrieverVectorizer(tfidf, svd)
    started = time.perf_counter()
    train_dense = np.empty((len(train_texts), svd.n_components), dtype=np.float32)
    for s in tqdm(range(0, len(train_texts), config.encode_batch), desc="encode train"):
        e = min(s + config.encode_batch, len(train_texts))
        block = svd.transform(train_sparse[s:e]).astype(np.float32, copy=False)
        faiss.normalize_L2(block)
        train_dense[s:e] = block
    log.info("[retriever] train_dense %s in %.1fs", train_dense.shape, time.perf_counter() - started)
    del train_sparse

    retriever = FaissCosineRetriever(dim=svd.n_components)
    log.info("[retriever] build FAISS IVFFlat (nlist target=%d)", config.ivf_nlist)
    started = time.perf_counter()
    retriever.build(train_dense, nlist=config.ivf_nlist, train_samples=config.ivf_train_samples, nprobe=config.ivf_nprobe)
    log.info("[retriever] FAISS build done in %.1fs", time.perf_counter() - started)

    return RetrievalArtifacts(vectorizer=vectorizer, retriever=retriever, train_dense=train_dense, train_df=train_df)


def collect_evidence(
    part_df: pd.DataFrame,
    artifacts: RetrievalArtifacts,
    config: RetrieverConfig,
    split_name: str,
    log,
) -> tuple[list[str], list[dict]]:
    if split_name == "train":
        queries = artifacts.train_dense
        n_neighbors = config.topk + 1
    else:
        log.info("[%s] vectorize %d queries", split_name, len(part_df))
        queries = artifacts.vectorizer.transform(
            part_df[TEXT_COL].tolist(), batch_size=config.encode_batch, desc=f"vectorize {split_name}"
        )
        n_neighbors = config.topk

    distances, indices = artifacts.retriever.search(
        queries, k=n_neighbors, batch_size=config.search_batch, desc=f"kNN {split_name}"
    )

    train = artifacts.train_df.reset_index(drop=True)
    part = part_df.reset_index(drop=True)
    augmented: list[str] = []
    evidence: list[dict] = []
    for row_pos, row in tqdm(part.iterrows(), total=len(part), desc=f"build aug {split_name}"):
        chunks: list[str] = []
        rank = 0
        for dist, idx in zip(distances[row_pos], indices[row_pos]):
            hit = train.iloc[int(idx)]
            if row["split"] == "train" and hit[ID_COL] == row[ID_COL]:
                continue
            rank += 1
            if rank > config.topk:
                break
            snippet = shorten(hit[TEXT_COL], config.snippet_chars)
            similarity = 1.0 - float(dist)
            chunks.append(f"Similar case {rank}. Issue: {hit[LABEL_COL]}. Narrative: {snippet}")
            evidence.append(
                {
                    "query_id": row[ID_COL],
                    "query_split": row["split"],
                    "rank": rank,
                    "hit_id": hit[ID_COL],
                    "hit_issue": hit[LABEL_COL],
                    "cosine_similarity": similarity,
                    "hit_snippet": snippet,
                }
            )
        augmented.append(
            f"Query narrative: {row[TEXT_COL]}\n\nRetrieved evidence:\n" + "\n".join(chunks)
        )
    return augmented, evidence


def build_augmented_dataframe(df: pd.DataFrame, retriever_cfg: RetrieverConfig, log) -> tuple[pd.DataFrame, list[dict]]:
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    log.info("Building dense SVD + FAISS IVFFlat retriever")
    artifacts = fit_retriever(train_df, retriever_cfg, log)
    parts: list[pd.DataFrame] = []
    evidence_all: list[dict] = []
    for split_name in ("train", "val", "test"):
        log.info("Building retrieval-augmented text for %s", split_name)
        part = df[df["split"] == split_name].copy()
        augmented, evidence = collect_evidence(part, artifacts, retriever_cfg, split_name, log)
        part["augmented_text"] = augmented
        parts.append(part)
        evidence_all.extend(evidence)
    df_aug = pd.concat(parts, ignore_index=True)
    del artifacts
    return df_aug, evidence_all


def evaluate_classifier(clf, X, y_true, split_name: str) -> tuple[dict, np.ndarray]:
    y_pred = clf.predict(X)
    return (
        {
            "split": split_name,
            "n": int(len(y_true)),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        },
        y_pred,
    )


def main() -> int:
    log = get_logger("07_rac")
    paths = Paths.from_env().ensure()
    limits = RuntimeLimits.from_env()
    retriever_cfg = RetrieverConfig.from_env()
    clf_cfg = ClassifierConfig()

    out_metrics = paths.out_dir / "rac_issue_metrics.csv"
    out_report = paths.out_dir / "rac_issue_classification_report.csv"
    out_preds = paths.out_dir / "rac_issue_predictions.csv"
    out_evidence = paths.out_dir / "rac_retrieval_evidence.csv"
    ckpt_aug = paths.out_dir / "rac_augmented_df.parquet"

    if ckpt_aug.exists():
        log.info("[ckpt] loading augmented dataframe from %s", ckpt_aug)
        df_aug = pd.read_parquet(ckpt_aug)
    else:
        with stage(log, "load_and_split"):
            df = stream_csv(
                paths.data_path,
                usecols=(ID_COL, TEXT_COL, LABEL_COL),
                chunksize=limits.csv_chunksize,
                max_rows=limits.max_rows,
                sample_bytes=limits.sample_bytes_for_estimate,
            )
            df = filter_usable(df, text_col=TEXT_COL, label_col=LABEL_COL)
            df = attach_split(df, paths.split_path, id_col=ID_COL)
        with stage(log, "retrieval_pipeline"):
            df_aug, evidence_all = build_augmented_dataframe(df, retriever_cfg, log)
            del df
        log.info("[ckpt] saving augmented dataframe to %s", ckpt_aug)
        df_aug.to_parquet(ckpt_aug, index=False)
        pd.DataFrame(evidence_all).to_csv(out_evidence, index=False)
        del evidence_all

    train_aug = df_aug[df_aug["split"] == "train"]
    val_aug = df_aug[df_aug["split"] == "val"]
    test_aug = df_aug[df_aug["split"] == "test"]

    log.info("Training retrieval-augmented Issue classifier")
    log.info("Train rows: %d", len(train_aug))

    with stage(log, "vectorize_augmented"):
        vectorizer = TfidfVectorizer(
            lowercase=True,
            min_df=clf_cfg.vec_min_df,
            max_df=clf_cfg.vec_max_df,
            ngram_range=clf_cfg.vec_ngram,
            max_features=clf_cfg.vec_max_features,
            sublinear_tf=True,
        )
        X_train = vectorizer.fit_transform(train_aug["augmented_text"])
        log.info("X_train shape=%s nnz=%d", X_train.shape, X_train.nnz)
        X_val = vectorizer.transform(val_aug["augmented_text"])
        X_test = vectorizer.transform(test_aug["augmented_text"])

    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=clf_cfg.max_iter,
            class_weight=clf_cfg.class_weight,
            solver=clf_cfg.solver,
            random_state=clf_cfg.random_state,
        ),
        n_jobs=clf_cfg.n_jobs,
        verbose=clf_cfg.verbose,
    )
    log.info("[fit] OvR LogReg starting (n_jobs=%d)", clf_cfg.n_jobs)
    started = time.perf_counter()
    clf.fit(X_train, train_aug[LABEL_COL])
    log.info("[fit] done in %.1fs", time.perf_counter() - started)

    metrics: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    val_metrics, val_pred = evaluate_classifier(clf, X_val, val_aug[LABEL_COL], "val")
    metrics.append(val_metrics)
    out = val_aug[[ID_COL, "split", LABEL_COL, TEXT_COL]].copy()
    out["pred_issue"] = val_pred
    pred_frames.append(out)

    test_metrics, test_pred = evaluate_classifier(clf, X_test, test_aug[LABEL_COL], "test")
    metrics.append(test_metrics)
    out = test_aug[[ID_COL, "split", LABEL_COL, TEXT_COL]].copy()
    out["pred_issue"] = test_pred
    pred_frames.append(out)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_metrics, index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(out_preds, index=False)
    pd.DataFrame(
        classification_report(test_aug[LABEL_COL], test_pred, output_dict=True, zero_division=0)
    ).transpose().to_csv(out_report)

    log.info("Saved: %s", out_metrics)
    log.info("Saved: %s", out_report)
    log.info("Saved: %s", out_preds)
    log.info("Saved: %s", out_evidence)
    log.info("\n%s", metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
