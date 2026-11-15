"""Stage 1: run every retriever on every dataset and store per-query results.

Outputs (results/):
  per_query.csv   one row per (dataset, query, system): nDCG@10, Recall@10, RR@10 + query features
  bm25_grid.npz   per-query nDCG@10 for a 5x5 grid of BM25 (k1, b), used for nested cross-validation
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATASETS, Dataset, load
from .metrics import ndcg_at_k, recall_at_k, rr_at_k, top_k
from .retrievers import ChunkedBM25, Dense, SparseIndex, rrf

OUT = Path(__file__).resolve().parent.parent / "results"
K1_GRID = (0.6, 0.9, 1.2, 1.5, 2.0)
B_GRID = (0.25, 0.4, 0.55, 0.75, 0.9)
DEFAULT = dict(k1=0.9, b=0.4)


def mask_self(ds: Dataset, scores: np.ndarray, qids: list[str]) -> np.ndarray:
    if ds.exclude_self:
        pos = {d: i for i, d in enumerate(ds.doc_ids)}
        for r, q in enumerate(qids):
            if q in pos:
                scores[r, pos[q]] = -np.inf
    return scores


def evaluate(ds: Dataset, scores: np.ndarray, qids: list[str]) -> np.ndarray:
    scores = mask_self(ds, scores, qids)
    top = top_k(scores)
    ids = np.array(ds.doc_ids)
    rows = []
    for r, q in enumerate(qids):
        ranked = list(ids[top[r]])
        rel = ds.qrels[q]
        rows.append((ndcg_at_k(ranked, rel), recall_at_k(ranked, rel), rr_at_k(ranked, rel)))
    return np.array(rows)


def query_features(ds: Dataset, idx: SparseIndex, qids: list[str]) -> pd.DataFrame:
    """Properties of each query that might explain when retrieval fails."""
    analyzer = idx.vec.build_analyzer()
    vocab = idx.vec.vocabulary_
    tf = idx.tf.tocsr()
    pos = {d: i for i, d in enumerate(ds.doc_ids)}
    rows = []
    for q in qids:
        toks = analyzer(ds.queries[q])
        uniq = sorted(set(toks))
        ids = [vocab[t] for t in uniq if t in vocab]
        idf = idx.idf_bm25[ids] if ids else np.array([0.0])
        overlaps = []
        for d in ds.qrels[q]:
            if d in pos and uniq:
                doc_terms = set(tf.indices[tf.indptr[pos[d]]:tf.indptr[pos[d] + 1]])
                overlaps.append(sum(1 for i in ids if i in doc_terms) / len(uniq))
        rows.append(dict(
            q_len=len(toks),
            mean_idf=float(idf.mean()),
            max_idf=float(idf.max()),
            oov_frac=1 - len(ids) / max(len(uniq), 1),
            overlap=max(overlaps) if overlaps else 0.0,
            n_rel=len(ds.qrels[q]),
        ))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    dense = Dense()
    frames, grid = [], {}
    for name in DATASETS:
        t0 = time.time()
        ds = load(name)
        qids = ds.qids
        qtexts = [ds.queries[q] for q in qids]
        idx = SparseIndex(ds.texts)
        print(f"[{name}] {len(ds.doc_ids)} docs, {len(qids)} test queries, vocab {idx.n_terms}")

        s_bm25 = idx.bm25(qtexts, **DEFAULT)
        doc_emb = dense.encode_docs(name, ds.texts)
        s_dense = dense.encode_queries(qtexts) @ doc_emb.T
        systems = {
            "TF-IDF": idx.tfidf(qtexts),
            "BM25": s_bm25,
            "BM25+chunks": ChunkedBM25(ds.texts).score(qtexts, **DEFAULT),
            "Dense": s_dense,
            "Hybrid": rrf(mask_self(ds, s_bm25.copy(), qids), mask_self(ds, s_dense.copy(), qids)),
        }
        feats = query_features(ds, idx, qids)
        for sys_name, scores in systems.items():
            m = evaluate(ds, scores, qids)
            f = feats.copy()
            f.insert(0, "dataset", name)
            f.insert(1, "qid", qids)
            f.insert(2, "system", sys_name)
            f["ndcg"], f["recall"], f["rr"] = m[:, 0], m[:, 1], m[:, 2]
            frames.append(f)
            print(f"   {sys_name:12s} nDCG@10 {m[:, 0].mean():.3f}")
        del systems

        g = np.zeros((len(K1_GRID), len(B_GRID), len(qids)), dtype=np.float32)
        for i, k1 in enumerate(K1_GRID):
            for j, b in enumerate(B_GRID):
                g[i, j] = evaluate(ds, idx.bm25(qtexts, k1, b), qids)[:, 0]
        grid[name] = g
        print(f"   BM25 grid done ({time.time() - t0:.0f}s total)")

    pd.concat(frames).to_csv(OUT / "per_query.csv", index=False)
    np.savez(OUT / "bm25_grid.npz", k1=np.array(K1_GRID), b=np.array(B_GRID), **grid)
    print("wrote results/per_query.csv and results/bm25_grid.npz")


if __name__ == "__main__":
    main()
