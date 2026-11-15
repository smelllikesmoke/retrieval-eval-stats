"""Retrievers: TF-IDF, BM25 (sparse, implemented from scratch), dense embeddings, hybrid fusion.

All scorers take a list of query strings and return a dense (n_queries, n_docs) score matrix.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

TOKEN = re.compile(r"[a-z0-9]+")
CACHE = Path(__file__).resolve().parent.parent / "cache"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS and len(t) > 1]


class SparseIndex:
    def __init__(self, texts: list[str]):
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, dtype=np.float32)
        self.tf = self.vec.fit_transform(texts).tocoo()
        self.n_docs, self.n_terms = self.tf.shape
        self.dl = np.asarray(self.tf.tocsr().sum(axis=1)).ravel()
        self.avgdl = float(self.dl.mean())
        self.df = np.bincount(self.tf.col, minlength=self.n_terms)
        self.idf_bm25 = np.log(1 + (self.n_docs - self.df + 0.5) / (self.df + 0.5)).astype(np.float32)
        self.idf_tfidf = (np.log((1 + self.n_docs) / (1 + self.df)) + 1).astype(np.float32)
        self._tfidf = None

    def query_matrix(self, queries: list[str], binary: bool = True) -> sparse.csr_matrix:
        q = self.vec.transform(queries).astype(np.float32)
        if binary:
            q.data[:] = 1.0
        return q

    def bm25_weights(self, k1: float, b: float) -> sparse.csr_matrix:
        tf = self.tf
        norm = k1 * (1 - b + b * self.dl[tf.row] / self.avgdl)
        w = tf.data * (k1 + 1) / (tf.data + norm) * self.idf_bm25[tf.col]
        return sparse.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)

    def bm25(self, queries: list[str], k1: float = 0.9, b: float = 0.4) -> np.ndarray:
        return (self.query_matrix(queries) @ self.bm25_weights(k1, b).T).toarray()

    def tfidf(self, queries: list[str]) -> np.ndarray:
        if self._tfidf is None:
            m = sparse.csr_matrix((self.tf.data, (self.tf.row, self.tf.col)), shape=self.tf.shape)
            m = m.multiply(self.idf_tfidf).tocsr()
            norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()) + 1e-9
            self._tfidf = sparse.diags(1 / norms) @ m
        q = self.query_matrix(queries, binary=False).multiply(self.idf_tfidf).tocsr()
        qn = np.sqrt(np.asarray(q.multiply(q).sum(axis=1)).ravel()) + 1e-9
        q = sparse.diags(1 / qn) @ q
        return (q @ self._tfidf.T).toarray()


def chunk(text: str, size: int = 100, stride: int = 50) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - size + stride, stride)]


class ChunkedBM25:
    """BM25 over overlapping passages; a document scores as its best passage (max-pooling)."""

    def __init__(self, texts: list[str], size: int = 100, stride: int = 50):
        passages, owner = [], []
        for i, t in enumerate(texts):
            for c in chunk(t, size, stride):
                passages.append(c)
                owner.append(i)
        owner = np.array(owner)
        # Passages of one document are contiguous, so max-pooling is a reduceat over these offsets.
        self.starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        self.n_docs = len(texts)
        self.n_passages = len(passages)
        self.index = SparseIndex(passages)

    def score(self, queries: list[str], k1: float = 0.9, b: float = 0.4, batch: int = 256) -> np.ndarray:
        wt = self.index.bm25_weights(k1, b).T.tocsc()
        out = np.empty((len(queries), self.n_docs), dtype=np.float32)
        for s in range(0, len(queries), batch):
            ps = (self.index.query_matrix(queries[s:s + batch]) @ wt).toarray()
            out[s:s + batch] = np.maximum.reduceat(ps, self.starts, axis=1)
        return out


class Dense:
    """Sentence-embedding retriever (cosine similarity), with on-disk caching of document vectors."""

    QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="mps")
        self.model.max_seq_length = 256

    def encode_docs(self, dataset: str, texts: list[str]) -> np.ndarray:
        path = CACHE / f"{dataset}__{self.model_name.replace('/', '_')}.npy"
        if path.exists():
            return np.load(path)
        emb = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        CACHE.mkdir(exist_ok=True)
        np.save(path, emb)
        return emb

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        prefix = self.QUERY_PREFIX.get(self.model_name, "")
        return self.model.encode([prefix + q for q in queries], batch_size=64, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)


def rrf(*score_mats: np.ndarray, k: int = 60, depth: int = 100) -> np.ndarray:
    """Reciprocal rank fusion over the top `depth` results of each system."""
    n_q, n_d = score_mats[0].shape
    fused = np.zeros((n_q, n_d), dtype=np.float32)
    rows = np.arange(n_q)[:, None]
    depth = min(depth, n_d)
    for s in score_mats:
        part = np.argpartition(-s, depth - 1, axis=1)[:, :depth]
        order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1)
        ranked = np.take_along_axis(part, order, axis=1)
        fused[rows, ranked] += 1.0 / (k + 1 + np.arange(depth))[None, :]
    return fused
