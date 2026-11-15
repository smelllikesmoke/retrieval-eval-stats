"""Does a retrieval change really help, or is it noise?

Compares four lexical retrieval configurations on BEIR SciFact and asks the
statistical question behind every "the new setting looks better":
  * per-query metrics (nDCG@10, Recall@10, MRR@10)
  * bootstrap confidence intervals for each system's mean
  * paired randomization tests for each change, Holm-corrected
  * a simulation of how often a 20-query hand check picks the wrong winner

BM25 parameters are tuned on the train split only; every reported number is on
the held-out test split.

Usage:  python retrieval_eval.py            (expects data/scifact/ from BEIR)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "scifact"
OUT = ROOT / "results"
K = 10
B_RESAMPLES = 10_000
SEED = 7
rng = np.random.default_rng(SEED)

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS and len(t) > 1]


# ---------------------------------------------------------------- data
def load():
    corpus = [json.loads(l) for l in open(DATA / "corpus.jsonl")]
    doc_ids = [d["_id"] for d in corpus]
    texts = [f'{d.get("title", "")} {d["text"]}' for d in corpus]
    queries = {q["_id"]: q["text"] for q in map(json.loads, open(DATA / "queries.jsonl"))}

    def qrels(split):
        rel: dict[str, set[str]] = {}
        for line in list(open(DATA / "qrels" / f"{split}.tsv"))[1:]:
            qid, did, score = line.strip().split("\t")
            if int(score) > 0:
                rel.setdefault(qid, set()).add(did)
        return rel

    return doc_ids, texts, queries, qrels("train"), qrels("test")


# ---------------------------------------------------------------- retrievers
class Index:
    def __init__(self, texts: list[str]):
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.tf = self.vec.fit_transform(texts).tocsc().astype(np.float32)  # docs x terms
        self.n = self.tf.shape[0]
        self.dl = np.asarray(self.tf.sum(axis=1)).ravel()
        self.avgdl = self.dl.mean()
        df = np.diff(self.tf.indptr)
        self.idf_bm25 = np.log(1 + (self.n - df + 0.5) / (df + 0.5)).astype(np.float32)
        idf_tfidf = np.log((1 + self.n) / (1 + df)) + 1
        tfidf = self.tf.tocsr().multiply(idf_tfidf).tocsr()
        norms = np.sqrt(np.asarray(tfidf.multiply(tfidf).sum(axis=1)).ravel()) + 1e-9
        self.tfidf = sparse.diags(1 / norms) @ tfidf
        self.idf_tfidf = idf_tfidf
        self._bm25_cache: dict[tuple[float, float], sparse.csr_matrix] = {}

    def qvec(self, q: str) -> sparse.csr_matrix:
        return self.vec.transform([q]).astype(np.float32)

    def bm25_matrix(self, k1: float, b: float) -> sparse.csr_matrix:
        key = (k1, b)
        if key not in self._bm25_cache:
            tf = self.tf.tocoo()
            norm = k1 * (1 - b + b * self.dl[tf.row] / self.avgdl)
            w = tf.data * (k1 + 1) / (tf.data + norm) * self.idf_bm25[tf.col]
            self._bm25_cache[key] = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        return self._bm25_cache[key]

    def score_bm25(self, q: str, k1: float, b: float) -> np.ndarray:
        qv = self.qvec(q)
        qv.data[:] = 1.0  # binary query terms
        return np.asarray((self.bm25_matrix(k1, b) @ qv.T).todense()).ravel()

    def score_tfidf(self, q: str) -> np.ndarray:
        qv = self.qvec(q).multiply(self.idf_tfidf).tocsr()
        qv = qv / (np.sqrt(qv.multiply(qv).sum()) + 1e-9)
        return np.asarray((self.tfidf @ sparse.csr_matrix(qv).T).todense()).ravel()


def chunk(text: str, size: int = 100, stride: int = 50) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - size + stride, stride)]


# ---------------------------------------------------------------- metrics
def per_query(ranked: list[str], rel: set[str]) -> tuple[float, float, float]:
    top = ranked[:K]
    gains = [1.0 if d in rel else 0.0 for d in top]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), K)))
    recall = sum(gains) / len(rel)
    mrr = next((1 / (i + 1) for i, g in enumerate(gains) if g), 0.0)
    return dcg / idcg, recall, mrr


def evaluate(score_fn, queries, qrels, doc_ids) -> np.ndarray:
    """Return array (n_queries, 3): nDCG@10, Recall@10, MRR@10 per query."""
    ids = np.array(doc_ids)
    rows = []
    for qid in sorted(qrels):
        s = score_fn(queries[qid])
        top = np.argpartition(-s, K)[:K]
        top = top[np.argsort(-s[top])]
        rows.append(per_query(list(ids[top]), qrels[qid]))
    return np.array(rows)


# ---------------------------------------------------------------- statistics
def bootstrap_ci(x: np.ndarray, alpha=0.05) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(B_RESAMPLES, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_randomization(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided sign-flip test on per-query differences (H0: no difference)."""
    d = b - a
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(B_RESAMPLES, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (B_RESAMPLES + 1))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def small_sample_flip_rate(a: np.ndarray, b: np.ndarray, n: int, reps=5000) -> float:
    """How often does a check on n random queries crown the opposite winner?"""
    full_winner = np.sign(b.mean() - a.mean())
    idx = rng.integers(0, len(a), size=(reps, n))
    sub = np.sign(b[idx].mean(axis=1) - a[idx].mean(axis=1))
    return float(np.mean(sub == -full_winner))  # strictly the opposite winner; ties not counted


# ---------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(exist_ok=True)
    doc_ids, texts, queries, train, test = load()
    print(f"corpus {len(doc_ids)} docs | train {len(train)} queries | test {len(test)} queries")

    idx = Index(texts)

    # Tune BM25 on train only (grid search on nDCG@10).
    grid = [(k1, b) for k1 in (0.6, 0.9, 1.2, 1.5, 2.0) for b in (0.25, 0.4, 0.55, 0.75, 0.9)]
    tune = {g: evaluate(lambda q, g=g: idx.score_bm25(q, *g), queries, train, doc_ids)[:, 0].mean()
            for g in grid}
    best = max(tune, key=tune.get)
    print(f"BM25 tuned on train: k1={best[0]}, b={best[1]} (train nDCG@10 {tune[best]:.4f})")

    # Chunked index: passages of 100 words, stride 50; a document scores as its best passage.
    chunks, owner = [], []
    for i, t in enumerate(texts):
        for c in chunk(t):
            chunks.append(c)
            owner.append(i)
    owner = np.array(owner)
    cidx = Index(chunks)
    print(f"chunked index: {len(chunks)} passages from {len(texts)} docs")

    def chunked_score(q):
        s = cidx.score_bm25(q, *best)
        out = np.full(len(texts), -np.inf)
        np.maximum.at(out, owner, s)
        return out

    systems = {
        "A  TF-IDF cosine (baseline)": idx.score_tfidf,
        "B  BM25 default (k1=0.9, b=0.4)": lambda q: idx.score_bm25(q, 0.9, 0.4),
        f"C  BM25 tuned (k1={best[0]}, b={best[1]})": lambda q: idx.score_bm25(q, *best),
        "D  BM25 tuned + 100-word chunks": chunked_score,
    }
    res = {name: evaluate(fn, queries, test, doc_ids) for name, fn in systems.items()}
    names = list(res)

    metrics = ["nDCG@10", "Recall@10", "MRR@10"]
    summary = {"n_test_queries": len(test), "bm25_tuned": {"k1": best[0], "b": best[1]}, "systems": {}, "comparisons": []}
    lines = ["| System | " + " | ".join(f"{m} (95% CI)" for m in metrics) + " |",
             "|---|" + "---|" * len(metrics)]
    for n in names:
        cells, s = [], {}
        for j, m in enumerate(metrics):
            lo, hi = bootstrap_ci(res[n][:, j])
            mean = res[n][:, j].mean()
            cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
            s[m] = {"mean": mean, "ci95": [lo, hi]}
        summary["systems"][n] = s
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    # Each change vs the step before it, on nDCG@10.
    pairs = [(names[0], names[1]), (names[1], names[2]), (names[2], names[3]), (names[0], names[3])]
    raw = [paired_randomization(res[a][:, 0], res[b][:, 0]) for a, b in pairs]
    adj = holm(raw)
    clines = ["| Change (nDCG@10) | Mean diff [95% CI] | Queries better / worse / tied | p (Holm) | Wrong-winner rate, 20-query check |",
              "|---|---|---|---|---|"]
    for (a, b), p in zip(pairs, adj):
        d = res[b][:, 0] - res[a][:, 0]
        lo, hi = bootstrap_ci(d)
        flip = small_sample_flip_rate(res[a][:, 0], res[b][:, 0], 20)
        better, worse = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        tied = len(d) - better - worse
        clines.append(f"| {a[:1]} → {b[:1]} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {better} / {worse} / {tied} | {p:.4f} | {flip:.0%} |")
        summary["comparisons"].append({"from": a, "to": b, "mean_diff": float(d.mean()), "ci95": [lo, hi],
                                       "better": better, "worse": worse, "tied": tied,
                                       "p_holm": p, "flip_rate_20q": flip})
    ctable = "\n".join(clines)

    print("\n" + table + "\n\n" + ctable)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "tables.md").write_text(table + "\n\n" + ctable + "\n")
    np.savez(OUT / "per_query.npz", **{n[:1]: res[n] for n in names})
    plot(res, names, pairs)


def plot(res, names, pairs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for i, n in enumerate(names):
        x = res[n][:, 0]
        lo, hi = bootstrap_ci(x)
        ax.errorbar(x.mean(), i, xerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o", capsize=4, color="#1E2A44")
    ax.set_yticks(range(len(names)), [n.split("  ")[1] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("nDCG@10 on 300 test queries (mean, 95% bootstrap CI)")
    ax.set_title("System quality with uncertainty")

    ax = axes[1]
    for a, b in pairs[:3]:
        d = res[b][:, 0] - res[a][:, 0]
        boot = d[rng.integers(0, len(d), size=(20000, 20))].mean(axis=1)
        ax.hist(boot, bins=60, histtype="step", density=True, label=f"{a[:1]} → {b[:1]}")
    ax.axvline(0, color="grey", lw=1)
    ax.set_ylim(0, 16)  # B -> C is a tall spike at 0 (279/300 queries tied); clipped so the others stay visible
    ax.set_xlabel("Mean nDCG@10 difference seen in a 20-query hand check")
    ax.set_title("What a small hand check would show")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=150)


if __name__ == "__main__":
    main()
