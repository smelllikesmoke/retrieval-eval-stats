import numpy as np
import pytest

from reteval.metrics import ndcg_at_k, recall_at_k, rr_at_k, top_k
from reteval.retrievers import ChunkedBM25, SparseIndex, chunk, rrf
from reteval.stats import bootstrap_ci, hierarchical_normal, holm, n_for_power, paired_randomization


# ---------------------------------------------------------------- metrics
def test_ndcg_perfect_and_empty():
    rel = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], rel) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], rel) == 0.0


def test_ndcg_graded_order_matters():
    rel = {"a": 2, "b": 1}
    assert ndcg_at_k(["b", "a"], rel) < ndcg_at_k(["a", "b"], rel)


def test_ndcg_hand_computed():
    # relevant doc at rank 2 only: DCG = 1/log2(3), IDCG = 1
    assert ndcg_at_k(["x", "a"], {"a": 1}) == pytest.approx(1 / np.log2(3))


def test_recall_and_rr():
    rel = {"a": 1, "b": 1}
    assert recall_at_k(["x", "a", "y"], rel) == 0.5
    assert rr_at_k(["x", "y", "a"], rel) == pytest.approx(1 / 3)
    assert rr_at_k(["x"], rel) == 0.0


def test_top_k_order():
    s = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert top_k(s, 3).tolist() == [[1, 3, 2]]


# ---------------------------------------------------------------- retrievers
def test_bm25_prefers_matching_doc():
    idx = SparseIndex(["cats purr loudly", "dogs bark", "stock markets fell"])
    s = idx.bm25(["dogs barking dogs"])
    assert s.argmax() == 1


def test_bm25_idf_rare_term_wins():
    idx = SparseIndex(["common word here", "common word there", "rare zebra common"])
    s = idx.bm25(["zebra common"])
    assert s.argmax() == 2


def test_chunking_and_maxpool():
    text = " ".join(f"w{i}" for i in range(250))
    assert len(chunk(text, 100, 50)) == 4
    cb = ChunkedBM25(["alpha beta " * 10, " ".join(["filler"] * 300) + " zebra"])
    assert cb.score(["zebra"]).argmax() == 1


def test_rrf_rewards_agreement():
    a = np.array([[3.0, 2.0, 1.0]])
    b = np.array([[1.0, 3.0, 2.0]])
    fused = rrf(a, b, depth=3)
    assert fused.argmax() == 1  # ranked 2nd and 1st beats 1st and 3rd


# ---------------------------------------------------------------- statistics
def test_holm_known_values():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_randomization_null_and_effect():
    rng = np.random.default_rng(0)
    null = rng.normal(0, 1, 200)
    shifted = rng.normal(0.5, 1, 200)
    assert paired_randomization(shifted, rng, b=2000) < 0.01
    assert paired_randomization(null - null.mean(), rng, b=2000) > 0.5


def test_bootstrap_covers_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(3, 1, 500)
    lo, hi = bootstrap_ci(x, rng, b=2000)
    assert lo < 3 < hi


def test_power_formula():
    # (1.96 + 0.8416)^2 * (1/0.5)^2 ~= 31.4 -> 32
    assert n_for_power(1.0, 0.5) == 32


def test_hierarchical_shrinks_towards_pooled_mean():
    y = np.array([0.10, 0.00, 0.05, 0.02])
    se = np.array([0.02, 0.02, 0.02, 0.02])
    r = hierarchical_normal(y, se, seed=0, draws=5000)
    assert min(y) < r["mu_mean"] < max(y)
    assert r["theta_mean"][0] < y[0] and r["theta_mean"][1] > y[1]
