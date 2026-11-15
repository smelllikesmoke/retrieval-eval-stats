"""Inference tools: bootstrap, paired randomization test, Holm correction, power,
and a Bayesian hierarchical model for effects across datasets (exact, grid over tau)."""

from __future__ import annotations

import numpy as np
from scipy import stats

B = 10_000


def bootstrap_ci(x: np.ndarray, rng: np.random.Generator, alpha: float = 0.05, b: int = B) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    idx = rng.integers(0, len(x), size=(b, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_randomization(d: np.ndarray, rng: np.random.Generator, b: int = B) -> float:
    """Two-sided sign-flip test of H0: per-query differences are symmetric around 0."""
    obs = abs(d.mean())
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(b, len(d)))
    null = np.abs(signs @ d.astype(np.float32) / len(d))
    return float((np.sum(null >= obs - 1e-12) + 1) / (b + 1))


def holm(p: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def n_for_power(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Queries needed for a two-sided paired test to detect a mean difference `delta`
    (normal approximation)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil((z * sd / delta) ** 2))


def simulated_power(d: np.ndarray, n: int, delta: float, rng: np.random.Generator,
                    reps: int = 2000, alpha: float = 0.05) -> float:
    """Empirical power: resample n per-query differences, shifted so their true mean is `delta`,
    and count how often a paired t-test rejects H0."""
    centred = d - d.mean() + delta
    idx = rng.integers(0, len(d), size=(reps, n))
    s = centred[idx]
    t = s.mean(axis=1) / (s.std(axis=1, ddof=1) / np.sqrt(n) + 1e-12)
    crit = stats.t.ppf(1 - alpha / 2, n - 1)
    return float(np.mean(np.abs(t) > crit))


def opposite_winner_rate(d: np.ndarray, n: int, rng: np.random.Generator, reps: int = 5000) -> float:
    """How often a check on n random queries gives the mean difference the opposite sign."""
    full = np.sign(d.mean())
    idx = rng.integers(0, len(d), size=(reps, n))
    return float(np.mean(np.sign(d[idx].mean(axis=1)) == -full))


def hierarchical_normal(y: np.ndarray, se: np.ndarray, tau_grid: np.ndarray | None = None,
                        tau_scale: float = 0.05, draws: int = 20_000, seed: int = 0) -> dict:
    """Bayesian random-effects model (the 'eight schools' model):

        y_j | theta_j ~ N(theta_j, se_j^2)     observed effect in dataset j
        theta_j | mu, tau ~ N(mu, tau^2)       true effect in dataset j
        mu ~ flat,  tau ~ half-Cauchy(0, tau_scale)

    The marginal posterior of tau is computed exactly on a grid; mu and theta_j are then
    drawn from their conditional normal posteriors (Gelman et al., BDA3, ch. 5).
    """
    rng = np.random.default_rng(seed)
    if tau_grid is None:
        tau_grid = np.linspace(1e-4, 0.3, 3000)
    v = se ** 2
    log_post = []
    for tau in tau_grid:
        w = 1 / (v + tau ** 2)
        mu_hat = np.sum(w * y) / np.sum(w)
        lp = (0.5 * np.log(1 / np.sum(w)) + 0.5 * np.sum(np.log(w))
              - 0.5 * np.sum(w * (y - mu_hat) ** 2)
              + np.log(2 / (np.pi * tau_scale * (1 + (tau / tau_scale) ** 2))))
        log_post.append(lp)
    log_post = np.array(log_post)
    p_tau = np.exp(log_post - log_post.max())
    p_tau /= p_tau.sum()
    taus = rng.choice(tau_grid, size=draws, p=p_tau)
    w = 1 / (v[None, :] + taus[:, None] ** 2)
    mu_hat = (w * y).sum(axis=1) / w.sum(axis=1)
    mu = rng.normal(mu_hat, np.sqrt(1 / w.sum(axis=1)))
    prec = 1 / v[None, :] + 1 / taus[:, None] ** 2
    theta_mean = (y / v + mu[:, None] / taus[:, None] ** 2) / prec
    theta = rng.normal(theta_mean, np.sqrt(1 / prec))
    new = rng.normal(mu, taus)  # effect on a new, unseen dataset
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
    return {
        "mu_mean": float(mu.mean()), "mu_ci95": q(mu), "p_mu_gt_0": float(np.mean(mu > 0)),
        "tau_median": float(np.median(taus)), "tau_ci95": q(taus),
        "theta_mean": theta.mean(axis=0).tolist(), "theta_ci95": [q(theta[:, j]) for j in range(len(y))],
        "new_dataset_mean": float(new.mean()), "new_dataset_ci95": q(new),
        "p_new_gt_0": float(np.mean(new > 0)),
        "_mu_draws": mu, "_new_draws": new,
    }
