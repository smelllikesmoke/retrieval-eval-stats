"""Stage 2: statistical analysis of the per-query results.

  1. Per-dataset system quality with bootstrap CIs
  2. Paired randomization tests vs BM25, Holm-corrected over all tests
  3. Pooling across datasets: frequentist mixed model and Bayesian hierarchical model
  4. Statistical modelling of *when* retrieval fails (GLM, OLS with model selection, Lasso)
  5. Power analysis: how many test queries a comparison needs
  6. Nested cross-validation: how much BM25 tuning gains are inflated in-sample

Writes results/summary.json, results/tables.md and figures to results/fig_*.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .stats import (bootstrap_ci, hierarchical_normal, holm, n_for_power, opposite_winner_rate,
                    paired_randomization, simulated_power)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SEED = 7
SYSTEMS = ["TF-IDF", "BM25", "BM25+chunks", "Dense", "Hybrid"]
DATASETS = ["scifact", "nfcorpus", "fiqa", "arguana"]
LABEL = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA", "arguana": "ArguAna"}
FEATURES = ["q_len", "mean_idf", "max_idf", "oov_frac", "overlap", "n_rel"]
NAVY, ORANGE, GREY = "#1E2A44", "#D9822B", "#9CA3AF"
COLORS = {"TF-IDF": "#9CA3AF", "BM25": "#1E2A44", "BM25+chunks": "#6B7FA8", "Dense": "#D9822B", "Hybrid": "#2E8B57"}


def wide(df: pd.DataFrame, metric: str = "ndcg") -> pd.DataFrame:
    return df.pivot_table(index=["dataset", "qid"], columns="system", values=metric).reset_index()


# --------------------------------------------------------------------------- 1 + 2
def quality_and_tests(df, rng, S):
    rows, lines = [], []
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        for sysn in SYSTEMS:
            x = sub[sub.system == sysn]
            r = {"dataset": ds, "system": sysn, "n": len(x)}
            for m in ("ndcg", "recall", "rr"):
                lo, hi = bootstrap_ci(x[m].to_numpy(), rng)
                r[m] = x[m].mean()
                r[m + "_ci"] = [lo, hi]
            rows.append(r)
    S["quality"] = rows

    w = wide(df)
    tests = []
    for ds in DATASETS:
        sub = w[w.dataset == ds]
        for sysn in [s for s in SYSTEMS if s != "BM25"]:
            d = (sub[sysn] - sub["BM25"]).to_numpy()
            lo, hi = bootstrap_ci(d, rng)
            tests.append(dict(dataset=ds, system=sysn, n=len(d), diff=float(d.mean()), ci=[lo, hi],
                              better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum()),
                              p_raw=paired_randomization(d, rng),
                              flip20=opposite_winner_rate(d, 20, rng),
                              sd=float(d.std(ddof=1))))
    for t, p in zip(tests, holm([t["p_raw"] for t in tests])):
        t["p_holm"] = float(p)
    S["tests_vs_bm25"] = tests

    # tables
    lines.append("### nDCG@10 by dataset (mean [95% bootstrap CI])\n")
    lines.append("| System | " + " | ".join(f"{LABEL[d]} (n={len(w[w.dataset == d])})" for d in DATASETS) + " |")
    lines.append("|---|" + "---|" * len(DATASETS))
    for sysn in SYSTEMS:
        cells = []
        for ds in DATASETS:
            r = next(r for r in rows if r["dataset"] == ds and r["system"] == sysn)
            cells.append(f'{r["ndcg"]:.3f} [{r["ndcg_ci"][0]:.3f}, {r["ndcg_ci"][1]:.3f}]')
        lines.append(f"| {sysn} | " + " | ".join(cells) + " |")
    lines.append("\n### Each system vs BM25 (paired, nDCG@10; Holm over all 16 tests)\n")
    lines.append("| Dataset | System | Diff [95% CI] | Better / worse | p (Holm) | 20-query check picks opposite winner |")
    lines.append("|---|---|---|---|---|---|")
    for t in tests:
        lines.append(f'| {LABEL[t["dataset"]]} | {t["system"]} | {t["diff"]:+.3f} [{t["ci"][0]:+.3f}, {t["ci"][1]:+.3f}] | '
                     f'{t["better"]} / {t["worse"]} | {t["p_holm"]:.4f} | {t["flip20"]:.0%} |')
    lines.append("\n### Does fusing BM25 into dense retrieval help? (Hybrid − Dense, paired, Holm over 4 tests)\n")
    lines.append("| Dataset | Diff [95% CI] | Better / worse | p (Holm) |\n|---|---|---|---|")
    hd = []
    for ds in DATASETS:
        sub = w[w.dataset == ds]
        d = (sub["Hybrid"] - sub["Dense"]).to_numpy()
        lo, hi = bootstrap_ci(d, rng)
        hd.append(dict(dataset=ds, diff=float(d.mean()), ci=[lo, hi], better=int((d > 1e-9).sum()),
                       worse=int((d < -1e-9).sum()), p_raw=paired_randomization(d, rng)))
    for t, p in zip(hd, holm([t["p_raw"] for t in hd])):
        t["p_holm"] = float(p)
        lines.append(f'| {LABEL[t["dataset"]]} | {t["diff"]:+.3f} [{t["ci"][0]:+.3f}, {t["ci"][1]:+.3f}] | '
                     f'{t["better"]} / {t["worse"]} | {t["p_holm"]:.4f} |')
    S["hybrid_vs_dense"] = hd
    return lines


def fig_quality(S):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for j, sysn in enumerate(SYSTEMS):
        for i, ds in enumerate(DATASETS):
            r = next(r for r in S["quality"] if r["dataset"] == ds and r["system"] == sysn)
            y = i + (j - 2) * 0.15
            ax.errorbar(r["ndcg"], y, xerr=[[r["ndcg"] - r["ndcg_ci"][0]], [r["ndcg_ci"][1] - r["ndcg"]]],
                        fmt="o", ms=4, capsize=2, color=COLORS[sysn], label=sysn if i == 0 else None)
    ax.set_yticks(range(len(DATASETS)), [LABEL[d] for d in DATASETS])
    ax.invert_yaxis()
    ax.set_xlabel("nDCG@10 (mean and 95% bootstrap CI over test queries)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "fig_quality.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- 3
def pooling(df, S):
    w = wide(df)
    out, lines = {}, ["\n### Pooling across datasets (effect vs BM25 on nDCG@10)\n",
                      "| Comparison | Mixed model: mean [95% CI] | Bayesian: mean [95% CrI] | P(effect > 0) | Between-dataset SD (tau) | Predicted effect on a new dataset [95%] | P(new > 0) |",
                      "|---|---|---|---|---|---|---|"]
    for sysn in ("Dense", "Hybrid", "BM25+chunks"):
        w["d"] = w[sysn] - w["BM25"]
        mm = smf.mixedlm("d ~ 1", w, groups=w["dataset"]).fit(reml=True)
        ci = mm.conf_int().loc["Intercept"].tolist()
        g = w.groupby("dataset")["d"]
        y = g.mean().reindex(DATASETS).to_numpy()
        se = (g.std(ddof=1) / np.sqrt(g.size())).reindex(DATASETS).to_numpy()
        bayes = hierarchical_normal(y, se, seed=SEED)
        out[sysn] = {"mixed_mean": float(mm.params["Intercept"]), "mixed_ci": ci,
                     "mixed_group_var": float(mm.cov_re.iloc[0, 0]),
                     "y": y.tolist(), "se": se.tolist(),
                     **{k: v for k, v in bayes.items() if not k.startswith("_")}}
        out[sysn]["_draws"] = bayes
        b = bayes
        lines.append(f'| {sysn} − BM25 | {mm.params["Intercept"]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | '
                     f'{b["mu_mean"]:+.3f} [{b["mu_ci95"][0]:+.3f}, {b["mu_ci95"][1]:+.3f}] | {b["p_mu_gt_0"]:.2f} | '
                     f'{b["tau_median"]:.3f} [{b["tau_ci95"][0]:.3f}, {b["tau_ci95"][1]:.3f}] | '
                     f'{b["new_dataset_mean"]:+.3f} [{b["new_dataset_ci95"][0]:+.3f}, {b["new_dataset_ci95"][1]:+.3f}] | {b["p_new_gt_0"]:.2f} |')
    # figure: Hybrid and Dense, per-dataset raw vs shrunk, plus pooled
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, sysn in zip(axes, ("Dense", "Hybrid")):
        o = out[sysn]
        for i, ds in enumerate(DATASETS):
            ax.errorbar(o["y"][i], i - 0.12, xerr=1.96 * o["se"][i], fmt="o", color=GREY, capsize=2,
                        label="observed (±1.96 SE)" if i == 0 else None)
            m, (lo, hi) = o["theta_mean"][i], o["theta_ci95"][i]
            ax.errorbar(m, i + 0.12, xerr=[[m - lo], [hi - m]], fmt="s", color=NAVY, capsize=2,
                        label="hierarchical posterior" if i == 0 else None)
        mu, (lo, hi) = o["mu_mean"], o["mu_ci95"]
        ax.errorbar(mu, len(DATASETS) + 0.3, xerr=[[mu - lo], [hi - mu]], fmt="D", color=ORANGE, capsize=3, label="pooled mean μ")
        n, (lo, hi) = o["new_dataset_mean"], o["new_dataset_ci95"]
        ax.errorbar(n, len(DATASETS) + 1.0, xerr=[[n - lo], [hi - n]], fmt="D", mfc="white", color=ORANGE, capsize=3, label="new dataset")
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(list(range(len(DATASETS))) + [len(DATASETS) + 0.3, len(DATASETS) + 1.0],
                      [LABEL[d] for d in DATASETS] + ["Pooled μ", "New dataset"])
        ax.invert_yaxis()
        ax.set_title(f"{sysn} − BM25 (nDCG@10)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(RES / "fig_bayes.png", dpi=160)
    plt.close(fig)
    S["pooling"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in out.items()}
    return lines


# --------------------------------------------------------------------------- 4
def modelling(df, S):
    lines = ["\n### When does retrieval fail? Statistical models of per-query outcomes\n"]
    w = wide(df).merge(df[df.system == "BM25"][["dataset", "qid"] + FEATURES], on=["dataset", "qid"])
    w["log_n_rel"] = np.log(w["n_rel"])
    w["log_q_len"] = np.log1p(w["q_len"])
    feats = ["log_q_len", "mean_idf", "max_idf", "oov_frac", "overlap", "log_n_rel"]
    z = w.copy()
    z[feats] = StandardScaler().fit_transform(w[feats])

    # (a) Logistic GLM: probability that BM25 finds no relevant document in the top 10.
    rec = wide(df, "recall")
    z = z.merge(rec[["dataset", "qid", "BM25"]].rename(columns={"BM25": "bm25_recall"}), on=["dataset", "qid"])
    z["bm25_fail"] = (z["bm25_recall"] == 0).astype(int)
    f_glm = "bm25_fail ~ " + " + ".join(feats) + " + C(dataset)"
    glm = smf.glm(f_glm, z, family=sm.families.Binomial()).fit()
    ors = np.exp(glm.params[feats])
    ci = np.exp(glm.conf_int().loc[feats])
    S["glm_bm25_fail"] = {"n": int(glm.nobs), "fail_rate": float(z["bm25_fail"].mean()),
                          "aic": float(glm.aic), "pseudo_r2": float(1 - glm.deviance / glm.null_deviance),
                          "odds_ratios": {f: [float(ors[f]), float(ci.loc[f, 0]), float(ci.loc[f, 1]), float(glm.pvalues[f])] for f in feats}}
    lines.append(f"**(a) Logistic GLM: P(BM25 finds nothing relevant in top 10)**, n = {int(glm.nobs)}, "
                 f"failure rate {z['bm25_fail'].mean():.1%}, deviance explained {S['glm_bm25_fail']['pseudo_r2']:.2f}. "
                 "Odds ratios per +1 SD, dataset fixed effects:\n")
    lines.append("| Feature | Odds ratio [95% CI] | p |\n|---|---|---|")
    for f in feats:
        o = S["glm_bm25_fail"]["odds_ratios"][f]
        lines.append(f"| {f} | {o[0]:.2f} [{o[1]:.2f}, {o[2]:.2f}] | {o[3]:.2g} |")

    # (b) Linear models for the dense-vs-BM25 difference, compared by AIC/BIC.
    z["d"] = z["Dense"] - z["BM25"]
    candidates = {
        "M0  intercept": "d ~ 1",
        "M1  dataset": "d ~ C(dataset)",
        "M2  dataset + overlap": "d ~ C(dataset) + overlap",
        "M3  dataset + all features": "d ~ C(dataset) + " + " + ".join(feats),
        "M4  M3 + overlap x dataset": "d ~ C(dataset) * overlap + " + " + ".join(f for f in feats if f != "overlap"),
    }
    comp = []
    for name, f in candidates.items():
        m = smf.ols(f, z).fit()
        comp.append((name, int(m.df_model), m.aic, m.bic, m.rsquared_adj))
    best_bic = min(comp, key=lambda r: r[3])[0]
    S["ols_model_selection"] = [dict(model=c[0], df=c[1], aic=c[2], bic=c[3], adj_r2=c[4]) for c in comp]
    lines.append("\n**(b) Linear models for Dense − BM25 (nDCG@10), selected by AIC/BIC**\n")
    lines.append("| Model | params | AIC | BIC | adj. R² |\n|---|---|---|---|---|")
    for c in comp:
        mark = " ← lowest BIC" if c[0] == best_bic else ""
        lines.append(f"| {c[0]}{mark} | {c[1]} | {c[2]:.0f} | {c[3]:.0f} | {c[4]:.3f} |")
    m3 = smf.ols(candidates["M3  dataset + all features"], z).fit(cov_type="HC3")
    S["ols_m3_hc3"] = {f: [float(m3.params[f]), *map(float, m3.conf_int().loc[f]), float(m3.pvalues[f])] for f in feats}
    lines.append("\nM3 coefficients (per +1 SD, heteroskedasticity-robust HC3 CIs):\n")
    lines.append("| Feature | Effect on Dense − BM25 [95% CI] | p |\n|---|---|---|")
    for f in feats:
        c = S["ols_m3_hc3"][f]
        lines.append(f"| {f} | {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}] | {c[3]:.2g} |")

    # (c) Lasso with all pairwise interactions; regularisation chosen by 10-fold CV.
    from sklearn.preprocessing import PolynomialFeatures
    X = pd.concat([z[feats], pd.get_dummies(z["dataset"], drop_first=True, dtype=float)], axis=1)
    poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
    Xp = StandardScaler().fit_transform(poly.fit_transform(X))
    names = poly.get_feature_names_out(X.columns)
    lasso = LassoCV(cv=KFold(10, shuffle=True, random_state=SEED), random_state=SEED, alphas=60).fit(Xp, z["d"])
    kept = sorted(((n, c) for n, c in zip(names, lasso.coef_) if abs(c) > 1e-6), key=lambda t: -abs(t[1]))
    cv = KFold(10, shuffle=True, random_state=SEED)
    r2_lasso = cross_val_score(LassoCV(cv=5, alphas=40, random_state=SEED), Xp, z["d"], cv=cv, scoring="r2").mean()
    r2_ols = cross_val_score(LinearRegression(), Xp, z["d"], cv=cv, scoring="r2").mean()
    S["lasso"] = {"alpha": float(lasso.alpha_), "n_candidates": len(names), "n_kept": len(kept),
                  "kept": [[n, float(c)] for n, c in kept], "cv_r2_lasso": float(r2_lasso), "cv_r2_ols_all": float(r2_ols)}
    lines.append(f"\n**(c) Lasso over {len(names)} candidate terms (features, datasets, all pairwise interactions)**: "
                 f"CV-chosen α = {lasso.alpha_:.4f} keeps {len(kept)} terms. Out-of-sample R² (10-fold): Lasso {r2_lasso:.3f} "
                 f"vs unpenalised OLS on all terms {r2_ols:.3f}. Largest kept terms: "
                 + ", ".join(f"{n} ({c:+.3f})" for n, c in kept[:6]) + ".")

    # figure: GLM odds ratios and OLS effects
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    for i, f in enumerate(feats):
        o = S["glm_bm25_fail"]["odds_ratios"][f]
        ax.errorbar(o[0], i, xerr=[[o[0] - o[1]], [o[2] - o[0]]], fmt="o", color=NAVY, capsize=3)
    ax.axvline(1, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(feats)), feats)
    ax.invert_yaxis()
    ax.set_title("BM25 failure: odds ratio per +1 SD")
    ax = axes[1]
    for i, f in enumerate(feats):
        c = S["ols_m3_hc3"][f]
        ax.errorbar(c[0], i, xerr=[[c[0] - c[1]], [c[2] - c[0]]], fmt="o", color=ORANGE, capsize=3)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(feats)), feats)
    ax.invert_yaxis()
    ax.set_title("Dense − BM25: effect per +1 SD (HC3)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    fig.tight_layout()
    fig.savefig(RES / "fig_models.png", dpi=160)
    plt.close(fig)

    # figure: dense advantage vs lexical overlap
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0, 1, 6)
    w2 = w.assign(d=w["Dense"] - w["BM25"], ob=pd.cut(w["overlap"], bins, include_lowest=True))
    for ds in DATASETS:
        s = w2[w2.dataset == ds].groupby("ob", observed=True)["d"].agg(["mean", "count"])
        s = s[s["count"] >= 15]
        mids = [iv.mid for iv in s.index]
        ax.plot(mids, s["mean"], marker="o", label=LABEL[ds])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Share of query terms that appear in the relevant document")
    ax.set_ylabel("Dense − BM25 (nDCG@10)")
    ax.legend(fontsize=8)
    ax.set_title("Dense retrieval wins when the words don't match")
    fig.tight_layout()
    fig.savefig(RES / "fig_overlap.png", dpi=160)
    plt.close(fig)
    return lines


# --------------------------------------------------------------------------- 5
def power(df, rng, S):
    w = wide(df)
    lines = ["\n### Power analysis: test queries needed (paired test, α = 0.05, power = 0.8)\n",
             "| Dataset | Comparison | SD of per-query diff | Observed diff | n for Δ = 0.01 | n for Δ = 0.02 | n for Δ = 0.05 | Test set has |",
             "|---|---|---|---|---|---|---|---|"]
    out = []
    for ds in DATASETS:
        sub = w[w.dataset == ds]
        for sysn in ("Hybrid", "Dense", "BM25+chunks"):
            d = (sub[sysn] - sub["BM25"]).to_numpy()
            sd = d.std(ddof=1)
            ns = [n_for_power(sd, delta) for delta in (0.01, 0.02, 0.05)]
            out.append(dict(dataset=ds, system=sysn, sd=float(sd), diff=float(d.mean()), n=ns, n_test=len(d)))
            lines.append(f"| {LABEL[ds]} | {sysn} − BM25 | {sd:.3f} | {d.mean():+.3f} | {ns[0]:,} | {ns[1]:,} | {ns[2]:,} | {len(d):,} |")
    S["power"] = out

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ns = np.array([10, 20, 40, 80, 150, 300, 600, 1000])
    for ds in DATASETS:
        d = (w[w.dataset == ds]["Hybrid"] - w[w.dataset == ds]["BM25"]).to_numpy()
        sim = [simulated_power(d, int(n), 0.02, rng) for n in ns]
        line, = ax.plot(ns, sim, marker="o", label=f"{LABEL[ds]} (simulated)")
        from scipy import stats as st
        sd = d.std(ddof=1)
        grid = np.linspace(10, 1000, 200)
        z = 0.02 / (sd / np.sqrt(grid))
        ax.plot(grid, st.norm.cdf(z - 1.96) + st.norm.cdf(-z - 1.96), ls="--", color=line.get_color(), lw=0.8)
    ax.axhline(0.8, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Number of test queries")
    ax.set_ylabel("Power to detect Δ = 0.02 nDCG@10")
    ax.set_title("Hybrid vs BM25: simulated power (points) vs normal approx. (dashed)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / "fig_power.png", dpi=160)
    plt.close(fig)
    return lines


# --------------------------------------------------------------------------- 6
def nested_cv(rng, S):
    g = np.load(RES / "bm25_grid.npz")
    k1s, bs = g["k1"], g["b"]
    di, dj = int(np.where(k1s == 0.9)[0][0]), int(np.where(bs == 0.4)[0][0])
    lines = ["\n### Nested cross-validation of BM25 tuning (5 outer folds × 20 repetitions)\n",
             "| Dataset | In-sample gain of best setting | Out-of-sample gain vs default [95% CI] | Optimism | Settings chosen |",
             "|---|---|---|---|---|"]
    out = []
    for ds in DATASETS:
        a = g[ds]  # (k1, b, queries)
        nq = a.shape[2]
        ins, outs, chosen = [], [], []
        for rep in range(20):
            perm = rng.permutation(nq)
            folds = np.array_split(perm, 5)
            oos = np.zeros(nq)
            for f in range(5):
                test = folds[f]
                train = np.concatenate([folds[k] for k in range(5) if k != f])
                means = a[:, :, train].mean(axis=2)
                bi, bj = np.unravel_index(np.argmax(means), means.shape)
                chosen.append((float(k1s[bi]), float(bs[bj])))
                ins.append(means[bi, bj] - means[di, dj])
                oos[test] = a[bi, bj, test] - a[di, dj, test]
            outs.append(oos)
        outs = np.array(outs)
        gain_oos = outs.mean()
        per_query = outs.mean(axis=0)  # averaged over repetitions
        lo, hi = bootstrap_ci(per_query, rng)
        uniq, cnt = np.unique(np.array(chosen), axis=0, return_counts=True)
        top = sorted(zip(cnt, map(tuple, uniq)), reverse=True)[:2]
        out.append(dict(dataset=ds, in_sample=float(np.mean(ins)), out_of_sample=float(gain_oos), ci=[lo, hi],
                        optimism=float(np.mean(ins) - gain_oos), n_distinct_choices=int(len(uniq)),
                        top_choices=[[c / len(chosen), k] for c, k in top]))
        tc = ", ".join(f"k1={k[0]}, b={k[1]} ({c / len(chosen):.0%})" for c, k in top)
        lines.append(f"| {LABEL[ds]} | {np.mean(ins):+.4f} | {gain_oos:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                     f"{np.mean(ins) - gain_oos:+.4f} | {len(uniq)} distinct; most often {tc} |")
    S["nested_cv"] = out

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(DATASETS))
    ax.bar(x - 0.18, [o["in_sample"] for o in out], 0.36, color=GREY, label="in-sample (training folds)")
    ax.bar(x + 0.18, [o["out_of_sample"] for o in out], 0.36, color=NAVY, label="out-of-sample (held-out folds)")
    ax.errorbar(x + 0.18, [o["out_of_sample"] for o in out],
                yerr=[[o["out_of_sample"] - o["ci"][0] for o in out], [o["ci"][1] - o["out_of_sample"] for o in out]],
                fmt="none", color="black", capsize=3)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x, [LABEL[d] for d in DATASETS])
    ax.set_ylabel("nDCG@10 gain of tuned BM25 over default")
    ax.legend(fontsize=8)
    ax.set_title("Tuning gains shrink on unseen queries")
    fig.tight_layout()
    fig.savefig(RES / "fig_nested_cv.png", dpi=160)
    plt.close(fig)
    return lines


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(RES / "per_query.csv")
    S: dict = {"n_queries": {d: int((df[(df.dataset == d) & (df.system == "BM25")]).shape[0]) for d in DATASETS}}
    lines = quality_and_tests(df, rng, S)
    fig_quality(S)
    lines += pooling(df, S)
    lines += modelling(df, S)
    lines += power(df, rng, S)
    lines += nested_cv(rng, S)
    (RES / "tables.md").write_text("\n".join(lines) + "\n")
    (RES / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
