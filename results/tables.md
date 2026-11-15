### nDCG@10 by dataset (mean [95% bootstrap CI])

| System | SciFact (n=300) | NFCorpus (n=323) | FiQA (n=648) | ArguAna (n=1406) |
|---|---|---|---|---|
| TF-IDF | 0.577 [0.529, 0.624] | 0.291 [0.257, 0.326] | 0.156 [0.136, 0.176] | 0.440 [0.421, 0.458] |
| BM25 | 0.662 [0.614, 0.708] | 0.312 [0.278, 0.346] | 0.235 [0.211, 0.259] | 0.383 [0.364, 0.401] |
| BM25+chunks | 0.639 [0.592, 0.683] | 0.297 [0.264, 0.331] | 0.211 [0.188, 0.235] | 0.441 [0.424, 0.460] |
| Dense | 0.701 [0.656, 0.744] | 0.344 [0.309, 0.380] | 0.391 [0.364, 0.419] | 0.603 [0.585, 0.622] |
| Hybrid | 0.699 [0.655, 0.743] | 0.341 [0.306, 0.375] | 0.342 [0.316, 0.370] | 0.528 [0.509, 0.546] |

### Each system vs BM25 (paired, nDCG@10; Holm over all 16 tests)

| Dataset | System | Diff [95% CI] | Better / worse | p (Holm) | 20-query check picks opposite winner |
|---|---|---|---|---|---|
| SciFact | TF-IDF | -0.085 [-0.119, -0.052] | 32 / 88 | 0.0016 | 10% |
| SciFact | BM25+chunks | -0.024 [-0.041, -0.008] | 31 / 47 | 0.0135 | 23% |
| SciFact | Dense | +0.038 [-0.000, +0.076] | 80 / 53 | 0.0490 | 31% |
| SciFact | Hybrid | +0.037 [+0.014, +0.060] | 75 / 33 | 0.0068 | 20% |
| NFCorpus | TF-IDF | -0.020 [-0.031, -0.010] | 55 / 106 | 0.0016 | 17% |
| NFCorpus | BM25+chunks | -0.014 [-0.021, -0.008] | 51 / 95 | 0.0016 | 13% |
| NFCorpus | Dense | +0.032 [+0.010, +0.055] | 128 / 85 | 0.0135 | 25% |
| NFCorpus | Hybrid | +0.029 [+0.017, +0.041] | 117 / 59 | 0.0016 | 11% |
| FiQA | TF-IDF | -0.079 [-0.097, -0.062] | 70 / 209 | 0.0016 | 4% |
| FiQA | BM25+chunks | -0.023 [-0.034, -0.013] | 86 / 138 | 0.0016 | 22% |
| FiQA | Dense | +0.157 [+0.131, +0.183] | 321 / 99 | 0.0016 | 2% |
| FiQA | Hybrid | +0.108 [+0.091, +0.124] | 292 / 49 | 0.0016 | 1% |
| ArguAna | TF-IDF | +0.057 [+0.041, +0.073] | 508 / 348 | 0.0016 | 20% |
| ArguAna | BM25+chunks | +0.059 [+0.047, +0.070] | 517 / 195 | 0.0016 | 11% |
| ArguAna | Dense | +0.221 [+0.201, +0.241] | 811 / 222 | 0.0016 | 1% |
| ArguAna | Hybrid | +0.145 [+0.134, +0.157] | 741 / 104 | 0.0016 | 0% |

### Does fusing BM25 into dense retrieval help? (Hybrid − Dense, paired, Holm over 4 tests)

| Dataset | Diff [95% CI] | Better / worse | p (Holm) |
|---|---|---|---|
| SciFact | -0.001 [-0.029, +0.025] | 59 / 43 | 1.0000 |
| NFCorpus | -0.003 [-0.018, +0.012] | 96 / 106 | 1.0000 |
| FiQA | -0.049 [-0.069, -0.029] | 144 / 221 | 0.0004 |
| ArguAna | -0.075 [-0.090, -0.060] | 316 / 541 | 0.0004 |

### Pooling across datasets (effect vs BM25 on nDCG@10)

| Comparison | Mixed model: mean [95% CI] | Bayesian: mean [95% CrI] | P(effect > 0) | Between-dataset SD (tau) | Predicted effect on a new dataset [95%] | P(new > 0) |
|---|---|---|---|---|---|---|
| Dense − BM25 | +0.113 [+0.022, +0.204] | +0.112 [-0.004, +0.226] | 0.97 | 0.092 [0.049, 0.220] | +0.112 [-0.144, +0.366] | 0.85 |
| Hybrid − BM25 | +0.081 [+0.025, +0.136] | +0.080 [+0.003, +0.157] | 0.98 | 0.060 [0.031, 0.156] | +0.080 [-0.093, +0.250] | 0.87 |
| BM25+chunks − BM25 | -0.000 [-0.040, +0.040] | -0.001 [-0.059, +0.057] | 0.48 | 0.044 [0.022, 0.121] | -0.001 [-0.132, +0.128] | 0.49 |

### When does retrieval fail? Statistical models of per-query outcomes

**(a) Logistic GLM: P(BM25 finds nothing relevant in top 10)**, n = 2677, failure rate 37.9%, deviance explained 0.38. Odds ratios per +1 SD, dataset fixed effects:

| Feature | Odds ratio [95% CI] | p |
|---|---|---|
| log_q_len | 0.31 [0.21, 0.46] | 4e-09 |
| mean_idf | 0.40 [0.31, 0.51] | 5.1e-14 |
| max_idf | 0.56 [0.42, 0.74] | 3.5e-05 |
| oov_frac | 0.62 [0.46, 0.84] | 0.0019 |
| overlap | 0.01 [0.01, 0.02] | 4.5e-93 |
| log_n_rel | 0.96 [0.72, 1.28] | 0.79 |

**(b) Linear models for Dense − BM25 (nDCG@10), selected by AIC/BIC**

| Model | params | AIC | BIC | adj. R² |
|---|---|---|---|---|
| M0  intercept | 0 | 2127 | 2133 | 0.000 |
| M1  dataset | 3 | 2016 | 2039 | 0.042 |
| M2  dataset + overlap | 4 | 1955 | 1984 | 0.064 |
| M3  dataset + all features | 9 | 1917 | 1976 | 0.079 |
| M4  M3 + overlap x dataset ← lowest BIC | 12 | 1832 | 1908 | 0.109 |

M3 coefficients (per +1 SD, heteroskedasticity-robust HC3 CIs):

| Feature | Effect on Dense − BM25 [95% CI] | p |
|---|---|---|
| log_q_len | -0.043 [-0.094, +0.008] | 0.1 |
| mean_idf | -0.017 [-0.036, +0.003] | 0.095 |
| max_idf | -0.013 [-0.041, +0.015] | 0.36 |
| oov_frac | -0.035 [-0.049, -0.021] | 9e-07 |
| overlap | -0.105 [-0.121, -0.088] | 1.2e-35 |
| log_n_rel | +0.056 [+0.037, +0.076] | 1.7e-08 |

**(c) Lasso over 45 candidate terms (features, datasets, all pairwise interactions)**: CV-chosen α = 0.0006 keeps 31 terms. Out-of-sample R² (10-fold): Lasso 0.092 vs unpenalised OLS on all terms 0.093. Largest kept terms: overlap (-0.297), overlap fiqa (+0.089), overlap nfcorpus (+0.083), overlap scifact (+0.056), log_q_len (-0.050), log_q_len nfcorpus (+0.041).

### Power analysis: test queries needed (paired test, α = 0.05, power = 0.8)

| Dataset | Comparison | SD of per-query diff | Observed diff | n for Δ = 0.01 | n for Δ = 0.02 | n for Δ = 0.05 | Test set has |
|---|---|---|---|---|---|---|---|
| SciFact | Hybrid − BM25 | 0.206 | +0.037 | 3,334 | 834 | 134 | 300 |
| SciFact | Dense − BM25 | 0.337 | +0.038 | 8,926 | 2,232 | 358 | 300 |
| SciFact | BM25+chunks − BM25 | 0.147 | -0.024 | 1,705 | 427 | 69 | 300 |
| NFCorpus | Hybrid − BM25 | 0.111 | +0.029 | 976 | 244 | 40 | 323 |
| NFCorpus | Dense − BM25 | 0.203 | +0.032 | 3,250 | 813 | 130 | 323 |
| NFCorpus | BM25+chunks − BM25 | 0.057 | -0.014 | 259 | 65 | 11 | 323 |
| FiQA | Hybrid − BM25 | 0.214 | +0.108 | 3,600 | 900 | 144 | 648 |
| FiQA | Dense − BM25 | 0.342 | +0.157 | 9,177 | 2,295 | 368 | 648 |
| FiQA | BM25+chunks − BM25 | 0.134 | -0.023 | 1,404 | 351 | 57 | 648 |
| ArguAna | Hybrid − BM25 | 0.226 | +0.145 | 4,002 | 1,001 | 161 | 1,406 |
| ArguAna | Dense − BM25 | 0.386 | +0.221 | 11,667 | 2,917 | 467 | 1,406 |
| ArguAna | BM25+chunks − BM25 | 0.216 | +0.059 | 3,653 | 914 | 147 | 1,406 |

### Nested cross-validation of BM25 tuning (5 outer folds × 20 repetitions)

| Dataset | In-sample gain of best setting | Out-of-sample gain vs default [95% CI] | Optimism | Settings chosen |
|---|---|---|---|---|
| SciFact | +0.0066 | +0.0008 [-0.0059, +0.0073] | +0.0058 | 4 distinct; most often k1=1.5, b=0.9 (50%), k1=0.6, b=0.9 (46%) |
| NFCorpus | +0.0015 | +0.0005 [-0.0020, +0.0030] | +0.0010 | 5 distinct; most often k1=1.2, b=0.4 (85%), k1=1.5, b=0.4 (10%) |
| FiQA | +0.0023 | -0.0005 [-0.0067, +0.0057] | +0.0028 | 4 distinct; most often k1=0.6, b=0.75 (82%), k1=0.9, b=0.4 (10%) |
| ArguAna | +0.1018 | +0.1018 [+0.0918, +0.1116] | -0.0000 | 1 distinct; most often k1=2.0, b=0.9 (100%) |
