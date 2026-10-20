"""Offline pre-screen for post-43 block-order candidates (plan_step11.md; ranker only).

The post-43 test block (steps 43-49) is a low-base-rate regime (~1.3-2.4% fraud on the public
sample). The only in-data analogue is the low-rate band E = steps {1-6, 10} (9,322 labelled
rows, 107 positives, 0.43-2.08% per step). Each candidate scorer is trained without E and
scored by its pooled AUC within E - how well it orders a low-base-rate regime it never saw.
The regime-matched trainer cannot hold E out (E is its training recipe), so it is scored
leave-one-step-out within the band: train on the other low-rate steps, score the held-out
step, pool the out-of-fold scores.

Caveats recorded in the plan: the proxy is anti-chronological (a later-trained model ordering
the earliest steps) and historical CV has been wrong about temporal choices. This script only
ranks candidates for floorsplice slots; the board reading is the evidence.

    .venv/bin/python src/post43_prescreen.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from models import LGBM_PARAMS
from paths import ARTIFACTS
from probes import Data

E_STEPS = [1, 2, 3, 4, 5, 6, 10]          # the low-rate band (fraud rate < 3%)
EXTRA_LOW = [12, 23]                      # the next-lowest steps (3.2% / 4.5%), training-only
B15 = {"num_leaves": 15, "min_child_samples": 40, "colsample_bytree": 0.6, "subsample": 1.0,
       "subsample_freq": 1, "reg_lambda": 1.0, "reg_alpha": 0.0, "learning_rate": 0.05}
B7 = {**B15, "num_leaves": 7, "min_child_samples": 20}
SEEDS = (0, 1, 2)


def lgbm_scores(Xtr, ytr, Xva, params, n_estimators=300):
    return np.mean([lgb.LGBMClassifier(**{**LGBM_PARAMS, **params, "n_estimators": n_estimators,
                                          "random_state": s}).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
                    for s in SEEDS], axis=0)


def knn_scores(Xtr, ytr, Xva):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    m = make_pipeline(StandardScaler(), FunctionTransformer(lambda z: np.clip(z, -10, 10)),
                      KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1))
    return m.fit(Xtr, ytr).predict_proba(Xva)[:, 1]


def main():
    d = Data()
    y = d.y.to_numpy().astype(int); ts = d.ts.to_numpy()
    in_E = np.isin(ts, E_STEPS)
    yE, tsE = y[in_E], ts[in_E]
    print(f"eval band E = steps {E_STEPS}: {in_E.sum():,} rows, {yE.sum()} positives "
          f"({yE.mean():.2%}) - the in-data analogue of the post-43 block")

    X = d.L[d.RAW_CLEAN]
    XE = X[in_E]

    def train_mask(steps_lo, steps_hi):
        return (ts >= steps_lo) & (ts <= steps_hi) & ~in_E

    cands: dict[str, np.ndarray] = {}

    # window / history candidates (single-stage stand-ins for the two-stage donors)
    cands["win14_31leaves (Final-1 base analogue)"] = lgbm_scores(X[train_mask(22, 35)], y[train_mask(22, 35)], XE, {})
    cands["win14_B15"] = lgbm_scores(X[train_mask(22, 35)], y[train_mask(22, 35)], XE, B15)
    cands["allhist_31leaves (sub_06 analogue)"] = lgbm_scores(X[train_mask(7, 35)], y[train_mask(7, 35)], XE, {})
    cands["allhist_B15"] = lgbm_scores(X[train_mask(7, 35)], y[train_mask(7, 35)], XE, B15)

    # local-columns and rank-normalised variants of the window model
    XL = d.L[d.LOCAL]
    cands["win14_local_31leaves"] = lgbm_scores(XL[train_mask(22, 35)], y[train_mask(22, 35)], XL[in_E], {})
    Xv2b, _ = d.features()
    cands["win14_ranknorm_v2b"] = lgbm_scores(Xv2b[train_mask(22, 35)], y[train_mask(22, 35)], Xv2b[in_E], {})

    # fraud-similarity kNN (trained on everything outside E)
    m_all = ~in_E
    cands["knn50_fraud_similarity"] = knn_scores(X[m_all], y[m_all], XE)

    # regime-matched trainer: leave-one-step-out within the band (+ the extra low steps)
    for pname, params in (("B15", B15), ("B7", B7)):
        oof = np.full(in_E.sum(), np.nan)
        extra = np.isin(ts, EXTRA_LOW)
        for s in E_STEPS:
            tr = (in_E & (ts != s)) | extra
            va_pos = np.flatnonzero(tsE == s)
            oof[va_pos] = lgbm_scores(X[tr], y[tr], XE.iloc[va_pos], params)
        assert not np.isnan(oof).any()
        cands[f"lowrate_LOSO_{pname} (regime-matched)"] = oof

    base = cands["win14_31leaves (Final-1 base analogue)"]
    rows = []
    for name, p in cands.items():
        per_step = [roc_auc_score(yE[tsE == s], p[tsE == s]) for s in E_STEPS
                    if len(np.unique(yE[tsE == s])) == 2]
        rows.append(dict(candidate=name,
                         auc_pooled_E=roc_auc_score(yE, p),
                         auc_step_mean_E=float(np.mean(per_step)),
                         spearman_vs_win14=spearmanr(p, base).correlation))
    out = pd.DataFrame(rows).sort_values("auc_pooled_E", ascending=False)
    out.to_csv(ARTIFACTS / "post43_prescreen.csv", index=False)
    print("\npooled AUC within the low-rate band E (candidates trained without E; "
          "lowrate = leave-one-step-out):")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
