"""Cross-validation of rank-average blends of two-stage pipelines (plan_step11.md, round 8).

A member's own CV says little about what it adds to a blend (the block model trailed Final 1's model
on CV seeds 5-6 yet lifted the blend to 0.96539 on the board). This scores the blends themselves on
the same rolling folds as `families.cv_summary` (three 7-step folds + the 14-step horizon fold):
every member pipeline is run once per fold and seed, its validation predictions are cached in
artifacts/blend_cv_preds.pkl, and any blend is then an equal-weight average of pooled ranks over the
validation rows, built the way the submission blend is.

    .venv/bin/python src/blend_cv.py            # run missing members, print the blend table
"""
from __future__ import annotations

import itertools
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import families
from families import FAMILIES
from paths import ARTIFACTS
from probes import Data
from validation import FOLDS_WITH_HORIZON, rolling_origin_folds, score

MEMBERS = {                      # blend name -> FAMILIES entry (its fit_predict is what is run)
    "goof": "B15_nbfeat_all_goof",            # Final 2 / half of Final 1
    "blocks": "B15_nbfeat_all_goof_blocks",   # the other half of Final 1
    "xgb": "B15_nbfeat_all_goof_xgb",         # held third member (round 7)
    "soof": "B15_nbfeat_all_soof",            # leave-one-step-out protocol, for protocol diversity
}
SEEDS = (0, 1, 5, 6)             # the two CV seed pairs used throughout step 11
CACHE = ARTIFACTS / "blend_cv_preds.pkl"


def member_predictions(d: Data) -> dict:
    """{(member, fold, seed): validation predictions}, computed once and cached."""
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    families._TS = d.ts.to_numpy(); families._PHASE = "cv"
    X, y = d.L[d.RAW_CLEAN], d.y
    labelled = y.notna().to_numpy()
    for m, fam in MEMBERS.items():
        fp = FAMILIES[fam][2]
        for fold, tr, va in rolling_origin_folds(d.ts, FOLDS_WITH_HORIZON):
            tr &= labelled; va &= labelled
            for seed in SEEDS:
                key = (m, fold.name, seed)
                if key in cache: continue
                cache[key] = fp(X[tr], y[tr].to_numpy().astype(int), X[va], seed)
                pickle.dump(cache, open(CACHE, "wb"))
                print(f"  {m:7s} {fold.name:28s} seed {seed}", flush=True)
    return cache


def blend_table(d: Data, cache: dict) -> pd.DataFrame:
    """CV of every member and every equal-weight rank-average blend, per seed pair."""
    y, ts = d.y, d.ts
    folds = [(f, va & y.notna().to_numpy()) for f, _, va in rolling_origin_folds(ts, FOLDS_WITH_HORIZON)]
    combos = [c for k in range(1, len(MEMBERS) + 1) for c in itertools.combinations(MEMBERS, k)]
    rows = []
    for combo in combos:
        for pair in ((0, 1), (5, 6)):
            per_fold = {}
            for fold, va in folds:
                yv, tv = y[va].to_numpy().astype(int), ts[va].to_numpy()
                aucs = []
                for seed in pair:
                    p = np.mean([rankdata(cache[(m, fold.name, seed)]) / va.sum() for m in combo], axis=0)
                    aucs.append(score(yv, p, tv)["auc_pooled"])
                per_fold[fold.name] = np.mean(aucs)
            std = [k for k in per_fold if not k.endswith("22-35")]
            rows.append({"blend": " + ".join(combo), "seeds": f"{pair[0]}-{pair[1]}",
                         "auc_3fold": np.mean([per_fold[k] for k in std]),
                         "auc_hard": np.mean([per_fold[k] for k in std if "22-28" in k]),
                         "auc_horizon": np.mean([per_fold[k] for k in per_fold if k.endswith("22-35")])})
    t = pd.DataFrame(rows)
    mean = t.groupby("blend")[["auc_3fold", "auc_hard", "auc_horizon"]].mean().add_suffix("_mean")
    wide = t.pivot(index="blend", columns="seeds", values=["auc_3fold", "auc_hard", "auc_horizon"])
    wide.columns = [f"{m}_s{s}" for m, s in wide.columns]
    return wide.join(mean).sort_values("auc_3fold_mean", ascending=False)


if __name__ == "__main__":
    d = Data()
    cache = member_predictions(d)
    t = blend_table(d, cache)
    t.round(4).to_csv(ARTIFACTS / "blend_cv.csv")
    pd.set_option("display.width", 250)
    print(t.round(4).to_string())
