"""Step 5 (plan_step5.md): structurally different model families for the public leaderboard.

Every upload before this step was LightGBM, and rolling CV cannot see the test steps that
decide the score, so the board has to compare families. A candidate is worth an upload only
if its test ranking differs from every file already scored (Spearman < 0.98). CV here is a
sanity check (3-fold AUC >= 0.95) and a table for the report, not a veto.

    .venv/bin/python src/families.py ref                     # reproduce sub_06 (wiring check)
    .venv/bin/python src/families.py rf_clean et_clean       # CV, final fit, fam_<name>.csv, ledger row
    .venv/bin/python src/families.py blend sub_06 sub_07     # rank-average scored files
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import ExtraTreesClassifier

from models import LGBM_PARAMS, LGBM_TUNED, logreg_fit_predict, rf_fit_predict
from probes import Data
from validation import FOLDS_WITH_HORIZON, evaluate
from paths import ARTIFACTS, LEDGER, SUBMISSIONS

SUBS = SUBMISSIONS
REFERENCE = "sub_06_lgbm_clean_bagged5.csv"
DISTINCT_MAX, SANITY_MIN = 0.98, 0.95

# Public LB from the Kaggle API (2026-09-27). Add each new file here once it is scored.
BOARD = {
    "sub_01_lgbm_clean.csv": 0.93901,
    "sub_02_lgbm_raw_with_period_ids.csv": 0.94091,
    "sub_03_lgbm_clean_fixed300_unweighted.csv": 0.94510,
    "sub_04_lgbm_v2b_ranknorm_bagged5.csv": 0.94200,
    "sub_05_lgbm_v2_ranknorm_discrete_bagged5.csv": 0.93977,
    "sub_06_lgbm_clean_bagged5.csv": 0.94374,
    "sub_07_lgbm_v2b_tuned_bagged5.csv": 0.94031,
    "sub_08_lgbm_v2b_tuned_stepmean_equalised.csv": 0.94134,
    "probe_adv_weights_hard.csv": 0.93765,
    "fam_rf_clean.csv": 0.92847,
    "sub_03_seed1.csv": 0.94198,
    "fam_recent14.csv": 0.94836,
    "fam_recent7.csv": 0.92830,
    "fam_recent10.csv": 0.93386,
    "fam_recent21.csv": 0.94617,
    "fam_recent14_2stage.csv": 0.95260,
    "fam_recent14_local.csv": 0.93674,
    "fam_recency_hl5.csv": 0.95096,
    "fam_recent14_2stage_tunedB.csv": 0.95504,
    "fam_hl5_2stage.csv": 0.95231,
    "fam_recent14_2stage_seedsB.csv": 0.95361,
    "fam_recent14_2stage_B15.csv": 0.95621,
    "fam_hl5_2stage_tunedB.csv": 0.95518,
    "fam_recent14_2stage_B23.csv": 0.95499,
    "fam_B15_interact.csv": 0.95620,
    "fam_B15_smooth.csv": 0.94462,
    "fam_B15_lambdarank.csv": 0.94502,
    "fam_B15_xgb.csv": 0.95397,
    "fam_B15_cat.csv": 0.94638,
    "fam_B15_et.csv": 0.91782,
    "fam_B15_Aoof_random.csv": 0.95849,
    "fam_B15_nb_mix50.csv": 0.95423,
    "fam_B15_nb_labels.csv": 0.94620,
}


# ---------------------------------------------------------------- families
# fit_predict(Xtr, ytr, Xva, seed) -> p_va, one model per seed; the final prediction is the
# mean over the family's seeds, so bagging is identical in CV and in the submission.

def lgbm(params: dict | None = None, n_estimators: int = 300):
    base = {**LGBM_PARAMS, **(params or {}), "n_estimators": n_estimators}
    def fit_predict(Xtr, ytr, Xva, seed=0):
        return lgb.LGBMClassifier(**{**base, "random_state": seed}).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    return fit_predict


def et_fit_predict(Xtr, ytr, Xva, seed=0):
    m = ExtraTreesClassifier(n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=2,
                             n_jobs=-1, random_state=seed)
    return m.fit(Xtr, ytr).predict_proba(Xva)[:, 1]


_TS = None   # set by run_family: time step per labelled train row (positional index of d.L)

def lgbm_recent(last_k: int, params: dict | None = None, n_estimators: int = 300):
    """sub_06 recipe trained only on the last `last_k` steps of whatever training window it
    is given (steps 22-35 for the final fit). Tests whether recency helps on the test period
    even though it hurts on historical windows."""
    base = {**LGBM_PARAMS, **(params or {}), "n_estimators": n_estimators}
    def fit_predict(Xtr, ytr, Xva, seed=0):
        t = _TS[Xtr.index.to_numpy()]; keep = t >= t.max() - (last_k - 1)
        return lgb.LGBMClassifier(**{**base, "random_state": seed}).fit(Xtr[keep], np.asarray(ytr)[keep]).predict_proba(Xva)[:, 1]
    return fit_predict


def lgbm_recency_weighted(half_life: float, params: dict | None = None, n_estimators: int = 300):
    """sub_06 recipe with exponential sample weights 0.5 ** ((t_last - t) / half_life)."""
    base = {**LGBM_PARAMS, **(params or {}), "n_estimators": n_estimators}
    def fit_predict(Xtr, ytr, Xva, seed=0):
        t = _TS[Xtr.index.to_numpy()]; w = 0.5 ** ((t.max() - t) / half_life)
        return lgb.LGBMClassifier(**{**base, "random_state": seed}).fit(Xtr, ytr, sample_weight=w).predict_proba(Xva)[:, 1]
    return fit_predict


_PHASE = "cv"   # run_family sets "final" before the fit on all rows; tells two-stage which rows Xva holds


def two_stage(last_k: int = 14, params: dict | None = None, n_estimators: int = 300, hop2: bool = False, unlab: bool = False,
              b_params: dict | None = None, b_estimators: int | None = None, a_all_history: bool = False, half_life: float | None = None,
              b_interact: int = 0, b_rank: bool = False, smooth: float | None = None,
              b_learner: str = "lgbm", b_topk: int = 0, b_monotone: bool = False,
              a_oof: str = "time", nb_labels: bool = False, a_bag: int = 1, nb_label_mix: float = 0.0):
    """Two-stage collective model on the recent-window base.

    Model A (sub_06 recipe) scores every row; each row then gets the mean and max of model-A
    scores over its incoming and outgoing neighbours (labelled rows of the same period only,
    so train and test see the same kind of neighbourhood) plus the two counts; model B (same
    recipe) is trained on the window rows with these extra columns.

    Model-A scores for training rows are out-of-fold in time: step s is scored by a model
    trained on all labelled rows with step < s. Validation / test rows are scored by a model
    trained on the window rows. Rows with no scored neighbour get NaN (LightGBM handles it).

    hop2: also min/std of neighbour scores and the mean of neighbours' own neighbour means.
    unlab: model A also scores the unlabelled training rows of the window, which then count
    as neighbours for training rows (test neighbours stay labelled-only: parity deliberately
    broken to test whether it matters)."""
    from probes import Data as _D
    base = {**LGBM_PARAMS, **(params or {}), "n_estimators": n_estimators}
    d = _D(); edges = d.edges
    L_tx, T_tx = d.L["txId"].to_numpy(), d.test["txId"].to_numpy()
    U = d.train[d.train.label.isna()].reset_index(drop=True); U_tx, U_ts = U["txId"].to_numpy(), U["time_step"].to_numpy()

    baseB = {**LGBM_PARAMS, **(b_params or params or {}), "n_estimators": b_estimators or n_estimators}

    class _Bag:
        """a_bag LightGBMs with consecutive seeds; predict_proba averages them."""
        def __init__(self, seed): self.ms = [lgb.LGBMClassifier(**{**base, "random_state": seed * 10 + i}) for i in range(a_bag)]
        def fit(self, X, y, sample_weight=None):
            for m in self.ms: m.fit(X, y, sample_weight=sample_weight)
            return self
        def predict_proba(self, X): return np.mean([m.predict_proba(X) for m in self.ms], axis=0)

    def model(seed):
        return _Bag(seed) if a_bag > 1 else lgb.LGBMClassifier(**{**base, "random_state": seed})

    def modelB(seed):
        return lgb.LGBMClassifier(**{**baseB, "random_state": seed})

    # top raw features by gain (step-1 importance table) for the interaction option
    TOP = list(pd.read_csv(ARTIFACTS / "lgbm_importance_raw_1to28.csv", index_col=0).index[:b_interact]) if b_interact else []

    def interact(X: pd.DataFrame) -> pd.DataFrame:
        """Pairwise products and ratios of the TOP features, appended for model B only."""
        cols = {}
        for i, c1 in enumerate(TOP):
            for c2 in TOP[i + 1:]:
                cols[f"ix_{c1}_x_{c2}"] = X[c1].to_numpy() * X[c2].to_numpy()
                cols[f"ix_{c1}_over_{c2}"] = X[c1].to_numpy() / (np.abs(X[c2].to_numpy()) + 1e-3)
        return pd.concat([X.reset_index(drop=True), pd.DataFrame(cols)], axis=1)

    TOPK = list(pd.read_csv(ARTIFACTS / "lgbm_importance_raw_1to28.csv", index_col=0).index[:b_topk]) if b_topk else None

    def fit_B(A_df, y_win, t_win, w, seed):
        """Model B as a classifier (LightGBM or XGBoost), or as a lambdarank ranker."""
        if b_learner == "et":
            from sklearn.ensemble import ExtraTreesClassifier
            m = ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=20, max_features=0.6, n_jobs=-1, random_state=seed)
            return m.fit(A_df.fillna(-1), y_win, sample_weight=w)
        if b_learner == "cat":
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(iterations=baseB["n_estimators"], learning_rate=baseB["learning_rate"], depth=4,
                                   l2_leaf_reg=3.0, rsm=0.6, random_seed=seed, verbose=False, thread_count=-1, allow_writing_files=False)
            return m.fit(A_df, y_win, sample_weight=w)
        if b_learner == "xgb":
            import xgboost as xgb
            m = xgb.XGBClassifier(n_estimators=baseB["n_estimators"], learning_rate=baseB["learning_rate"], max_depth=4,
                                  subsample=1.0, colsample_bytree=0.6, min_child_weight=10, reg_lambda=1.0, tree_method="hist",
                                  n_jobs=-1, random_state=seed)
            return m.fit(A_df, y_win, sample_weight=w)
        if not b_rank:
            params = dict(baseB)
            if b_monotone:
                params["monotone_constraints"] = [1 if c in ("nb_in_mean", "nb_in_max", "nb_out_mean", "nb_out_max") else 0 for c in A_df.columns]
            return lgb.LGBMClassifier(**{**params, "random_state": seed}).fit(A_df, y_win, sample_weight=w)
        # one group over the whole window: the objective is then pairwise ordering across all
        # rows, which is what pooled AUC measures (per-step groups made cross-step scales
        # arbitrary and the pooled AUC collapsed: 3-fold 0.917)
        # LightGBM caps a query at 10,000 rows: use random groups of <= 5,000 rows (random, not
        # by step, so pairs still span steps and the objective stays close to pooled AUC)
        n = len(y_win); order = np.random.default_rng(seed).permutation(n)
        k = int(np.ceil(n / 5000)); sizes = [len(g) for g in np.array_split(order, k)]
        rk = lgb.LGBMRanker(**{**baseB, "objective": "lambdarank", "lambdarank_truncation_level": 5000, "random_state": seed})
        return rk.fit(A_df.iloc[order], y_win[order], group=sizes)

    def agg(tx_rows: np.ndarray, sc: pd.Series, val: pd.Series, prefix: str) -> pd.DataFrame:
        """Aggregate `val` (indexed by txId) over predecessors and successors of tx_rows, edges
        restricted to nodes in `sc` (the scored node set)."""
        e = edges[edges.txId1.isin(sc.index) & edges.txId2.isin(sc.index)]
        pred = e.assign(v=val.reindex(e.txId1).to_numpy()).groupby("txId2").v.agg(["mean", "max", "min", "std", "size"])
        succ = e.assign(v=val.reindex(e.txId2).to_numpy()).groupby("txId1").v.agg(["mean", "max", "min", "std", "size"])
        out = pd.DataFrame(index=tx_rows)
        for side, g in (("in", pred), ("out", succ)):
            for stat in ("mean", "max", "min", "std"):
                out[f"{prefix}_{side}_{stat}"] = g[stat].reindex(tx_rows).to_numpy()
            out[f"{prefix}_{side}_n"] = g["size"].reindex(tx_rows).fillna(0).to_numpy()
        return out

    def neighbour_feats(tx_rows: np.ndarray, sc: pd.Series) -> pd.DataFrame:
        f = agg(tx_rows, sc, sc, "nb")
        keep = ["nb_in_mean", "nb_in_max", "nb_in_n", "nb_out_mean", "nb_out_max", "nb_out_n"]
        if hop2:
            keep += ["nb_in_min", "nb_in_std", "nb_out_min", "nb_out_std"]
            full = agg(sc.index.to_numpy(), sc, sc, "nb")            # neighbour means for every scored node
            for side in ("in", "out"):
                h = agg(tx_rows, sc, full[f"nb_{side}_mean"].fillna(sc.mean()), f"h2{side}")
                f[f"h2_{side}_mean"] = h[f"h2{side}_{side}_mean"].to_numpy(); keep.append(f"h2_{side}_mean")
        return f[keep].reset_index(drop=True)

    def fit_predict(Xtr, ytr, Xva, seed=0):
        ytr = np.asarray(ytr); t = _TS[Xtr.index.to_numpy()]
        win = np.ones(len(t), bool) if half_life else t >= t.max() - (last_k - 1)
        def wts(mask, t_ref):   # exponential recency weights relative to t_ref; None when unweighted
            return None if not half_life else 0.5 ** ((t_ref - t[mask]) / half_life)
        pA = np.full(len(Xtr), np.nan)
        scored = {}
        if a_oof == "random":
            # 5-fold random out-of-fold scoring within the window: each row scored by a model that
            # saw the other 4/5 of the window (same periods), the way test rows are scored by a
            # model that knows their period
            from sklearn.model_selection import StratifiedKFold
            idx_w = np.flatnonzero(win)
            for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx_w, ytr[idx_w]):
                a_i, b_i = idx_w[tr_i], idx_w[va_i]
                pA[b_i] = model(seed).fit(Xtr.iloc[a_i], ytr[a_i]).predict_proba(Xtr.iloc[b_i])[:, 1]
        else:
            for s_ in np.unique(t[win]):
                hist = t < s_
                if hist.sum() < 500: continue
                mA_s = model(seed).fit(Xtr[hist], ytr[hist], sample_weight=wts(hist, s_ - 1))
                pA[t == s_] = mA_s.predict_proba(Xtr[t == s_])[:, 1]
                if unlab:
                    m_u = U_ts == s_
                    if m_u.any():
                        scored.update(zip(U_tx[m_u], mA_s.predict_proba(U[Xtr.columns][m_u])[:, 1]))
        if nb_labels:
            # training rows: neighbour features from the true labels of labelled neighbours
            pA = np.where(win, ytr.astype(float), np.nan)
        elif nb_label_mix:
            # training rows: a mix of the true label and the model-A score
            pA = np.where(np.isnan(pA), np.nan, nb_label_mix * ytr.astype(float) + (1 - nb_label_mix) * pA)
        tx_tr = L_tx[Xtr.index.to_numpy()]
        sc_tr = pd.Series(pA, index=tx_tr); sc_tr = sc_tr[sc_tr.notna()]
        if unlab and scored:
            sc_tr = pd.concat([sc_tr, pd.Series(scored)])
        mA = model(seed).fit(Xtr, ytr) if a_all_history else model(seed).fit(Xtr[win], ytr[win], sample_weight=wts(win, t.max()))
        pA_va = mA.predict_proba(Xva)[:, 1]
        tx_va = T_tx[Xva.index.to_numpy()] if _PHASE == "final" else L_tx[Xva.index.to_numpy()]
        Ftr = neighbour_feats(tx_tr, sc_tr); Fva = neighbour_feats(tx_va, pd.Series(pA_va, index=tx_va))
        Xtr_b, Xva_b = (interact(Xtr), interact(Xva)) if b_interact else (Xtr.reset_index(drop=True), Xva.reset_index(drop=True))
        if TOPK:
            Xtr_b, Xva_b = Xtr_b[TOPK], Xva_b[TOPK]
        A = pd.concat([Xtr_b, Ftr], axis=1)[win]
        B = pd.concat([Xva_b, Fva], axis=1)
        mB = fit_B(A, ytr[win], t[win], wts(win, t.max()), seed)
        pB = mB.predict(B) if b_rank else mB.predict_proba(B.fillna(-1) if b_learner == "et" else B)[:, 1]
        if b_rank:
            pB = rankdata(pB) / len(pB)
        if smooth:
            sc = pd.Series(pB, index=tx_va)
            e = edges[edges.txId1.isin(sc.index) & edges.txId2.isin(sc.index)]
            both = pd.concat([e.assign(node=e.txId2, v=sc.reindex(e.txId1).to_numpy()), e.assign(node=e.txId1, v=sc.reindex(e.txId2).to_numpy())])
            nb = both.groupby("node").v.mean().reindex(tx_va).to_numpy()
            nb = np.where(np.isnan(nb), pB, nb)
            pB = (1 - smooth) * rankdata(pB) / len(pB) + smooth * rankdata(nb) / len(nb)
        return pB
    return fit_predict


def two_stage_iter2(last_k: int = 14, params: dict | None = None, n_estimators: int = 300, min_hist_steps: int = 3):
    """Second pass of the two-stage model. Model B's scores (out-of-fold in time for the
    training rows: step s scored by a model B trained on window rows with step < s, once at
    least `min_hist_steps` such steps exist) become a second set of neighbour features, and a
    model C is trained on features + first-pass + second-pass neighbour numbers."""
    from probes import Data as _D
    base = {**LGBM_PARAMS, **(params or {}), "n_estimators": n_estimators}
    d = _D(); edges = d.edges
    L_tx, T_tx = d.L["txId"].to_numpy(), d.test["txId"].to_numpy()

    def model(seed):
        return lgb.LGBMClassifier(**{**base, "random_state": seed})

    def neighbour_feats(tx_rows, sc: pd.Series, prefix: str) -> pd.DataFrame:
        sc = sc[sc.notna()]
        e = edges[edges.txId1.isin(sc.index) & edges.txId2.isin(sc.index)]
        pred = e.assign(v=sc.reindex(e.txId1).to_numpy()).groupby("txId2").v.agg(["mean", "max", "size"])
        succ = e.assign(v=sc.reindex(e.txId2).to_numpy()).groupby("txId1").v.agg(["mean", "max", "size"])
        out = pd.DataFrame(index=tx_rows)
        for side, g in (("in", pred), ("out", succ)):
            out[f"{prefix}_{side}_mean"] = g["mean"].reindex(tx_rows).to_numpy()
            out[f"{prefix}_{side}_max"] = g["max"].reindex(tx_rows).to_numpy()
            out[f"{prefix}_{side}_n"] = g["size"].reindex(tx_rows).fillna(0).to_numpy()
        return out.reset_index(drop=True)

    def fit_predict(Xtr, ytr, Xva, seed=0):
        ytr = np.asarray(ytr); t = _TS[Xtr.index.to_numpy()]; win = t >= t.max() - (last_k - 1)
        tx_tr = L_tx[Xtr.index.to_numpy()]
        tx_va = T_tx[Xva.index.to_numpy()] if _PHASE == "final" else L_tx[Xva.index.to_numpy()]
        # pass 1: model A, out-of-fold in time
        pA = np.full(len(Xtr), np.nan)
        for s_ in np.unique(t[win]):
            hist = t < s_
            if hist.sum() >= 500:
                pA[t == s_] = model(seed).fit(Xtr[hist], ytr[hist]).predict_proba(Xtr[t == s_])[:, 1]
        pA_va = model(seed).fit(Xtr[win], ytr[win]).predict_proba(Xva)[:, 1]
        F1tr = neighbour_feats(tx_tr, pd.Series(pA, index=tx_tr), "nb"); F1va = neighbour_feats(tx_va, pd.Series(pA_va, index=tx_va), "nb")
        Xb = pd.concat([Xtr.reset_index(drop=True), F1tr], axis=1); Xb_va = pd.concat([Xva.reset_index(drop=True), F1va], axis=1)
        # pass 2: model B scores, out-of-fold in time inside the window
        pB = np.full(len(Xtr), np.nan); wsteps = np.unique(t[win])
        for s_ in wsteps:
            hist = win & (t < s_)
            if (np.unique(t[hist]).size >= min_hist_steps):
                pB[t == s_] = model(seed).fit(Xb[hist], ytr[hist]).predict_proba(Xb[t == s_])[:, 1]
        pB_va = model(seed).fit(Xb[win], ytr[win]).predict_proba(Xb_va)[:, 1]
        F2tr = neighbour_feats(tx_tr, pd.Series(pB, index=tx_tr), "nb2"); F2va = neighbour_feats(tx_va, pd.Series(pB_va, index=tx_va), "nb2")
        Xc = pd.concat([Xb, F2tr], axis=1); Xc_va = pd.concat([Xb_va, F2va], axis=1)
        return model(seed).fit(Xc[win], ytr[win]).predict_proba(Xc_va)[:, 1]
    return fit_predict


B15_PARAMS = {"num_leaves": 15, "min_child_samples": 40, "colsample_bytree": 0.6, "subsample": 1.0, "subsample_freq": 1, "reg_lambda": 1.0, "reg_alpha": 0.0, "learning_rate": 0.05}
BAG5, BAG3 = (0, 1, 2, 3, 4), (0, 1, 2)
# name: (description, feature set, fit_predict, final seeds); "clean" = 159 columns without
# the six period identifiers, "all" = the 165 raw columns
FAMILIES = {
    "ref": ("sub_06 recipe: LightGBM 31 leaves, 300 rounds, unweighted, bagged x5", "clean", lgbm(), BAG5),
    "rf_clean": ("random forest, 500 trees, balanced_subsample, min leaf 2, bagged x3", "clean", rf_fit_predict, BAG3),
    "et_clean": ("extra trees, 500 trees, balanced_subsample, min leaf 2, bagged x3", "clean", et_fit_predict, BAG3),
    "lgbm_deep_raw": ("sub_06 recipe with 127 leaves, min_child_samples 20, bagged x5", "clean",
                      lgbm({"num_leaves": 127, "min_child_samples": 20}), BAG5),
    "lgbm_raw_ids": ("sub_06 recipe + the six period identifiers, bagged x5", "all", lgbm(), BAG5),
    "logreg": ("logistic regression, standardised, clipped at 10 sd, C=0.1, balanced", "clean", logreg_fit_predict, (0,)),
    # step 6: hypotheses our historical CV rejects, tested on the board (plan_step4/5 lesson)
    "recent14": ("sub_06 recipe trained on the last 14 training steps only (22-35), bagged x5", "clean", lgbm_recent(14), BAG5),
    "recent21": ("sub_06 recipe trained on the last 21 training steps only (15-35), bagged x5", "clean", lgbm_recent(21), BAG5),
    "recent10": ("sub_06 recipe trained on the last 10 training steps only (26-35), bagged x5", "clean", lgbm_recent(10), BAG5),
    "recent17": ("sub_06 recipe trained on the last 17 training steps only (19-35), bagged x5", "clean", lgbm_recent(17), BAG5),
    "recent14_norm": ("recent14 base + within-step rank of the 9 drifting features (v2b), bagged x5", "v2b", lgbm_recent(14), BAG5),
    "recent14_2stage": ("recent14 base + two-stage neighbour predictions (mean/max/count of model-A scores over incoming and outgoing labelled neighbours), bagged x5", "clean", two_stage(14), BAG5),
    "recent14_local": ("recent14 base on the 93 local columns only (aggregated block 94-165 dropped), bagged x5", "local", lgbm_recent(14), BAG5),
    "recent14_2stage_hop2": ("recent14 two-stage + min/std of neighbour scores and 2-hop neighbour-score means, bagged x5", "clean", two_stage(14, hop2=True), BAG5),
    "recent14_2stage_unlab": ("recent14 two-stage with the unlabelled training rows scored by model A as extra neighbours (train side only), bagged x5", "clean", two_stage(14, unlab=True), BAG5),
    "recent21_2stage": ("two-stage on the 21-step window (15-35), bagged x5", "clean", two_stage(21), BAG5),
    "recent14_2stage_iter2": ("recent14 two-stage, second pass: model-B scores as a second set of neighbour features into a model C, bagged x5", "clean", two_stage_iter2(14), BAG5),
    "recent14_2stage_tunedB": ("recent14 two-stage with model B = tuned shallow LightGBM (7 leaves, 600 rounds, lr 0.05), bagged x5", "clean",
                               two_stage(14, b_params={k: LGBM_TUNED[k] for k in ("num_leaves", "min_child_samples", "colsample_bytree", "subsample", "subsample_freq", "reg_lambda", "reg_alpha", "learning_rate")}, b_estimators=600), BAG5),
    "recent14_2stage_Aall": ("recent14 two-stage with model A for val/test rows trained on all 35 steps (model B still on the window), bagged x5", "clean", two_stage(14, a_all_history=True), BAG5),
    "hl5_2stage": ("two-stage on the soft-weighted base (all 35 steps, half-life 5) instead of the hard 14-step window, bagged x5", "clean", two_stage(half_life=5.0), BAG5),
    "recent14_2stage_seedsB": ("recent14 two-stage, seeds 5-9 (replicate of the 0.9526 file to measure its seed noise on the board)", "clean", two_stage(14), (5, 6, 7, 8, 9)),
    "hl5_2stage_tunedB": ("two-stage on the soft-weighted base (half-life 5) with model B = tuned shallow LightGBM, bagged x5", "clean",
                          two_stage(half_life=5.0, b_params={k: LGBM_TUNED[k] for k in ("num_leaves", "min_child_samples", "colsample_bytree", "subsample", "subsample_freq", "reg_lambda", "reg_alpha", "learning_rate")}, b_estimators=600), BAG5),
    "recent14_2stage_B15": ("recent14 two-stage with model B = 15 leaves, 300 rounds, lr 0.05, no row subsampling (middle depth), bagged x5", "clean",
                            two_stage(14, b_params={"num_leaves": 15, "min_child_samples": 40, "colsample_bytree": 0.6, "subsample": 1.0, "subsample_freq": 1, "reg_lambda": 1.0, "reg_alpha": 0.0, "learning_rate": 0.05}, b_estimators=300), BAG5),
    "recent14_2stage_tunedAB": ("recent14 two-stage with both model A and model B = tuned shallow LightGBM, bagged x5", "clean",
                                two_stage(14, params={k: LGBM_TUNED[k] for k in ("num_leaves", "min_child_samples", "colsample_bytree", "subsample", "subsample_freq", "reg_lambda", "reg_alpha", "learning_rate")}, n_estimators=600), BAG5),
    "recent14_2stage_B5": ("recent14 two-stage with model B = 5 leaves, 900 rounds, lr 0.05, no row subsampling (shallower than the best), bagged x5", "clean",
                           two_stage(14, b_params={"num_leaves": 5, "min_child_samples": 40, "colsample_bytree": 0.6, "subsample": 1.0, "subsample_freq": 1, "reg_lambda": 1.0, "reg_alpha": 0.0, "learning_rate": 0.05}, b_estimators=900), BAG5),
    "recent14_2stage_B23": ("recent14 two-stage with model B = 23 leaves, 300 rounds, lr 0.05, no row subsampling (between 15 and 31), bagged x5", "clean",
                            two_stage(14, b_params={"num_leaves": 23, "min_child_samples": 40, "colsample_bytree": 0.6, "subsample": 1.0, "subsample_freq": 1, "reg_lambda": 1.0, "reg_alpha": 0.0, "learning_rate": 0.05}, b_estimators=300), BAG5),
    "B15_interact": ("B15 two-stage + pairwise products and ratios of the top-8 raw features for model B (56 columns), bagged x5", "clean",
                     two_stage(14, b_params=B15_PARAMS, b_estimators=300, b_interact=8), BAG5),
    "B15_lambdarank": ("B15 two-stage with model B trained with the lambdarank objective, random groups of <= 5,000 rows (pairwise ordering across steps), bagged x5", "clean",
                       two_stage(14, b_params=B15_PARAMS, b_estimators=300, b_rank=True), BAG5),
    "B15_smooth": ("B15 two-stage + final-score smoothing along labelled edges (lambda 0.3), bagged x5", "clean",
                   two_stage(14, b_params=B15_PARAMS, b_estimators=300, smooth=0.3), BAG5),
    "B15_xgb": ("B15 two-stage with model B = shallow XGBoost (depth 4, 300 rounds, lr 0.05) on the same inputs, bagged x5", "clean",
                two_stage(14, b_params=B15_PARAMS, b_estimators=300, b_learner="xgb"), BAG5),
    "B15_top80": ("B15 two-stage with model B on the top-80 raw features by gain + 6 neighbour numbers, bagged x5", "clean",
                  two_stage(14, b_params=B15_PARAMS, b_estimators=300, b_topk=80), BAG5),
    "B15_mono": ("B15 two-stage with model B monotone non-decreasing in the four neighbour mean/max columns, bagged x5", "clean",
                 two_stage(14, b_params=B15_PARAMS, b_estimators=300, b_monotone=True), BAG5),
    "B15_cat": ("B15 two-stage with model B = CatBoost (depth 4, 300 iterations, lr 0.05) on the same inputs, bagged x5", "clean",
                two_stage(14, b_params=B15_PARAMS, b_estimators=300, b_learner="cat"), BAG5),
    "B15_reg": ("B15 two-stage with model B strongly regularised (15 leaves, min 100 rows/leaf, 30% columns, lambda 5), bagged x5", "clean",
                two_stage(14, b_params={**B15_PARAMS, "min_child_samples": 100, "colsample_bytree": 0.3, "reg_lambda": 5.0}, b_estimators=300), BAG5),
    "B15_et": ("B15 two-stage with model B = ExtraTrees (500 trees, depth 8, min 20 rows/leaf) on the same inputs, bagged x5", "clean",
               two_stage(14, b_params=B15_PARAMS, b_estimators=300, b_learner="et"), BAG5),
    "B15_Aoof_random": ("B15 two-stage with model-A scores for training rows from 5-fold random OOF within the window (not time-ordered), bagged x5", "clean",
                        two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random"), BAG5),
    "B15_nb_labels": ("B15 two-stage with model B trained on neighbours' true labels (test uses model-A scores), bagged x5", "clean",
                      two_stage(14, b_params=B15_PARAMS, b_estimators=300, nb_labels=True), BAG5),
    "B15_Abag3": ("B15 two-stage with model A = 3-seed bag inside each outer seed (smoother neighbour scores), bagged x5", "clean",
                  two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_bag=3), BAG5),
    "B15_nb_mix50": ("B15 two-stage with training-row neighbour features from a 50/50 mix of true labels and model-A scores, bagged x5", "clean",
                     two_stage(14, b_params=B15_PARAMS, b_estimators=300, nb_label_mix=0.5), BAG5),
    "recent7": ("sub_06 recipe trained on the last 7 training steps only (29-35), bagged x5", "clean", lgbm_recent(7), BAG5),
    "recency_hl5": ("sub_06 recipe with exponential recency weights, half-life 5 steps, bagged x5", "clean", lgbm_recency_weighted(5.0), BAG5),
    "all_time": ("sub_06 recipe on all 165 columns + time_step as a feature, bagged x5", "all_time", lgbm(), BAG5),
}


# ------------------------------------------------------------------ checks

def cv_summary(d, fit_predict, X, seeds):
    r, _ = evaluate(fit_predict, X, d.y, d.ts, folds=FOLDS_WITH_HORIZON, seeds=seeds, verbose=False)
    f = r.groupby("fold").auc_pooled.mean()
    std = [k for k in f.index if not k.endswith("22-35")]
    return dict(auc_3fold=f[std].mean(), auc_hard=f[[k for k in std if "22-28" in k]].mean(),
                auc_horizon=f[[k for k in f.index if k.endswith("22-35")]].mean(), pr_auc=r.pr_auc.mean())


def board_spearman(pred, exclude=()) -> pd.Series:
    """Spearman of `pred` against every scored file on disk, highest first."""
    out = {f: spearmanr(pred, pd.read_csv(SUBS / f)["target"]).correlation
           for f in BOARD if f not in exclude and (SUBS / f).exists()}
    return pd.Series(out).sort_values(ascending=False)


def verdict(rho: pd.Series, cv: dict | None) -> str:
    if cv is not None and cv["auc_3fold"] < SANITY_MIN:
        return f"fails sanity gate (3-fold {cv['auc_3fold']:.4f} < {SANITY_MIN})"
    if rho.iloc[0] >= DISTINCT_MAX:
        return f"too close to {rho.index[0]} (Spearman {rho.iloc[0]:.3f}): not uploaded"
    return "passes distinctness gate: upload candidate"


# ------------------------------------------------------------------ output

def write_submission(d, pred, name) -> Path:
    sub = d.sample.copy(); sub["target"] = pred
    assert len(sub) == 15_329 and (sub["index"].to_numpy() == d.sample["index"].to_numpy()).all()
    assert sub["target"].between(0, 1).all() and sub["target"].notna().all()
    path = SUBS / f"fam_{name}.csv"; sub.to_csv(path, index=False); return path


HEADER = ("\n## Step 5: model families (plan_step5.md)\n\n"
          "CV = four rolling folds, one model per seed (2 seeds); a sanity check, not a veto. "
          f"Upload gate: Spearman < {DISTINCT_MAX} against every scored file. Delta is vs {REFERENCE} (0.94374).\n\n"
          "| date | file | family | n features | CV 3-fold | CV hard 22-28 | CV horizon 22-35 | CV PR-AUC | "
          "closest scored file (Spearman) | public LB | delta vs sub_06 | verdict |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def append_ledger(path: Path, desc: str, n_features, cv: dict | None, rho: pd.Series, verdict_: str):
    ledger = LEDGER; txt = ledger.read_text()
    if "## Step 5: model families" not in txt:
        txt += HEADER
    if path.name in txt:
        print(f"  ledger already has {path.name}; row not rewritten"); return
    c = ["" if cv is None else f"{cv[k]:.4f}" for k in ("auc_3fold", "auc_hard", "auc_horizon", "pr_auc")]
    row = (f"| {date.today()} | {path.name} | {desc} | {n_features} | {' | '.join(c)} | "
           f"{rho.index[0].split('.')[0]} ({rho.iloc[0]:.3f}) | _pending_ | | {verdict_} |")
    ledger.write_text(txt + row + "\n")


# ------------------------------------------------------------------ runner

def run_family(d, name):
    desc, fset, fit_predict, seeds = FAMILIES[name]
    global _TS; _TS = d.ts.to_numpy()
    if fset == "v2b":
        Xtr, Xte = d.features(); cols = list(Xtr.columns)
    else:
        cols = {"clean": d.RAW_CLEAN, "all": d.RAW, "all_time": d.RAW + ["time_step"], "local": d.LOCAL}[fset]
        Xtr, Xte = d.L[cols], d.test[cols]
    y = d.y.to_numpy().astype(int)
    print(f"\n=== {name}: {desc}", flush=True)
    cv = cv_summary(d, fit_predict, Xtr, seeds[:2])
    print(f"  CV 3-fold={cv['auc_3fold']:.4f} hard={cv['auc_hard']:.4f} horizon={cv['auc_horizon']:.4f} "
          f"PR={cv['pr_auc']:.4f}", flush=True)
    global _PHASE; _PHASE = "final"
    pred = np.mean([fit_predict(Xtr, y, Xte, seed=s) for s in seeds], axis=0)
    _PHASE = "cv"
    if name == "ref":
        ref = pd.read_csv(SUBS / REFERENCE)["target"].to_numpy()
        print(f"  reference check: max |diff| vs {REFERENCE} = {np.abs(pred - ref).max():.2e}"); return
    rho = board_spearman(pred)
    print("  Spearman vs scored files:\n" + rho.round(3).to_string())
    v = verdict(rho, cv); print(f"  verdict: {v}")
    path = write_submission(d, pred, name); print(f"  wrote {path}")
    json.dump({**cv, "spearman_vs_board": rho.round(4).to_dict()}, open(ARTIFACTS / f"family_{name}.json", "w"), indent=1)
    append_ledger(path, desc, len(cols), cv, rho, v)


def resolve(key: str) -> str:
    """'sub_06' -> 'sub_06_lgbm_clean_bagged5.csv'; must match exactly one file."""
    hits = sorted(p.name for p in SUBS.glob(f"{key}*.csv"))
    assert len(hits) == 1, f"{key!r} matches {hits}"
    return hits[0]


def run_blend(d, keys):
    files = [resolve(k) for k in keys]
    name = "blend_" + "_".join(k.replace("sub_", "").replace("fam_", "") for k in keys)
    print(f"\n=== {name}: equal-weight rank average of {files}", flush=True)
    pred = np.mean([rankdata(pd.read_csv(SUBS / f)["target"]) / len(d.sample) for f in files], axis=0)
    rho = board_spearman(pred)
    print("  Spearman vs scored files:\n" + rho.round(3).to_string())
    v = verdict(rho, None); print(f"  verdict: {v}")
    path = write_submission(d, pred, name); print(f"  wrote {path}")
    append_ledger(path, f"rank average of {', '.join(keys)}", "-", None, rho, v)


if __name__ == "__main__":
    args = sys.argv[1:]
    d = Data()
    if args[:1] == ["blend"]:
        run_blend(d, args[1:])
    else:
        for n in args:
            run_family(d, n)
