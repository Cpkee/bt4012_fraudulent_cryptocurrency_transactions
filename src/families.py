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
    "fam_recent21_Aoof_random.csv": 0.95413,
    "fam_recent28_Aoof_random.csv": 0.95298,
    "fam_B15_stack_logreg.csv": 0.95485,
    "fam_B15_stack_knn.csv": 0.95589,
    "fam_B15_pseudo_unlab.csv": 0.95278,
    "fam_B15_nb_steprank.csv": 0.95560,
    "fam_B15_nbfeat.csv": 0.96069,
    "fam_B15_nbfeat_all.csv": 0.96177,
    "fam_B15_nbfeat_all_seedsB.csv": 0.96323,
    "fam_B15_nbfeat_all_bag10.csv": 0.96263,
    "fam_B15_nbfeat_gnn.csv": 0.95570,
    "fam_B15_nbfeat_comm.csv": 0.95732,
    "fam_B15_nbfeat_all_bag15.csv": 0.96313,
    "fam_B15_Aoof_random_bag10.csv": 0.95970,
    "fam_bag15_steprank.csv": 0.94124,
    "fam_bag15_levelamp25.csv": 0.96165,
    "fam_bag15_post43floor.csv": 0.89203,
    "fam_B15_nbfeat_all_bag30.csv": 0.96307,
    "fam_B15_nbfeat_pseudo_pre43.csv": 0.95553,
    "fam_post43floor_sub06.csv": 0.88509,
    "fam_post43floor_lowrate.csv": 0.87796,
    "fam_B15_nbfeat_all_goof.csv": 0.96209,
    "fam_B15_nbfeat_all_goof_bag15.csv": 0.96310,
    "fam_bag15_post43floorrev.csv": 0.83841,
    "fam_B15_nbfeat_all_combo.csv": 0.96362,
    "fam_B15_nbfeat_combo.csv": 0.95948,
    "fam_B15_nbfeat_all_combo_bag15.csv": 0.96316,
    "fam_B15_nbfeat_all_soof.csv": 0.96319,
    "fam_B15_nbfeat_all_soof_bag15.csv": 0.96290,
    "fam_B15_nbfeat_all_goof_blocks.csv": 0.96382,
    "fam_B15_nbfeat_all_goof_blocks_bag15.csv": 0.96385,
    "fam_blend_goof_blocks_bag15.csv": 0.96539,
    "fam_B15_nbfeat_all_goof_xgb.csv": 0.96189,
    "fam_blend3_goof_blocks_xgb_bag15.csv": 0.96532,
    "fam_B15_nbfeat_all_goof_dropadv.csv": 0.95957,
    "fam_B15_nbfeat_all_goof_xt.csv": 0.96678,
    "fam_B15_nbfeat_all_goof_xt_r600.csv": 0.96363,
    "fam_B15_nbfeat_all_goof_xt_r150.csv": 0.96389,
    "fam_F1_xtwin.csv": 0.96861,
    "fam_B15_nbfeat_all_goof_xt_unbal.csv": 0.96996,
    "fam_F1_unbal.csv": 0.97020,
    "fam_B15_nbfeat_all_goof_xt_spw12.csv": 0.96917,
    "fam_F1_unbalw.csv": 0.97032,
    "fam_B15_nbfeat_all_goof_xt_stepunbal.csv": 0.96954,
    "fam_F1_wavg.csv": 0.97066,
    "fam_F1_wavg_win.csv": 0.97034,
    "fam_B15_nbfeat_all_goof_xgb_unbal_bag15.csv": 0.96254,
    "fam_B15_nbfeat_all_goof_xt_seedsBC.csv": 0.96736,
    "fam_B15_nbfeat_all_goof_blocks_xt.csv": 0.96630,
    "fam_blend_B15_nbfeat_all_goof_xt_bag15_B15_nbfeat_all_goof_blocks_xt_bag15.csv": 0.96828,
    "fam_B15_nbfeat_all_goof_xt_w21.csv": 0.96755,
    "fam_B15_nbfeat_all_goof_xt_all.csv": 0.96486,
    "fam_B15_nbfeat_all_goof_xt_w21_seedsBC.csv": 0.96676,
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


def lgbm_steps(steps: tuple, params: dict | None = None, n_estimators: int = 300):
    """sub_06 recipe trained only on the listed time steps (step 11, plan_step11.md: the
    regime-matched post-43 splice donor trains on the low-fraud-rate steps, which the recency
    window excludes entirely). Steps outside the training window a CV fold provides are
    silently absent from that fold's fit."""
    base = {**LGBM_PARAMS, **(params or {}), "n_estimators": n_estimators}
    def fit_predict(Xtr, ytr, Xva, seed=0):
        keep = np.isin(_TS[Xtr.index.to_numpy()], steps)
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
_DRO_LOG: list = []   # per-round Group DRO diagnostics (two_stage(b_dro=...)); run_family writes artifacts/dro_<name>.csv


_COMM = None


def communities(edges: pd.DataFrame, step_of: pd.Series) -> pd.Series:
    """Louvain community id per transaction (string "<step>_<k>"), computed once on the whole
    edge list and cached in ARTIFACTS. Edges never cross steps, so each connected component of
    the edge graph belongs to one step; Louvain (networkx, seed 0, resolution 1) runs on each
    step's graph with every node present in the edge list (featureless test-period nodes are
    structure only). Labels and features play no part, so train and test are treated alike."""
    global _COMM
    if _COMM is not None:
        return _COMM
    cache = ARTIFACTS / "communities_louvain_s0.csv"
    if cache.exists():
        _COMM = pd.read_csv(cache, index_col=0)["comm"]; return _COMM
    import networkx as nx
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    nodes = pd.Index(pd.unique(np.r_[edges.txId1, edges.txId2]))
    a, b = nodes.get_indexer(edges.txId1), nodes.get_indexer(edges.txId2)
    _, comp = connected_components(sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(len(nodes),) * 2), directed=False)
    cs = pd.DataFrame(dict(c=comp, s=step_of.reindex(nodes).to_numpy())).dropna().groupby("c").s.agg(["min", "max"])
    assert (cs["min"] == cs["max"]).all(), "a connected component spans two time steps"
    step = cs["min"].reindex(np.arange(comp.max() + 1)).to_numpy()[comp[a]]
    out = {}
    for s in np.unique(step[~np.isnan(step)]):
        e = edges[step == s]
        G = nx.Graph(); G.add_edges_from(zip(e.txId1, e.txId2))
        for k, c in enumerate(nx.community.louvain_communities(G, seed=0)):
            for n in c: out[n] = f"{int(s)}_{k}"
    _COMM = pd.Series(out, name="comm"); _COMM.to_csv(cache); return _COMM


def two_stage(last_k: int = 14, params: dict | None = None, n_estimators: int = 300, hop2: bool = False, unlab: bool = False,
              b_params: dict | None = None, b_estimators: int | None = None, a_all_history: bool = False, half_life: float | None = None,
              b_interact: int = 0, b_rank: bool = False, smooth: float | None = None,
              b_learner: str = "lgbm", b_topk: int = 0, b_monotone: bool = False,
              a_oof: str = "time", nb_labels: bool = False, a_bag: int = 1, nb_label_mix: float = 0.0,
              oof_folds: int = 5, a_test_from_folds: bool = False,
              a_extra_hist: int = 0, b_pseudo_unlab: bool = False, b_stack_logreg: bool = False, b_stack_model: str = "logreg",
              nb_steprank: bool = False, step_ctx: bool = False, nb_feats: int = 0, nb_feat_stats: tuple = ("mean",), nb_feat_delta: bool = False, nb_feats_into_A: bool = False, nb_feat_hop2: bool = False, b_step_balance: bool = False, b_step_unbal: bool = False, b_gnn_embed: bool = False,
              nb_comm: bool = False, b_pseudo_pre43: int = 0, b_pseudo_post43_neg: bool = False,
              nb_feat_scorew: bool = False, nb_feat_top1: bool = False, nb_extra_k: int = 0, nb_extra_stats: tuple = (), b_blocks: bool = False,
              a_cols: str = "all", b_rsub: int = 0, b_rsub_frac: float = 0.5,
              b_dro: float | None = None, b_dro_rho: float = 8.0, b_dro_ess: float = 0.65, b_dro_rounds: int = 3,
              b_dro_risk: str = "group", b_nbonly: bool = False, b_blocks_inout: bool = False):
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
    broken to test whether it matters).
    nb_comm (step 10, docs/plans/plan_step10.md): the community as the neighbourhood unit. Louvain
    communities on the full per-step graph (unlabelled nodes as structure only, `communities()`);
    three more model-B columns = mean, max and count of model-A scores over the *other* scored
    rows of the row's community (blank where there is none), built like the six numbers."""
    from probes import Data as _D
    base = {**LGBM_PARAMS, **(params or {}), "n_estimators": n_estimators}
    d = _D(); edges = d.edges
    L_tx, T_tx = d.L["txId"].to_numpy(), d.test["txId"].to_numpy()
    t_all = d.ts.to_numpy(); t_all_by_tx = pd.Series(t_all, index=L_tx)
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

    # step 8 (docs/plans/plan_step8.md): neighbour *feature* aggregates for model B — the mean of
    # the top `nb_feats` raw features (step-1 gain order) over labelled in- and out-neighbours,
    # built exactly like the six score numbers (labelled-only neighbour set on both sides, blank
    # where there is no neighbour). CV-rejected on the single-stage in step 1; never board-read.
    IMP_ORDER = list(pd.read_csv(ARTIFACTS / "lgbm_importance_raw_1to28.csv", index_col=0).index) if nb_feats else []

    def feat_agg(tx_rows: np.ndarray, sc_index: pd.Index, vals: pd.DataFrame, stats: tuple | None = None) -> pd.DataFrame:
        """Mean of `vals` (indexed by txId) over predecessors and successors of tx_rows, edges
        restricted to nodes in `sc_index`."""
        e = edges[edges.txId1.isin(sc_index) & edges.txId2.isin(sc_index)]
        parts = []
        for side, src, dst in (("in", e.txId1, e.txId2), ("out", e.txId2, e.txId1)):
            g = vals.reindex(src).set_axis(dst.to_numpy()).groupby(level=0)
            for stat in (stats or nb_feat_stats):           # "mean" (default), optionally "max" / "min" / "std"
                parts.append(g.agg(stat).reindex(tx_rows).add_prefix(f"nbf_{side}_{stat}_"))
        return pd.concat(parts, axis=1).reset_index(drop=True)

    def feat_agg_scorew(tx_rows: np.ndarray, sc: pd.Series, vals: pd.DataFrame) -> pd.DataFrame:
        """Step 11 (plan_step11.md, 2026-10-13): score-weighted mean of `vals` over predecessors and
        successors, weights = the neighbour's model-A score: sum(s_j * x_j) / sum(s_j). Answers
        "what do the *suspicious* neighbours look like" where the unweighted mean answers "what do
        neighbours look like" — a ratio of sums over a variable neighbour set, which the trees
        cannot form from the existing columns. Same labelled-only neighbour set as everything else."""
        e = edges[edges.txId1.isin(sc.index) & edges.txId2.isin(sc.index)]
        parts = []
        for side, src, dst in (("in", e.txId1, e.txId2), ("out", e.txId2, e.txId1)):
            w = sc.reindex(src).to_numpy()
            num = pd.DataFrame(vals.reindex(src).to_numpy() * w[:, None], columns=vals.columns,
                               index=dst.to_numpy()).groupby(level=0).sum()
            den = pd.Series(w, index=dst.to_numpy()).groupby(level=0).sum().replace(0, np.nan)
            parts.append(num.div(den, axis=0).reindex(tx_rows).add_prefix(f"nbf_{side}_wmean_"))
        return pd.concat(parts, axis=1).reset_index(drop=True)

    def feat_agg_top1(tx_rows: np.ndarray, sc: pd.Series, vals: pd.DataFrame) -> pd.DataFrame:
        """Step 11: the raw features of the single highest-scoring predecessor and successor (the
        most suspicious neighbour's profile). A selection, not an aggregate: per-feature max was
        unreadable, the argmax-neighbour's vector was never tried. Ties break deterministically."""
        e = edges[edges.txId1.isin(sc.index) & edges.txId2.isin(sc.index)]
        parts = []
        for side, src, dst in (("in", e.txId1, e.txId2), ("out", e.txId2, e.txId1)):
            df = pd.DataFrame({"dst": dst.to_numpy(), "s": sc.reindex(src).to_numpy(), "src": src.to_numpy()})
            best = df.sort_values("s", kind="stable").groupby("dst").src.last()
            prof = vals.reindex(best.to_numpy()).set_axis(best.index)
            parts.append(prof.reindex(tx_rows).add_prefix(f"nbf_{side}_top1_"))
        return pd.concat(parts, axis=1).reset_index(drop=True)

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
            # round 26 (2026-10-24): is_unbalance in b_params -> XGBoost scale_pos_weight = the window's licit/illicit
            # ratio (what LightGBM's is_unbalance applies); absent -> 1.0, XGBoost's default (earlier xgb files unchanged)
            spw = float((y_win == 0).sum() / (y_win == 1).sum()) if baseB.get("is_unbalance") else 1.0
            m = xgb.XGBClassifier(n_estimators=baseB["n_estimators"], learning_rate=baseB["learning_rate"], max_depth=4,
                                  subsample=1.0, colsample_bytree=0.6, min_child_weight=10, reg_lambda=1.0, tree_method="hist",
                                  n_jobs=-1, random_state=seed, scale_pos_weight=spw)
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
        # step 9 (docs/plans/plan_step9.md): the neighbour feature means need no model, so they can
        # also feed model A (train: labelled window rows as the neighbour set; test: the test file)
        Xa_tr, Xa_va = Xtr, Xva
        if nb_feats_into_A:
            assert nb_feats and not (unlab or b_pseudo_unlab or a_extra_hist), "nb_feats_into_A: unsupported combination"
            _tx_tr = L_tx[Xtr.index.to_numpy()]
            _tx_va = T_tx[Xva.index.to_numpy()] if _PHASE == "final" else L_tx[Xva.index.to_numpy()]
            _nbf = [c for c in IMP_ORDER if c in Xtr.columns][:nb_extra_k or nb_feats]
            _v_tr = pd.DataFrame(Xtr[_nbf].to_numpy(), index=_tx_tr, columns=_nbf)[win]
            _v_va = pd.DataFrame(Xva[_nbf].to_numpy(), index=_tx_va, columns=_nbf)
            Xa_tr = pd.concat([Xtr.reset_index(drop=True), feat_agg(_tx_tr, _v_tr.index, _v_tr)], axis=1).set_index(Xtr.index)
            Xa_va = pd.concat([Xva.reset_index(drop=True), feat_agg(_tx_va, _v_va.index, _v_va)], axis=1).set_index(Xva.index)
        if a_cols == "local":
            # step 11 round 9 (2026-10-17): model A sees the local block only, so the six neighbour numbers come
            # from a different scorer while model B keeps all columns (a blend member with different inputs)
            loc_a = [c for c in Xa_tr.columns if c in set(d.LOCAL)]
            Xa_tr, Xa_va = Xa_tr[loc_a], Xa_va[loc_a]
        if a_oof in ("random", "group", "step"):
            # 5-fold random out-of-fold scoring within the window: each row scored by a model that
            # saw the other 4/5 of the window (same periods), the way test rows are scored by a
            # model that knows their period
            from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
            idx_w = np.flatnonzero(win)
            fold_models = []
            # optional: rows from the a_extra_hist steps just before the window are added to every
            # fold scorer's training data (never scored themselves): stronger scorers, same targets
            extra = np.flatnonzero((~win) & (t >= t.max() - (last_k - 1) - a_extra_hist)) if a_extra_hist else np.array([], dtype=int)
            if a_oof == "group":
                # step 11 round 3 (plan_step11.md): under random folds the scorer of neighbour j saw
                # row i (label included) 80% of the time, so i's neighbour numbers carry a dose of
                # its own label (nb_out_mean AUC 0.902 random vs 0.889 grouped, 2026-10-13); test
                # scorers never saw a test label. Folds by connected component of the labelled
                # window subgraph: no scorer of j has seen any neighbour of j. Same period kept.
                import scipy.sparse as sp
                from scipy.sparse.csgraph import connected_components
                tx_w = L_tx[Xtr.index.to_numpy()[idx_w]]; pos = pd.Series(np.arange(len(idx_w)), index=tx_w)
                e_w = edges[edges.txId1.isin(pos.index) & edges.txId2.isin(pos.index)]
                adj = sp.coo_matrix((np.ones(len(e_w)), (pos[e_w.txId1].to_numpy(), pos[e_w.txId2].to_numpy())), shape=(len(idx_w),) * 2)
                groups = connected_components(adj, directed=False)[1]
                splits = StratifiedGroupKFold(oof_folds, shuffle=True, random_state=seed).split(idx_w, ytr[idx_w], groups)
            elif a_oof == "step":
                # step 11 round 5: leave-one-step-out within the window — each step scored by a model on
                # the other 13 window steps (before and after it): like the test scorer it never saw the
                # row's period and has ~the same data; no leak, since edges never cross steps
                tw = t[idx_w]
                splits = [(np.flatnonzero(tw != s_), np.flatnonzero(tw == s_)) for s_ in np.unique(tw)]
            else:
                splits = StratifiedKFold(oof_folds, shuffle=True, random_state=seed).split(idx_w, ytr[idx_w])
            for tr_i, va_i in splits:
                a_i, b_i = np.r_[idx_w[tr_i], extra], idx_w[va_i]
                m_f = model(seed).fit(Xa_tr.iloc[a_i], ytr[a_i]); fold_models.append(m_f)
                pA[b_i] = m_f.predict_proba(Xa_tr.iloc[b_i])[:, 1]
        else:
            for s_ in np.unique(t[win]):
                hist = t < s_
                if hist.sum() < 500: continue
                mA_s = model(seed).fit(Xa_tr[hist], ytr[hist], sample_weight=wts(hist, s_ - 1))
                pA[t == s_] = mA_s.predict_proba(Xa_tr[t == s_])[:, 1]
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
        if a_test_from_folds and a_oof in ("random", "group", "step"):
            # test / validation rows scored by the average of the OOF fold models, so they are
            # scored by the same kind of model (trained on (k-1)/k of the window) as the training rows
            pA_va = np.mean([m_f.predict_proba(Xa_va)[:, 1] for m_f in fold_models], axis=0)
        else:
            mA = model(seed).fit(Xa_tr, ytr) if a_all_history else model(seed).fit(Xa_tr[win], ytr[win], sample_weight=wts(win, t.max()))
            pA_va = mA.predict_proba(Xa_va)[:, 1]
        tx_va = T_tx[Xva.index.to_numpy()] if _PHASE == "final" else L_tx[Xva.index.to_numpy()]
        # step 7 (plan_step7.md): model-A scores on a period-free scale before aggregation, and/or
        # the period level handed to model B as its own column
        T_ts = d.test["time_step"].to_numpy(); ts_va = T_ts[Xva.index.to_numpy()] if _PHASE == "final" else t_all[Xva.index.to_numpy()]
        sc_va = pd.Series(pA_va, index=tx_va)
        if nb_steprank:
            sc_tr = pd.Series(pd.Series(sc_tr.to_numpy()).groupby(t_all_by_tx.reindex(sc_tr.index).to_numpy()).rank(pct=True).to_numpy(), index=sc_tr.index)
            sc_va = pd.Series(pd.Series(pA_va).groupby(ts_va).rank(pct=True).to_numpy(), index=tx_va)
        Ftr = neighbour_feats(tx_tr, sc_tr); Fva = neighbour_feats(tx_va, sc_va)
        if nb_feats:
            nbf = [c for c in IMP_ORDER if c in Xtr.columns][:nb_feats]
            vals_tr = pd.DataFrame(Xtr[nbf].to_numpy(), index=tx_tr, columns=nbf)
            vals_tr = vals_tr[vals_tr.index.isin(sc_tr.index)]            # labelled scored window rows only
            vals_va = pd.DataFrame(Xva[nbf].to_numpy(), index=tx_va, columns=nbf)
            Ftr = pd.concat([Ftr, feat_agg(tx_tr, sc_tr.index, vals_tr)], axis=1)
            Fva = pd.concat([Fva, feat_agg(tx_va, sc_va.index, vals_va)], axis=1)
            if nb_feat_hop2:
                # two-hop feature means: for every scored node its in-/out-neighbour feature means,
                # then averaged again over the row's in-/out-neighbours (features of neighbours' neighbours)
                for F_, vals_, tx_, sc_ in ((Ftr, vals_tr, tx_tr, sc_tr), (Fva, vals_va, tx_va, sc_va)):
                    full = feat_agg(vals_.index.to_numpy(), sc_.index, vals_).set_index(vals_.index)
                    for side in ("in", "out"):
                        cols = [f"nbf_{side}_mean_{c}" for c in nbf]
                        h = feat_agg(tx_, sc_.index, full[cols])          # mean over neighbours of their own side-means
                        for c in nbf:
                            F_[f"nbf_h2_{side}_{c}"] = h[f"nbf_{side}_mean_nbf_{side}_mean_{c}"].to_numpy()
            # step 11 round 4 (combined candidate): the extras below may use only the first
            # nb_extra_k features (step-1 gain order) while the means cover all nb_feats
            nbx = nbf[:nb_extra_k] if nb_extra_k else nbf
            vx_tr, vx_va = vals_tr[nbx], vals_va[nbx]
            if nb_extra_stats:
                Ftr = pd.concat([Ftr, feat_agg(tx_tr, sc_tr.index, vx_tr, nb_extra_stats)], axis=1)
                Fva = pd.concat([Fva, feat_agg(tx_va, sc_va.index, vx_va, nb_extra_stats)], axis=1)
            if nb_feat_delta:
                # the row's own value minus its neighbours' mean: a difference trees cannot form from
                # two columns with a few splits (blank where there is no neighbour)
                for side in ("in", "out"):
                    for c in nbx:
                        Ftr[f"nbf_{side}_delta_{c}"] = Xtr[c].to_numpy() - Ftr[f"nbf_{side}_mean_{c}"].to_numpy()
                        Fva[f"nbf_{side}_delta_{c}"] = Xva[c].to_numpy() - Fva[f"nbf_{side}_mean_{c}"].to_numpy()
            if nb_feat_scorew:
                Ftr = pd.concat([Ftr, feat_agg_scorew(tx_tr, sc_tr, vx_tr)], axis=1)
                Fva = pd.concat([Fva, feat_agg_scorew(tx_va, sc_va, vx_va)], axis=1)
            if nb_feat_top1:
                Ftr = pd.concat([Ftr, feat_agg_top1(tx_tr, sc_tr, vx_tr)], axis=1)
                Fva = pd.concat([Fva, feat_agg_top1(tx_va, sc_va, vx_va)], axis=1)
        if nb_comm:
            comm = communities(edges, pd.concat([d.train.set_index("txId")["time_step"], d.test.set_index("txId")["time_step"]]))

            def comm_feats(tx_rows: np.ndarray, sc: pd.Series) -> pd.DataFrame:
                """mean / max / count of `sc` over the other scored rows of each row's community."""
                c_sc = comm.reindex(sc.index).to_numpy()
                g = pd.DataFrame(dict(c=c_sc, v=sc.to_numpy())).dropna().groupby("c").v
                top2 = g.nlargest(2).groupby(level=0).agg(["max", "min", "size"])    # largest and second-largest
                stats = pd.concat([g.agg(["sum", "size"]), top2.rename(columns={"max": "m1", "min": "m2", "size": "nt"})], axis=1)
                c_rows = comm.reindex(tx_rows).to_numpy()
                st = stats.reindex(c_rows)
                own = sc.reindex(tx_rows).to_numpy()
                in_sc = ~np.isnan(own)
                n_other = st["size"].fillna(0).to_numpy() - in_sc
                mean_other = (st["sum"].to_numpy() - np.where(in_sc, own, 0)) / np.maximum(n_other, 1)
                # max over the others: the community max unless the row alone holds it
                max_other = np.where(in_sc & (own >= st["m1"].to_numpy()) & (st["nt"].to_numpy() >= 2), st["m2"].to_numpy(), st["m1"].to_numpy())
                blank = n_other < 1
                return pd.DataFrame({"comm_mean": np.where(blank, np.nan, mean_other), "comm_max": np.where(blank, np.nan, max_other),
                                     "comm_n": n_other})
            Ftr = pd.concat([Ftr, comm_feats(tx_tr, sc_tr)], axis=1); Fva = pd.concat([Fva, comm_feats(tx_va, sc_va)], axis=1)
        if b_gnn_embed:
            # step 9: strictly out-of-fold GraphSAGE embeddings (+ its probability) as extra model-B
            # columns, from artifacts/gnn_cache (src/gnn_cache_window.py, a PyTorch-only process):
            # window rows 5-fold OOF inside the window, apply rows from a model on the whole window
            vlo, vhi = (36, 49) if _PHASE == "final" else (int(ts_va.min()), int(ts_va.max()))
            z = np.load(ARTIFACTS / "gnn_cache" / f"win{int(t.max()) - (last_k - 1)}-{int(t.max())}_apply{vlo}-{vhi}_s{seed}.npz")
            H = z["h_oof"].shape[1]; cols = ["gnn_p"] + [f"gnn_h{i}" for i in range(H)]
            G_tr = pd.DataFrame(np.c_[z["p_oof"], z["h_oof"]], index=z["tx_oof"], columns=cols).reindex(tx_tr).reset_index(drop=True)
            G_va = pd.DataFrame(np.c_[z["p_apply"], z["h_apply"]], index=z["tx_apply"], columns=cols).reindex(tx_va).reset_index(drop=True)
            assert G_va.notna().all().all() and G_tr.iloc[np.flatnonzero(win)].notna().all().all(), "GNN cache does not cover these rows"
            Ftr = pd.concat([Ftr, G_tr], axis=1); Fva = pd.concat([Fva, G_va], axis=1)
        if step_ctx:
            ctx_tr = pd.Series(pA).groupby(t).transform("mean").to_numpy()          # NaN outside the window (unscored rows)
            ctx_va = pd.Series(pA_va).groupby(ts_va).transform("mean").to_numpy()
            Ftr = Ftr.assign(step_ctx=ctx_tr); Fva = Fva.assign(step_ctx=ctx_va)
        Xtr_b, Xva_b = (interact(Xtr), interact(Xva)) if b_interact else (Xtr.reset_index(drop=True), Xva.reset_index(drop=True))
        if TOPK:
            Xtr_b, Xva_b = Xtr_b[TOPK], Xva_b[TOPK]
        A = pd.concat([Xtr_b, Ftr], axis=1)[win]
        B = pd.concat([Xva_b, Fva], axis=1)
        yB, tB, wB = ytr[win], t[win], wts(win, t.max())
        if b_step_balance:
            # step 9 (plan_step9.md): every time step of the window carries the same total weight, so
            # the large steps (22: 1,763 rows) stop dominating the small ones (27: 206 rows)
            cnt = pd.Series(tB).map(pd.Series(tB).value_counts()).to_numpy()
            wB = len(tB) / (np.unique(tB).size * cnt)
        if b_step_unbal:
            # round 24 (2026-10-22): per-step class balance. Each illicit row is weighted by its own step's
            # licit/illicit ratio (2.3-21 in the window, vs 5.6 for one global is_unbalance), so fraud in low-rate
            # steps (23, 33, 34, 22), the most like the test's late block, counts most; licit rows keep weight 1
            yv = np.asarray(yB, float)
            rate = pd.Series(yv).groupby(np.asarray(tB)).transform("mean").to_numpy()
            wcls = np.where(yv == 1, (1 - rate) / rate, 1.0)
            wB = wcls if wB is None else np.asarray(wB, float) * wcls
        if b_dro is not None:
            # step 12 (docs/plans/plan_step12_dro.md): Group DRO over the window steps. Each round scores the
            # window rows out-of-fold with model B (folds grouped by connected component, as a_oof="group"),
            # takes each step's risk 1 - AUC, shrinks it toward the mean by its Hanley-McNeil standard error
            # (empirical Bayes), and multiplies the step's row weight by exp(eta * standardised risk).
            # Row-weight ratio capped at b_dro_rho, ESS kept >= b_dro_ess; b_dro=0.0 runs the loop with no
            # update (wiring check: must equal the unweighted model).
            assert a_oof == "group" and not (b_pseudo_unlab or b_pseudo_pre43 or b_pseudo_post43_neg or b_stack_logreg or b_rank), \
                "b_dro: needs a_oof='group' and no extra model-B rows / columns"
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import StratifiedGroupKFold as _SGKF
            steps_ = np.unique(tB); g_ix = np.searchsorted(steps_, tB)
            n_g = np.bincount(g_ix); p_g = np.bincount(g_ix, weights=yB).astype(int)
            w0 = np.ones(len(yB)) if wB is None else np.asarray(wB, float)
            mult = np.ones(len(steps_))
            if b_dro_risk == "step":
                # plan_step12_dro.md §11: leave-one-step-out — each window step scored by model B trained on the
                # other 13 steps, so a step's risk measures transfer to an unseen period (in-window grouped folds
                # scored every step ~0.98-1.0 and found no between-step spread beyond noise)
                folds_ = [(np.flatnonzero(g_ix != k), np.flatnonzero(g_ix == k)) for k in range(len(steps_))]
            else:
                folds_ = list(_SGKF(5, shuffle=True, random_state=seed).split(np.zeros(len(yB)), yB, groups))

            def _ess(w):
                return w.sum() ** 2 / (w ** 2).sum() / len(w)

            for r_ in range(b_dro_rounds + 1):
                w_cur = w0 * mult[g_ix]
                oof = np.full(len(yB), np.nan)
                for tr_i, va_i in folds_:
                    oof[va_i] = fit_B(A.iloc[tr_i], yB[tr_i], tB[tr_i], w_cur[tr_i], seed).predict_proba(A.iloc[va_i])[:, 1]
                auc_g = np.array([roc_auc_score(yB[g_ix == k], oof[g_ix == k]) if 0 < p_g[k] < n_g[k] else np.nan
                                  for k in range(len(steps_))])
                a_ = np.clip(np.nan_to_num(auc_g, nan=np.nanmean(auc_g)), 1e-3, 1 - 1e-3)
                q1, q2 = a_ / (2 - a_), 2 * a_ ** 2 / (1 + a_)
                n_neg = np.maximum(n_g - p_g, 1); n_pos = np.maximum(p_g, 1)
                se = np.sqrt((a_ * (1 - a_) + (n_pos - 1) * (q1 - a_ ** 2) + (n_neg - 1) * (q2 - a_ ** 2)) / (n_pos * n_neg))
                R = 1 - a_; R_bar = R.mean()
                tau2 = max(0.0, R.var() - (se ** 2).mean())
                R_s = R_bar + tau2 / (tau2 + se ** 2) * (R - R_bar) if tau2 > 0 else np.full_like(R, R_bar)
                _DRO_LOG.extend(dict(phase=_PHASE, seed=seed, t_max=int(t.max()), round=r_, step=int(s_), n=int(n_g[k]),
                                     pos=int(p_g[k]), auc=auc_g[k], se=se[k], risk_shrunk=R_s[k], mult=mult[k],
                                     ess=_ess(w_cur), tau2=tau2) for k, s_ in enumerate(steps_))
                print(f"    DRO[{b_dro_risk}] {_PHASE} seed {seed} round {r_}: pooled OOF AUC {roc_auc_score(yB, oof):.4f}, "
                      f"worst step {steps_[np.nanargmin(auc_g)]} ({np.nanmin(auc_g):.3f}), tau2 {tau2:.2e}, "
                      f"max/min weight {mult.max() / mult.min():.2f}, ESS {_ess(w_cur):.3f}", flush=True)
                if r_ == b_dro_rounds or tau2 == 0 or not b_dro:
                    break                                      # last round measured only; no update past it
                z = (R_s - R_s.mean()) / R_s.std()
                new = mult * np.exp(b_dro * z)
                lg = np.log(new); mid = (lg.max() + lg.min()) / 2
                new = np.exp(np.clip(lg - mid, -np.log(b_dro_rho) / 2, np.log(b_dro_rho) / 2))
                new /= np.average(new, weights=n_g)            # mean row weight 1
                if _ess(w0 * new[g_ix]) < b_dro_ess:
                    print(f"    DRO: round {r_ + 1} would take ESS below {b_dro_ess}: stopped", flush=True)
                    break
                mult = new
            wB = w0 * mult[g_ix]
        if b_stack_logreg:
            # one extra column for model B: a logistic-regression score on the raw features,
            # random OOF within the window for training rows, full-window fit for the rows to predict
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler, FunctionTransformer
            from sklearn.model_selection import StratifiedKFold as _SKF
            def lr():
                if b_stack_model == "mlp":
                    from sklearn.neural_network import MLPClassifier
                    head = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=200, early_stopping=True, random_state=seed)
                elif b_stack_model == "knn":
                    from sklearn.neighbors import KNeighborsClassifier
                    head = KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1)
                else:
                    head = LogisticRegression(C=0.1, max_iter=2000)
                return make_pipeline(StandardScaler(), FunctionTransformer(lambda z: np.clip(z, -10, 10)), head)
            Xw = Xtr.iloc[np.flatnonzero(win)]; yw = ytr[win]; s_tr = np.zeros(len(Xw))
            for f_i, g_i in _SKF(5, shuffle=True, random_state=seed).split(Xw, yw):
                s_tr[g_i] = lr().fit(Xw.iloc[f_i], yw[f_i]).predict_proba(Xw.iloc[g_i])[:, 1]
            A = A.assign(lr_score=s_tr); B = B.assign(lr_score=lr().fit(Xw, yw).predict_proba(Xva)[:, 1])
        if b_pseudo_unlab:
            # unlabelled rows of the window as extra model-B training rows, with hard pseudo-labels
            # from the fold-average scorer (confident rows only); they are not used as neighbours
            m_u = (U_ts >= t.max() - (last_k - 1)) & (U_ts <= t.max())
            XU = U[Xtr.columns][m_u]; pu = np.mean([m_f.predict_proba(XU)[:, 1] for m_f in fold_models], axis=0)
            keep_u = (pu > 0.9) | (pu < 0.02)
            FU = neighbour_feats(U_tx[m_u][keep_u], sc_tr)      # neighbour numbers from the labelled scored rows
            AU = pd.concat([XU[keep_u].reset_index(drop=True), FU], axis=1)
            if b_stack_logreg: AU = AU.assign(lr_score=lr().fit(Xw, yw).predict_proba(XU[keep_u])[:, 1])
            A = pd.concat([A, AU[A.columns]], ignore_index=True); yB = np.r_[yB, (pu[keep_u] > 0.5).astype(int)]
            tB = np.r_[tB, U_ts[m_u][keep_u]]; wB = None if wB is None else np.r_[wB, np.ones(keep_u.sum())]
        if b_pseudo_pre43:
            # step 10 (plan_step10.md, 2026-10-12): the rows to predict from the first `b_pseudo_pre43` steps of
            # the prediction period (test: 36-42, where the model's ordering is good and both classes are
            # confidently predicted) join model B's training set with hard pseudo-labels from model A when it is
            # confident (> 0.9 or < 0.02). Transductive: their features only, never a label. Their model-B
            # columns are the ones already built for them (B), so column order is untouched. The recency
            # principle pushed to its limit: training rows from inside the test period. Post-43 rows are
            # never pseudo-labelled (their confident predictions are all licit: plan_step8 diagnostics).
            early = ts_va <= ts_va.min() + (b_pseudo_pre43 - 1)
            keep_p = early & ((pA_va > 0.9) | (pA_va < 0.02))
            A = pd.concat([A, B[keep_p]], ignore_index=True); yB = np.r_[yB, (pA_va[keep_p] > 0.5).astype(int)]
            tB = np.r_[tB, ts_va[keep_p]]; wB = None if wB is None else np.r_[wB, np.ones(keep_p.sum())]
        if b_pseudo_post43_neg:
            # step 11 (plan_step11.md, 2026-10-13): the confidently licit-scored post-43 rows join model B's
            # training set as pseudo-label-0 rows (their model-B columns are the ones already built in B),
            # teaching model B what post-43 licit looks like so the block's residual order sharpens. In CV
            # no validation step reaches 43, so the flag is a no-op there (CV = the parent's CV). Read
            # through the floorsplice only (the full file perturbs model B too little to pass the gate).
            keep_n = (ts_va >= 43) & (pA_va < 0.02)
            A = pd.concat([A, B[keep_n]], ignore_index=True); yB = np.r_[yB, np.zeros(int(keep_n.sum()), int)]
            tB = np.r_[tB, ts_va[keep_n]]; wB = None if wB is None else np.r_[wB, np.ones(keep_n.sum())]
        if b_nbonly:
            # round 15 (plan_step11.md): model B sees the neighbour columns only (6 score numbers + neighbour feature
            # means), no raw features of the row itself: the one column view the blocks member never fits alone
            nbc = [c for c in A.columns if c not in set(Xtr_b.columns)]
            A, B = A[nbc], B[nbc]
        if b_blocks_inout:
            # round 16 (plan_step11.md): block ensemble split by money direction — model B fitted on all columns, on the
            # raw columns + the in-neighbour columns (who paid into the row), and on the raw columns + the out-neighbour
            # columns (whom it paid); probabilities averaged. Column order inside each block is kept.
            raw_b = set(Xtr_b.columns)
            side = lambda c, d: c.startswith((f"nb_{d}_", f"nbf_{d}_"))
            blocks = [list(A.columns), [c for c in A.columns if c in raw_b or side(c, "in")],
                      [c for c in A.columns if c in raw_b or side(c, "out")]]
            return np.mean([fit_B(A[cols], yB, tB, wB, seed).predict_proba(B[cols])[:, 1] for cols in blocks], axis=0)
        if b_blocks:
            # step 11 round 6 (2026-10-15): feature-block ensemble — model B fitted three times, on all
            # columns, on the local raw block + the neighbour columns, and on the aggregated raw block + the
            # neighbour columns; probabilities averaged. Column order inside each block is kept.
            loc = set(d.LOCAL); raw_b = list(Xtr_b.columns); nb_b = [c for c in A.columns if c not in set(raw_b)]
            blocks = [list(A.columns), [c for c in raw_b if c in loc] + nb_b, [c for c in raw_b if c not in loc] + nb_b]
            return np.mean([fit_B(A[cols], yB, tB, wB, seed).predict_proba(B[cols])[:, 1] for cols in blocks], axis=0)
        if b_rsub:
            # step 11 round 9: random-subspace model B — b_rsub fits, each on a random b_rsub_frac of the raw columns
            # (original order kept) + every neighbour column, averaged; the subsets are drawn from the seed
            raw_b = list(Xtr_b.columns); nb_b = [c for c in A.columns if c not in set(raw_b)]
            rng = np.random.default_rng(1000 + seed); k_raw = int(round(b_rsub_frac * len(raw_b)))
            subsets = [[c for c in raw_b if c in set(rng.choice(raw_b, k_raw, replace=False))] + nb_b for _ in range(b_rsub)]
            return np.mean([fit_B(A[cols], yB, tB, wB, seed).predict_proba(B[cols])[:, 1] for cols in subsets], axis=0)
        mB = fit_B(A, yB, tB, wB, seed)
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
# step 11 round 10 (2026-10-18): drift-aware column removal (dropped from the raw block and, through
# Xtr.columns, from the neighbour-mean block too). ADV = single-feature adversarial AUC train vs test
# > 0.85 after the six period identifiers (artifacts/single_feature_adversarial_auc_raw.csv; label-free).
# UNSTABLE = pooled window AUC (steps 22-35) at least 0.05 from 0.5 but per-step direction opposite to
# the pooled one in >= 3 of the 14 window steps (selected on window labels: its CV is optimistic).
DRIFT_ADV = ["feat_2", "feat_106", "feat_107", "feat_109", "feat_142"]
DRIFT_UNSTABLE = ["feat_22", "feat_28", "feat_3", "feat_82", "feat_20", "feat_65", "feat_59", "feat_21", "feat_112",
                  "feat_93", "feat_92", "feat_80", "feat_81", "feat_118", "feat_67", "feat_61", "feat_134", "feat_77",
                  "feat_135", "feat_86", "feat_1", "feat_87", "feat_113"]
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
    # step 6 (docs/plans/plan_step6.md): one change each on B15_Aoof_random
    "B15_Aoof10": ("B15 two-stage, training rows scored by 10-fold random OOF within the window, bagged x5", "clean",
                   two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", oof_folds=10), BAG5),
    "B7_Aoof_random": ("two-stage with random 5-fold OOF scoring and model B = 7 leaves, 600 rounds (depth re-check under the new protocol), bagged x5", "clean",
                       two_stage(14, b_params={**B15_PARAMS, "num_leaves": 7}, b_estimators=600, a_oof="random"), BAG5),
    "recent21_Aoof_random": ("B15 two-stage with random 5-fold OOF scoring on the 21-step window (15-35), bagged x5", "clean",
                             two_stage(21, b_params=B15_PARAMS, b_estimators=300, a_oof="random"), BAG5),
    "B15_Aoof_foldavg": ("B15 two-stage, random 5-fold OOF, test rows scored by the average of the 5 fold models (same scorer kind as training rows), bagged x5", "clean",
                         two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", a_test_from_folds=True), BAG5),
    "recent28_Aoof_random": ("B15 two-stage with random 5-fold OOF scoring on the 28-step window (8-35), bagged x5", "clean",
                             two_stage(28, b_params=B15_PARAMS, b_estimators=300, a_oof="random"), BAG5),
    "recent10_Aoof_random": ("B15 two-stage with random 5-fold OOF scoring on the 10-step window (26-35), bagged x5", "clean",
                             two_stage(10, b_params=B15_PARAMS, b_estimators=300, a_oof="random"), BAG5),
    # step 7: new mechanisms on B15_Aoof_random
    "B15_Aoof_hist21": ("B15 two-stage, random OOF, fold scorers also trained on steps 15-21 (outside the window, never scored), bagged x5", "clean",
                        two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", a_extra_hist=7), BAG5),
    "B15_stack_logreg": ("B15 two-stage, random OOF, + a logistic-regression score as one extra model-B column, bagged x5", "clean",
                         two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", b_stack_logreg=True), BAG5),
    "B15_pseudo_unlab": ("B15 two-stage, random OOF, + confidently pseudo-labelled unlabelled window rows as extra model-B training rows, bagged x5", "clean",
                         two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", b_pseudo_unlab=True), BAG5),
    "B15_stack_mlp": ("B15 two-stage, random OOF, + an MLP score (128-64, early stopping) as one extra model-B column, bagged x5", "clean",
                      two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", b_stack_logreg=True, b_stack_model="mlp"), BAG5),
    "B15_stack_knn": ("B15 two-stage, random OOF, + a 50-NN distance-weighted score as one extra model-B column, bagged x5", "clean",
                      two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", b_stack_logreg=True, b_stack_model="knn"), BAG5),
    # step 7 (docs/plans/plan_step7.md): the input-scale axis
    "B15_nb_steprank": ("B15 two-stage, random OOF, model-A scores replaced by their within-step percentile rank before neighbour aggregation, bagged x5", "clean",
                        two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_steprank=True), BAG5),
    "B15_step_ctx": ("B15 two-stage, random OOF, + one model-B column: mean model-A score of the row's time step, bagged x5", "clean",
                     two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", step_ctx=True), BAG5),
    "B15_nb_steprank_ctx": ("B15 two-stage, random OOF, within-step-ranked neighbour scores + the step-context column, bagged x5", "clean",
                            two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_steprank=True, step_ctx=True), BAG5),
    # step 8 (docs/plans/plan_step8.md): neighbour feature aggregates inside the two-stage
    "B15_nbfeat": ("B15 two-stage, random OOF, + means of the top-10 raw features over labelled in- and out-neighbours (20 extra model-B columns), bagged x5", "clean",
                   two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10), BAG5),
    # read 2026-10-08: B15_nbfeat 0.96069 (+0.0022, new best) -> map the strength of the axis, one change each
    "B15_nbfeat20": ("B15 two-stage, random OOF, + means of the top-20 raw features over labelled in- and out-neighbours (40 extra model-B columns), bagged x5", "clean",
                     two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=20), BAG5),
    "B15_nbfeat_max": ("B15 two-stage, random OOF, + mean and max of the top-10 raw features over labelled in- and out-neighbours (40 extra model-B columns), bagged x5", "clean",
                       two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10, nb_feat_stats=("mean", "max")), BAG5),
    # gated 2026-10-08: nbfeat20 0.989 and nbfeat_max 0.994 like B15_nbfeat (unreadable) -> change the kind of information, not the amount
    "B15_nbfeat_delta": ("B15 two-stage, random OOF, + top-10 neighbour feature means and the row's own value minus each mean (40 extra model-B columns), bagged x5", "clean",
                         two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10, nb_feat_delta=True), BAG5),
    "B15_nbfeat_all": ("B15 two-stage, random OOF, + means of all 159 raw features over labelled in- and out-neighbours (318 extra model-B columns), bagged x5", "clean",
                       two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=159), BAG5),
    # step 9 (docs/plans/plan_step9.md): read 2026-10-08 nbfeat_all 0.96177 (+0.0011, tie) -> the neighbour feature means also feed model A
    "B15_nbfeat_A": ("B15 two-stage, random OOF, top-10 neighbour feature means fed to model A as well as model B, bagged x5", "clean",
                     two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10, nb_feats_into_A=True), BAG5),
    # gated 2026-10-08: nbfeat_A 0.991 (unreadable) -> candidate 3: features of the neighbours' neighbours
    "B15_nbfeat_hop2": ("B15 two-stage, random OOF, top-10 neighbour feature means + two-hop feature means (20 more model-B columns), bagged x5", "clean",
                        two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10, nb_feat_hop2=True), BAG5),
    # step 9, approved 2026-10-08: seed-noise reading of the leader; closed setting axes re-read once under the larger input
    "B15_nbfeat_all_seedsB": ("B15 two-stage, random OOF, means of all 159 features over labelled neighbours, seeds 5-9 (replicate of the 0.96177 file: seed noise of the leader)", "clean",
                              two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=159), (5, 6, 7, 8, 9)),
    # step 10 (docs/plans/plan_step10.md): variance reduction for the finals — more seeds of the two finals models
    "B15_nbfeat_all_seedsC": ("B15 two-stage, random OOF, means of all 159 features over labelled neighbours, seeds 10-14 (third 5-seed bag for the 15-seed finals file)", "clean",
                              two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=159), (10, 11, 12, 13, 14)),
    "B15_nbfeat_pseudo_pre43": ("B15 two-stage, random OOF, top-10 neighbour feature means, + the confidently scored test rows of steps 36-42 as pseudo-labelled model-B training rows (transductive; post-43 rows never pseudo-labelled), bagged x5", "clean",
                                two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10, b_pseudo_pre43=7), BAG5),
    "B15_nbfeat_all_seedsD": ("B15 two-stage, random OOF, means of all 159 features over labelled neighbours, seeds 15-29 (15 more seeds for the 30-seed finals file)", "clean",
                              two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=159), tuple(range(15, 30))),
    "B15_Aoof_random_seedsB": ("B15 two-stage with random 5-fold OOF scoring, seeds 5-9 (second 5-seed bag for the 10-seed Final 2)", "clean",
                               two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random"), (5, 6, 7, 8, 9)),
    "B15_nbfeat_stepbal": ("B15 two-stage, random OOF, top-10 neighbour feature means, model B with per-step balanced row weights (each window step carries equal total weight), bagged x5", "clean",
                           two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10, b_step_balance=True), BAG5),
    "B23_nbfeat": ("two-stage, random OOF, top-10 neighbour feature means, model B = 23 leaves (depth re-check under the larger input), bagged x5", "clean",
                   two_stage(14, b_params={**B15_PARAMS, "num_leaves": 23}, b_estimators=300, a_oof="random", nb_feats=10), BAG5),
    "B15_nbfeat_gnn": ("B15 two-stage, random OOF, top-10 neighbour feature means + strictly out-of-fold GraphSAGE embedding (64 dims + probability) as model-B columns, bagged x5", "clean",
                       two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10, b_gnn_embed=True), BAG5),
    # step 10 (docs/plans/plan_step10.md, Lecture 8 SNA): the community as the neighbourhood unit
    "B15_nbfeat_comm": ("B15 two-stage, random OOF, top-10 neighbour feature means + mean/max/count of model-A scores over the other scored rows of the row's Louvain community (3 model-B columns), bagged x5", "clean",
                        two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10, nb_comm=True), BAG5),
    # step 11, round 2 (user-approved 2026-10-13): two kind-changes on the closed neighbour-feature
    # axis, gate-only (a pass takes slot 2 behind the floorrev calibration; failures close the axis)
    "B15_nbfeat_scorew": ("B15 two-stage, random OOF, top-10 neighbour feature means + their score-weighted means (each neighbour's features weighted by its model-A score: the suspicious-neighbour profile, 20 extra model-B columns), bagged x5", "clean",
                          two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10, nb_feat_scorew=True), BAG5),
    "B15_nbfeat_top1": ("B15 two-stage, random OOF, top-10 neighbour feature means + the raw features of the single highest-scoring in- and out-neighbour (the most suspicious neighbour's profile, 20 extra model-B columns), bagged x5", "clean",
                        two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10, nb_feat_top1=True), BAG5),
    "B15_nbfeat_pseudo_p43neg": ("B15 two-stage, random OOF, top-10 neighbour feature means, + the confidently licit-scored post-43 test rows as pseudo-label-0 model-B training rows (transductive negatives; CV is a no-op for the flag; post-43 splice donor), bagged x5", "clean",
                                 two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10, b_pseudo_post43_neg=True), BAG5),
    # step 11, round 3 (user-approved 2026-10-13): the training-row scorer's label leak through
    # neighbours — OOF folds grouped by connected component of the labelled window subgraph
    "B15_nbfeat_goof": ("B15 two-stage, top-10 neighbour feature means, model-A OOF folds grouped by connected component of the labelled window subgraph (no scorer of a neighbour saw the row's label), bagged x5", "clean",
                        two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=10), BAG5),
    "B15_nbfeat_all_goof": ("B15 two-stage, means of all 159 features over labelled neighbours, model-A OOF folds grouped by connected component of the labelled window subgraph, bagged x5", "clean",
                            two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    # step 11 round 4 (2026-10-14): every CV-positive gate-failed change stacked into one model —
    # grouped OOF + neighbour means into model A + max / delta / argmax-neighbour profile of the
    # top-10 features + the step's mean model-A score; each alone was unreadable (0.988-0.994)
    "B15_nbfeat_combo": ("B15 two-stage, top-10 neighbour feature means, grouped OOF + top-10 means into model A + max, delta and argmax-neighbour profile of the top-10 + step-mean model-A score (all CV-positive single changes stacked), bagged x5", "clean",
                         two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=10, nb_feats_into_A=True,
                                   nb_extra_stats=("max",), nb_feat_delta=True, nb_feat_top1=True, step_ctx=True), BAG5),
    "B15_nbfeat_all_combo": ("B15 two-stage, means of all 159 features over labelled neighbours, grouped OOF + top-10 means into model A + max, delta and argmax-neighbour profile of the top-10 + step-mean model-A score, bagged x5", "clean",
                             two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=159, nb_extra_k=10, nb_feats_into_A=True,
                                       nb_extra_stats=("max",), nb_feat_delta=True, nb_feat_top1=True, step_ctx=True), BAG5),
    # step 11 round 5 (2026-10-14): leave-one-step-out OOF scoring of training rows (period-blind, data-matched)
    "B15_nbfeat_all_soof": ("B15 two-stage, means of all 159 features over labelled neighbours, model-A OOF leave-one-step-out within the window (scorer never saw the row's period, like the test scorer; no neighbour leak), bagged x5", "clean",
                            two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="step", nb_feats=159), BAG5),
    # step 11 round 6 (2026-10-15): one model-B change each on Final 1 (fam_B15_nbfeat_all_goof_bag15, 0.96310)
    "B15_nbfeat_all_goof_col10": ("B15 two-stage, all-159 neighbour means, grouped OOF, model B colsample_bytree 1.0 (every tree sees the six neighbour-score numbers), bagged x5", "clean",
                                  two_stage(14, b_params={**B15_PARAMS, "colsample_bytree": 1.0}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_dart": ("B15 two-stage, all-159 neighbour means, grouped OOF, model B with DART boosting (drop rate 0.1), bagged x5", "clean",
                                 two_stage(14, b_params={**B15_PARAMS, "boosting_type": "dart", "drop_rate": 0.1}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_blocks": ("B15 two-stage, all-159 neighbour means, grouped OOF, model B = average of three fits: all columns / local raw block + neighbour columns / aggregated raw block + neighbour columns, bagged x5", "clean",
                                   two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), BAG5),
    # step 11 round 7 (2026-10-16): learner diversity for the blend — XGBoost as model B on Final 1's inputs
    # (on the older base it ranked 0.944 like the LightGBM model B at -0.002 public: the block model's profile)
    "B15_nbfeat_all_goof_xgb": ("B15 two-stage inputs (all-159 neighbour means, grouped OOF), model B = XGBoost depth 4, 300 rounds, lr 0.05, 60% columns (blend member), bagged x5", "clean",
                                two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=159, b_learner="xgb"), BAG5),
    "B15_nbfeat_all_goof_xgb_seedsBC": ("B15_nbfeat_all_goof_xgb, seeds 5-14 (seed check and the 15-seed bag fam_B15_nbfeat_all_goof_xgb_bag15)", "clean",
                                        two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=159, b_learner="xgb"), tuple(range(5, 15))),
    # step 11 round 9 (2026-10-17): two new blend members, each read alone first (join if < 0.95 like every
    # member and >= 0.962 alone)
    "B15_nbfeat_all_goof_localA": ("Final 1 inputs, but model A trained on the local block (feat_1-93) only: different neighbour scores; model B = LightGBM on all columns, grouped OOF, bagged x5", "clean",
                                   two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=159, a_cols="local"), BAG5),
    "B15_nbfeat_all_goof_rsub": ("Final 1 inputs, model B = mean of 5 LightGBM fits, each on a random half of the 159 raw columns + all neighbour columns (random-subspace views), grouped OOF, bagged x5", "clean",
                                 two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=159, b_rsub=5), BAG5),
    # step 11 round 10 (2026-10-18, user-approved): drift-aware feature removal on the goof model (one change each)
    "B15_nbfeat_all_goof_dropadv": ("Final 1 member (goof) without the 5 drifting columns with adversarial AUC > 0.85 (feat_2, 106, 107, 109, 142), raw and neighbour means, bagged x5", "clean_noadv",
                                    two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_dropunstable": ("Final 1 member (goof) without the 23 columns whose per-step direction reverses in >= 3 of 14 window steps, raw and neighbour means, bagged x5", "clean_stable",
                                         two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    # step 11 round 11 (2026-10-18, user-approved): one model-B hyperparameter change each on goof, as blend-member
    # candidates (join rule of round 9: < 0.95 like each member and >= 0.962 alone)
    "B15_nbfeat_all_goof_xt": ("Final 1 member (goof) with model B = LightGBM extra_trees=True (random split thresholds), bagged x5", "clean",
                               two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    # step 11 round 12 (2026-10-18): xt read 0.96678 (+0.0047 vs goof 5-seed, highest public); build on it
    "B15_nbfeat_all_goof_xt_seedsBC": ("B15_nbfeat_all_goof_xt, seeds 5-14 (seed check of its CV and the 15-seed bag fam_B15_nbfeat_all_goof_xt_bag15)", "clean",
                                       two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_blocks_xt": ("block-ensemble model B (all / local+nb / aggregated+nb) with extra_trees=True: the blend partner rebuilt on the xt recipe, grouped OOF, bagged x5", "clean",
                                      two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), BAG5),
    "B15_nbfeat_all_goof_xt_r600": ("B15_nbfeat_all_goof_xt with 600 model-B rounds (random thresholds make each tree weaker), bagged x5", "clean",
                                    two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=600, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_xt_bin63": ("B15_nbfeat_all_goof_xt with max_bin 63 (coarser candidate thresholds on top of the random choice), bagged x5", "clean",
                                     two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "max_bin": 63}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_xt_L31": ("B15_nbfeat_all_goof_xt with 31 model-B leaves (random splits may move the depth optimum), bagged x5", "clean",
                                   two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "num_leaves": 31}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    # round 14 (2026-10-19, plan_step11.md): randomised blend members on the confirmed xt recipe (join if < 0.95 like
    # each member and >= 0.962 alone)
    "B15_nbfeat_all_goof_xt_rsub": ("xt with random-subspace model B: mean of 5 extra_trees fits, each on a random half of the 159 raw columns + all neighbour columns, bagged x5", "clean",
                                    two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_rsub=5), BAG5),
    "B15_nbfeat_all_goof_xt_col04": ("xt with model B colsample_bytree 0.4 (each tree sees 40% of the columns), bagged x5", "clean",
                                     two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "colsample_bytree": 0.4}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_blocks_xt_seedsBC": ("B15_nbfeat_all_goof_blocks_xt, seeds 5-14 (for the 15-seed bag fam_B15_nbfeat_all_goof_blocks_xt_bag15 and the xt + blocks_xt blend)", "clean",
                                              two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), tuple(range(5, 15))),
    # round 15 (2026-10-19, user-approved, plan_step11.md): third-member search for the xt + blocks_xt blend; joins only
    # if >= 0.9655 alone and < 0.95 like both members
    "B15_nbfeat_all_soof_xt": ("xt with leave-one-step-out model-A OOF (the soof protocol) + extra_trees model B, bagged x5", "clean",
                               two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="step", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_nbonly_xt": ("model B (extra_trees) on the neighbour columns only: 6 score numbers + 318 neighbour feature means, no raw features, grouped OOF, bagged x5", "clean",
                                      two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_nbonly=True), BAG5),
    "B15_nbfeat_all_goof_xtA": ("xt with extra_trees in model A as well (the neighbour scores change), grouped OOF, bagged x5", "clean",
                                two_stage(14, params={"extra_trees": True}, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    # round 16 (2026-10-19, user-approved, plan_step11.md): routes to 0.9700
    "B15_nbfeat_all_goof_xt_w21": ("xt with a 21-step training window (steps 15-35): does extra_trees change the window optimum?, bagged x5", "clean",
                                   two_stage(21, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_inout_xt": ("block ensemble by money direction with extra_trees: mean of model-B fits on all / raw + in-neighbour / raw + out-neighbour columns, grouped OOF, bagged x5", "clean",
                                     two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks_inout=True), BAG5),
    # round 17 (2026-10-20, user-approved): longer windows under extra_trees as a third blend member (21 steps tied xt,
    # so the long-window penalty is gone; more history may rank differently at equal strength). Joins if >= 0.9655 alone
    # and < 0.95 like both xt and blocks_xt
    "B15_nbfeat_all_goof_xt_w28": ("xt with a 28-step training window (steps 8-35), bagged x5", "clean",
                                   two_stage(28, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_xt_w21_seedsBC": ("B15_nbfeat_all_goof_xt_w21, seeds 5-14 (replicate of the 0.96755 reading, no shared seed; with it the 15-seed bag fam_B15_nbfeat_all_goof_xt_w21_bag15)", "clean",
                                           two_stage(21, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_xt_all": ("xt trained on all 35 training steps, bagged x5", "clean",
                                   two_stage(35, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    # round 19 (2026-10-21, user-approved): the untested side of the one xt knob the board read (600 rounds −0.0032,
    # 0.974 like xt). If random thresholds help by fitting the training period's cut-points less, fewer rounds is next
    "B15_nbfeat_all_goof_xt_r150": ("B15_nbfeat_all_goof_xt with 150 model-B rounds (less fitting: the other side of r600's −0.0032), bagged x5", "clean",
                                    two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=150, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_xt_r200": ("B15_nbfeat_all_goof_xt with 200 model-B rounds (midpoint between r150 and xt's 300), bagged x5", "clean",
                                    two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=200, a_oof="group", nb_feats=159), BAG5),
    # round 20 (2026-10-22, user-approved): the untested model-B knobs under extra_trees, gate-only (a slot only if
    # < 0.98 like xt and CV not clearly lower)
    "B15_nbfeat_all_goof_xt_sub08": ("xt with model-B row subsampling 0.8 per tree (more randomisation, as extra_trees), bagged x5", "clean",
                                     two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "subsample": 0.8}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_xt_unbal": ("xt with model-B is_unbalance=True (positive class up-weighted to balance: changes split choice), bagged x5", "clean",
                                     two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "is_unbalance": True}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_xt_lr025": ("xt with model-B lr 0.025 x 600 rounds (same total step as 0.05 x 300, smoother), bagged x5", "clean",
                                     two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "learning_rate": 0.025}, b_estimators=600, a_oof="group", nb_feats=159), BAG5),
    # round 21 (2026-10-22): fam_F1_xtwin (xt half = 14- + 21-step bags) read 0.96861; the blocks half gets the same
    # window average, so the blend is symmetric: (xt14 + xt21 + bxt14 + bxt21) / 4
    "B15_nbfeat_all_goof_blocks_xt_w21": ("blocks_xt with a 21-step training window (steps 15-35), bagged x5", "clean",
                                          two_stage(21, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), BAG5),
    "B15_nbfeat_all_goof_blocks_xt_w21_seedsBC": ("blocks_xt with a 21-step window, seeds 5-14 (for the 15-seed bag)", "clean",
                                                  two_stage(21, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), tuple(range(5, 15))),
    # round 22 (2026-10-22): xt_unbal read 0.96996 (+0.0029 vs the xt mean, 0.0001 under the replicate line):
    # replicate on seeds 5-14, then the blend partner rebuilt with is_unbalance
    "B15_nbfeat_all_goof_xt_unbal_seedsBC": ("xt_unbal, seeds 5-14 (replicate of the 0.96996 reading, no shared seed; with it the 15-seed bag)", "clean",
                                             two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "is_unbalance": True}, b_estimators=300, a_oof="group", nb_feats=159), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_blocks_xt_unbal": ("blocks_xt with model-B is_unbalance=True (the blend partner on the unbal recipe), bagged x5", "clean",
                                            two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "is_unbalance": True}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), BAG5),
    "B15_nbfeat_all_goof_blocks_xt_unbal_seedsBC": ("blocks_xt_unbal, seeds 5-14 (for the 15-seed bag)", "clean",
                                                    two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "is_unbalance": True}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_xt_spw3": ("xt with model-B scale_pos_weight 3 (about half of is_unbalance's balanced weight, 5.6 in the window: 15.1% positives), bagged x5", "clean",
                                    two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "scale_pos_weight": 3.0}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_xt_spw12": ("xt with model-B scale_pos_weight 12 (about twice is_unbalance's balanced 5.6: past the point that read 0.96996), bagged x5", "clean",
                                     two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "scale_pos_weight": 12.0}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    # round 23 (2026-10-22): fam_F1_unbal read 0.97020; the weight-12 blend is built ahead of xt_spw12's reading
    "B15_nbfeat_all_goof_xt_spw12_seedsBC": ("xt_spw12, seeds 5-14 (for the 15-seed bag)", "clean",
                                             two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "scale_pos_weight": 12.0}, b_estimators=300, a_oof="group", nb_feats=159), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_blocks_xt_spw12": ("blocks_xt with model-B scale_pos_weight 12, bagged x5", "clean",
                                            two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "scale_pos_weight": 12.0}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), BAG5),
    "B15_nbfeat_all_goof_blocks_xt_spw12_seedsBC": ("blocks_xt_spw12, seeds 5-14 (for the 15-seed bag)", "clean",
                                                    two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "scale_pos_weight": 12.0}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), tuple(range(5, 15))),
    # round 24 (2026-10-22, user-approved): the class-weighting mechanism aimed at the low-rate steps / at model A
    "B15_nbfeat_all_goof_xt_stepunbal": ("xt with per-step class balance in model B (each illicit row weighted by its step's licit/illicit ratio, 2.3-21; replaces the global is_unbalance 5.6), bagged x5", "clean",
                                         two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_step_unbal=True), BAG5),
    "B15_nbfeat_all_goof_xt_stepunbal_seedsBC": ("xt_stepunbal, seeds 5-14 (for the 15-seed bag)", "clean",
                                                 two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_step_unbal=True), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_blocks_xt_stepunbal": ("blocks_xt with per-step class balance in model B, bagged x5", "clean",
                                                two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_step_unbal=True, b_blocks=True), BAG5),
    "B15_nbfeat_all_goof_blocks_xt_stepunbal_seedsBC": ("blocks_xt_stepunbal, seeds 5-14 (for the 15-seed bag)", "clean",
                                                        two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_step_unbal=True, b_blocks=True), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_xtA_unbal": ("xt_unbal with is_unbalance in model A as well (the neighbour scores change), bagged x5", "clean",
                                      two_stage(14, params={"is_unbalance": True}, b_params={**B15_PARAMS, "extra_trees": True, "is_unbalance": True}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    # round 25 (2026-10-23): the six weighted members of fam_F1_wavg (0.97066) rebuilt on the 21-step window (14 and 21
    # read equal under extra_trees; window averaging added +0.0003 as fam_F1_xtwin) for a 12-member window x scheme blend
    "B15_nbfeat_all_goof_xt_unbal_w21": ("xt_unbal with a 21-step window (steps 15-35), seeds 0-4", "clean",
        two_stage(21, b_params={**B15_PARAMS, "extra_trees": True, "is_unbalance": True}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_xt_unbal_w21_seedsBC": ("xt_unbal with a 21-step window (steps 15-35), seeds 5-14", "clean",
        two_stage(21, b_params={**B15_PARAMS, "extra_trees": True, "is_unbalance": True}, b_estimators=300, a_oof="group", nb_feats=159), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_blocks_xt_unbal_w21": ("blocks_xt_unbal with a 21-step window (steps 15-35), seeds 0-4", "clean",
        two_stage(21, b_params={**B15_PARAMS, "extra_trees": True, "is_unbalance": True}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), BAG5),
    "B15_nbfeat_all_goof_blocks_xt_unbal_w21_seedsBC": ("blocks_xt_unbal with a 21-step window (steps 15-35), seeds 5-14", "clean",
        two_stage(21, b_params={**B15_PARAMS, "extra_trees": True, "is_unbalance": True}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_xt_spw12_w21": ("xt_spw12 with a 21-step window (steps 15-35), seeds 0-4", "clean",
        two_stage(21, b_params={**B15_PARAMS, "extra_trees": True, "scale_pos_weight": 12.0}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_xt_spw12_w21_seedsBC": ("xt_spw12 with a 21-step window (steps 15-35), seeds 5-14", "clean",
        two_stage(21, b_params={**B15_PARAMS, "extra_trees": True, "scale_pos_weight": 12.0}, b_estimators=300, a_oof="group", nb_feats=159), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_blocks_xt_spw12_w21": ("blocks_xt_spw12 with a 21-step window (steps 15-35), seeds 0-4", "clean",
        two_stage(21, b_params={**B15_PARAMS, "extra_trees": True, "scale_pos_weight": 12.0}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), BAG5),
    "B15_nbfeat_all_goof_blocks_xt_spw12_w21_seedsBC": ("blocks_xt_spw12 with a 21-step window (steps 15-35), seeds 5-14", "clean",
        two_stage(21, b_params={**B15_PARAMS, "extra_trees": True, "scale_pos_weight": 12.0}, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_xt_stepunbal_w21": ("xt_stepunbal with a 21-step window (steps 15-35), seeds 0-4", "clean",
        two_stage(21, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_step_unbal=True), BAG5),
    "B15_nbfeat_all_goof_xt_stepunbal_w21_seedsBC": ("xt_stepunbal with a 21-step window (steps 15-35), seeds 5-14", "clean",
        two_stage(21, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_step_unbal=True), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_blocks_xt_stepunbal_w21": ("blocks_xt_stepunbal with a 21-step window (steps 15-35), seeds 0-4", "clean",
        two_stage(21, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_step_unbal=True, b_blocks=True), BAG5),
    "B15_nbfeat_all_goof_blocks_xt_stepunbal_w21_seedsBC": ("blocks_xt_stepunbal with a 21-step window (steps 15-35), seeds 5-14", "clean",
        two_stage(21, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159, b_step_unbal=True, b_blocks=True), tuple(range(5, 15))),
    # round 26 (2026-10-24): class weighting on the XGBoost model B (unweighted: 0.96189, ranks 0.944 like the LightGBM
    # family): a third learner as a blend member if weighting lifts it to the members' level
    "B15_nbfeat_all_goof_xgb_unbal": ("XGBoost model B (depth 4) with scale_pos_weight = the window's licit/illicit ratio (5.6), grouped OOF, bagged x5", "clean",
                                      two_stage(14, b_params={**B15_PARAMS, "is_unbalance": True}, b_estimators=300, a_oof="group", nb_feats=159, b_learner="xgb"), BAG5),
    "B15_nbfeat_all_goof_xgb_unbal_seedsBC": ("xgb_unbal, seeds 5-14 (for the 15-seed bag)", "clean",
                                              two_stage(14, b_params={**B15_PARAMS, "is_unbalance": True}, b_estimators=300, a_oof="group", nb_feats=159, b_learner="xgb"), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_xt_mcs100": ("xt with model-B min_child_samples 100 (vs 40: larger leaves), bagged x5", "clean",
                                      two_stage(14, b_params={**B15_PARAMS, "extra_trees": True, "min_child_samples": 100}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    # step 12 (docs/plans/plan_step12_dro.md, user-approved 2026-10-18): Group DRO over window steps for model B
    "B15_nbfeat_all_goof_xt_dro": ("B15_nbfeat_all_goof_xt with Group DRO row weights for model B (worst-step reweighting: eta 1.0, 3 rounds, step risk 1 - OOF AUC shrunk by its s.e., weight ratio <= 8, ESS >= 0.65), bagged x5", "clean",
                                   two_stage(14, b_params={**B15_PARAMS, "extra_trees": True}, b_estimators=300, a_oof="group", nb_feats=159,
                                             b_dro=1.0, b_dro_rho=8.0, b_dro_ess=0.65, b_dro_rounds=3), BAG5),
    "B15_nbfeat_all_goof_reg": ("Final 1 member (goof) with model B strongly regularised (15 leaves, min 100 rows/leaf, 30% columns, lambda 5), bagged x5", "clean",
                                two_stage(14, b_params={**B15_PARAMS, "min_child_samples": 100, "colsample_bytree": 0.3, "reg_lambda": 5.0}, b_estimators=300, a_oof="group", nb_feats=159), BAG5),
    "B15_nbfeat_all_goof_blocks_seedsBC": ("B15_nbfeat_all_goof_blocks, seeds 5-14 (for the 15-seed bag fam_B15_nbfeat_all_goof_blocks_bag15)", "clean",
                                           two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=159, b_blocks=True), tuple(range(5, 15))),
    "B15_nbfeat_all_soof_seedsBC": ("B15_nbfeat_all_soof, seeds 5-14 (seed check of its CV and the 15-seed bag fam_B15_nbfeat_all_soof_bag15)", "clean",
                                    two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="step", nb_feats=159), tuple(range(5, 15))),
    "B15_nbfeat_all_combo_seedsBC": ("B15_nbfeat_all_combo, seeds 5-14 (for the 15-seed bag fam_B15_nbfeat_all_combo_bag15)", "clean",
                                     two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=159, nb_extra_k=10, nb_feats_into_A=True,
                                               nb_extra_stats=("max",), nb_feat_delta=True, nb_feat_top1=True, step_ctx=True), tuple(range(5, 15))),
    "B15_nbfeat_all_goof_seedsBC": ("B15 two-stage, means of all 159 features over labelled neighbours, grouped OOF, seeds 5-14 (for the 15-seed bag fam_B15_nbfeat_all_goof_bag15)", "clean",
                                    two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="group", nb_feats=159), tuple(range(5, 15))),
    # step 11 (docs/plans/plan_step11.md): post-43 splice donors. Not direct upload candidates
    # (a low-rate trainer has no high-rate competence, so the CV sanity gate does not apply);
    # only their post-43 block order is read, through post43_probes.py floorsplice.
    "lowrate": ("post-43 splice donor: 7-leaf LightGBM (lr 0.05, 300 rounds) trained on the low-fraud-rate steps only (1-6, 10, 12, 23: 176 positives, 0.4-4.5% per step, the regime most like post-43), bagged x10", "clean",
                lgbm_steps((1, 2, 3, 4, 5, 6, 10, 12, 23), {"num_leaves": 7, "min_child_samples": 20, "colsample_bytree": 0.6, "subsample": 1.0, "subsample_freq": 1, "reg_lambda": 1.0, "reg_alpha": 0.0, "learning_rate": 0.05}), tuple(range(10))),
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
        cols = {"clean": d.RAW_CLEAN, "all": d.RAW, "all_time": d.RAW + ["time_step"], "local": d.LOCAL,
                "clean_noadv": [c for c in d.RAW_CLEAN if c not in DRIFT_ADV],
                "clean_stable": [c for c in d.RAW_CLEAN if c not in DRIFT_UNSTABLE]}[fset]
        Xtr, Xte = d.L[cols], d.test[cols]
    y = d.y.to_numpy().astype(int)
    print(f"\n=== {name}: {desc}", flush=True)
    _DRO_LOG.clear()
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
    if _DRO_LOG:
        pd.DataFrame(_DRO_LOG).to_csv(ARTIFACTS / f"dro_{name}.csv", index=False)
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
