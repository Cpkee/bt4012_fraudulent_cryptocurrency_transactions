"""Probe factory for the leaderboard campaign (plan_step4.md).

Each probe is the reference pipeline (sub_07: v2b features, tuned LightGBM, 5-seed bag)
with exactly one component changed. For every probe this script reports the four
standard rolling folds and the eight far-horizon windows (the CV veto), writes
submissions/probe_<name>.csv from a fit on all labelled rows, and appends a ledger row.

    .venv/bin/python src/probes.py ref identifiers_back topology        # build and submit-ready
    .venv/bin/python src/probes.py rank7 --cv-only                       # CV veto only
"""
from __future__ import annotations

import sys, json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest

from validation import evaluate, FOLDS_WITH_HORIZON
from models import LGBM_PARAMS, LGBM_TUNED, adversarial_auc
from drift import StepNormalizer
from paths import ARTIFACTS, DATA, LEDGER, SUBMISSIONS

D = DATA
FAR = [(1, 7, 15, 21), (1, 14, 22, 28), (1, 21, 29, 35), (1, 10, 18, 24), (1, 17, 25, 31), (1, 7, 8, 21), (1, 14, 15, 28), (1, 21, 22, 35)]
PROXY = ["feat_136", "feat_101", "feat_103", "feat_100", "feat_139", "feat_137"]
NORM9 = ["feat_2", "feat_106", "feat_107", "feat_109", "feat_142", "feat_115", "feat_143", "feat_151", "feat_145"]
NORM7 = [c for c in NORM9 if c not in ("feat_115", "feat_151")]
TUNED = {k: LGBM_TUNED[k] for k in ("num_leaves", "min_child_samples", "colsample_bytree", "subsample", "subsample_freq",
                                    "reg_lambda", "reg_alpha", "learning_rate", "n_estimators")}
SEEDS_CV, SEEDS_FINAL = (0, 1, 2), (0, 1, 2, 3, 4)
REFERENCE_FILE = "sub_07_lgbm_v2b_tuned_bagged5.csv"


class Data:
    def __init__(self):
        self.train = pd.read_csv(D / "train.csv"); self.test = pd.read_csv(D / "test.csv")
        self.edges = pd.read_csv(D / "txs_edgelist.csv"); self.sample = pd.read_csv(D / "sample_submission.csv")
        self.RAW = [c for c in self.train.columns if c.startswith("feat_")]
        self.RAW_CLEAN = [c for c in self.RAW if c not in PROXY]
        self.LOCAL = [c for c in self.RAW if int(c.split("_")[1]) <= 93]
        self.L = self.train[self.train.label.notna()].reset_index(drop=True)
        self.y, self.ts = self.L["label"], self.L["time_step"]

    def features(self, rank_cols=NORM9, base=None, extra=None):
        """v2b-style matrix: `base` columns with `rank_cols` rank-replaced within step.
        `extra` = (train_df, test_df) of additional columns aligned to L / test rows."""
        base = list(self.RAW_CLEAN if base is None else base)
        rank_cols = [c for c in rank_cols if c in base]
        n = StepNormalizer(rank_cols, "rank", "replace"); cols = n.output_cols(base)
        Xtr = n.transform(self.L[self.RAW], self.ts)[cols]
        Xte = n.transform(self.test[self.RAW], self.test["time_step"])[cols]
        if extra is not None:
            Xtr = pd.concat([Xtr, extra[0].reset_index(drop=True)], axis=1)
            Xte = pd.concat([Xte, extra[1].reset_index(drop=True)], axis=1)
        return Xtr, Xte


def lgbm_fit_predict(Xtr, ytr, Xva, seed=0, sample_weight=None, params=TUNED):
    m = lgb.LGBMClassifier(**{**LGBM_PARAMS, **params, "random_state": seed})
    return m.fit(Xtr, ytr, sample_weight=sample_weight).predict_proba(Xva)[:, 1]


# ------------------------------------------------------------------ probes
# Each returns dict(Xtr, Xte, fit_predict, weights=None). The final prediction is the
# mean of fit_predict(Xtr_all, y_all, Xte, seed) over SEEDS_FINAL, so any component
# (blend, pseudo-labelling) lives inside fit_predict and is applied identically in CV.

def p_ref(d):
    Xtr, Xte = d.features(); return dict(Xtr=Xtr, Xte=Xte, fit_predict=lgbm_fit_predict)


def p_identifiers_back(d):
    Xtr, Xte = d.features(extra=(d.L[PROXY], d.test[PROXY])); return dict(Xtr=Xtr, Xte=Xte, fit_predict=lgbm_fit_predict)


def p_rank7(d):
    Xtr, Xte = d.features(rank_cols=NORM7); return dict(Xtr=Xtr, Xte=Xte, fit_predict=lgbm_fit_predict)


def p_local_only(d):
    Xtr, Xte = d.features(base=d.LOCAL); return dict(Xtr=Xtr, Xte=Xte, fit_predict=lgbm_fit_predict)


TOPO = ["g_in_deg", "g_out_deg", "g_deg", "g_in_out_ratio", "g_unk_pred", "g_unk_succ", "g_unk_share", "g_dag_depth", "g_dag_height", "g_pagerank"]

def p_topology(d):
    from graph_features import build_period_features
    ids = pd.Index(d.train.txId); one = d.RAW[:1]
    gtr = build_period_features(d.edges, ids, "train", d.train.set_index("txId")[one])[TOPO].reindex(d.L.txId)
    gte = build_period_features(d.edges, ids, "test", d.test.set_index("txId")[one])[TOPO].reindex(d.test.txId)
    Xtr, Xte = d.features(extra=(gtr, gte)); return dict(Xtr=Xtr, Xte=Xte, fit_predict=lgbm_fit_predict)


def _anomaly(w):
    def fit_predict(Xtr, ytr, Xva, seed=0, sample_weight=None):
        p = lgbm_fit_predict(Xtr, ytr, Xva, seed, sample_weight)
        a = -IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xtr[np.asarray(ytr) == 0]).score_samples(Xva)
        return (1 - w) * rankdata(p) / len(p) + w * rankdata(a) / len(a)
    return fit_predict

def p_anomaly_10(d):
    Xtr, Xte = d.features(); return dict(Xtr=Xtr, Xte=Xte, fit_predict=_anomaly(0.10))

def p_anomaly_20(d):
    Xtr, Xte = d.features(); return dict(Xtr=Xtr, Xte=Xte, fit_predict=_anomaly(0.20))


def p_adv_weights(d):
    """Covariate-shift importance weights: w = p(test|x) / p(train|x), clipped, mean 1."""
    Xtr, Xte = d.features()
    X = pd.concat([Xtr, Xte], ignore_index=True); z = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros(len(X))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, z):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, colsample_bytree=0.5, verbose=-1, n_jobs=-1, random_state=0)
        oof[va] = m.fit(X.iloc[tr], z[tr]).predict_proba(X.iloc[va])[:, 1]
    p = np.clip(oof[:len(Xtr)], 1e-3, 1 - 1e-3); w = np.clip(p / (1 - p), 0.2, 5.0); w = w / w.mean()
    print(f"  adv_weights: weight range {w.min():.2f}-{w.max():.2f}, share of rows > 1: {(w > 1).mean():.2f}")
    return dict(Xtr=Xtr, Xte=Xte, fit_predict=lgbm_fit_predict, weights=pd.Series(w))


def p_self_train(d, hi=0.9, lo=0.02):
    def fit_predict(Xtr, ytr, Xva, seed=0, sample_weight=None):
        p = lgbm_fit_predict(Xtr, ytr, Xva, seed, sample_weight)
        pos, neg = p > hi, p < lo
        X2 = pd.concat([Xtr, Xva[pos], Xva[neg]], ignore_index=True)
        y2 = np.r_[np.asarray(ytr), np.ones(pos.sum()), np.zeros(neg.sum())].astype(int)
        w2 = None if sample_weight is None else np.r_[np.asarray(sample_weight), np.ones(pos.sum() + neg.sum())]
        return lgbm_fit_predict(X2, y2, Xva, seed, w2)
    Xtr, Xte = d.features(); return dict(Xtr=Xtr, Xte=Xte, fit_predict=fit_predict)


def _adv_weights(d, power=1.0, clip=(0.2, 5.0)):
    Xtr, Xte = d.features()
    X = pd.concat([Xtr, Xte], ignore_index=True); z = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros(len(X))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, z):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, colsample_bytree=0.5, verbose=-1, n_jobs=-1, random_state=0)
        oof[va] = m.fit(X.iloc[tr], z[tr]).predict_proba(X.iloc[va])[:, 1]
    p = np.clip(oof[:len(Xtr)], 1e-3, 1 - 1e-3); w = np.clip((p / (1 - p)) ** power, *clip); w = w / w.mean()
    print(f"  adv weights (power {power}, clip {clip}): range {w.min():.2f}-{w.max():.2f}, share > 1: {(w > 1).mean():.2f}")
    return dict(Xtr=Xtr, Xte=Xte, fit_predict=lgbm_fit_predict, weights=pd.Series(w))

def p_adv_weights_soft(d):   # square root of the odds ratio: gentler re-weighting
    return _adv_weights(d, power=0.5)

def p_adv_weights_hard(d):   # squared odds ratio, wider clip: stronger re-weighting
    return _adv_weights(d, power=2.0, clip=(0.1, 20.0))


def p_pseudo_train_unlab(d, hi=0.9, lo=0.02):
    """Pseudo-labels from the 110k *unlabelled training* rows (same periods as the
    labelled ones), not from the test rows. Inside CV only unlabelled rows from the
    fold's training steps are used, so no validation period leaks in."""
    U = d.train[d.train.label.isna()].reset_index(drop=True)
    n = StepNormalizer(NORM9, "rank", "replace"); cols = n.output_cols(d.RAW_CLEAN)
    XU = n.transform(U[d.RAW], U["time_step"])[cols]; tsU = U["time_step"].to_numpy()
    Xtr, Xte = d.features()
    def fit_predict(Xtr_, ytr, Xva, seed=0, sample_weight=None):
        # unlabelled rows from steps no later than the last labelled training step
        t_max = d.ts[Xtr_.index].max() if len(Xtr_.index) and Xtr_.index.max() < len(d.ts) else d.ts.max()
        m = tsU <= t_max
        p = lgbm_fit_predict(Xtr_, ytr, XU[m], seed, sample_weight)
        pos, neg = p > hi, p < lo
        X2 = pd.concat([Xtr_, XU[m][pos], XU[m][neg]], ignore_index=True)
        y2 = np.r_[np.asarray(ytr), np.ones(pos.sum()), np.zeros(neg.sum())].astype(int)
        w2 = None if sample_weight is None else np.r_[np.asarray(sample_weight), np.ones(pos.sum() + neg.sum())]
        return lgbm_fit_predict(X2, y2, Xva, seed, w2)
    return dict(Xtr=Xtr, Xte=Xte, fit_predict=fit_predict)


def p_xgb_blend(d, w=0.3):
    """Rank blend with a second boosted family (shallow XGBoost, same depth regime)."""
    import xgboost as xgb
    def fit_predict(Xtr, ytr, Xva, seed=0, sample_weight=None):
        p = lgbm_fit_predict(Xtr, ytr, Xva, seed, sample_weight)
        m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=3, subsample=1.0, colsample_bytree=0.6,
                              min_child_weight=10, reg_lambda=1.0, tree_method="hist", n_jobs=-1, random_state=seed)
        q = m.fit(Xtr, ytr, sample_weight=sample_weight).predict_proba(Xva)[:, 1]
        return (1 - w) * rankdata(p) / len(p) + w * rankdata(q) / len(q)
    Xtr, Xte = d.features(); return dict(Xtr=Xtr, Xte=Xte, fit_predict=fit_predict)


def p_xgb_alone(d):
    import xgboost as xgb
    def fit_predict(Xtr, ytr, Xva, seed=0, sample_weight=None):
        m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=3, subsample=1.0, colsample_bytree=0.6,
                              min_child_weight=10, reg_lambda=1.0, tree_method="hist", n_jobs=-1, random_state=seed)
        return m.fit(Xtr, ytr, sample_weight=sample_weight).predict_proba(Xva)[:, 1]
    Xtr, Xte = d.features(); return dict(Xtr=Xtr, Xte=Xte, fit_predict=fit_predict)


def _gnn_setup(d):
    """GNN outputs come from artifacts/gnn_cache (built by gnn_cache.py in a PyTorch-only
    process; LightGBM and PyTorch hang together in one process on macOS). Node order in the
    cache: labelled train rows (L order) then test rows."""
    Xtr, Xte = d.features(); Xte.index = np.arange(len(Xtr), len(Xtr) + len(Xte))
    return Xtr, Xte

def _gnn_load(d, Xtr, Xva, seed):
    """Locate the cached fold from the row indices: training steps (lo, hi) and validation
    steps (vlo, vhi) of the rows, or the final fit when Xva holds test rows."""
    cache = ARTIFACTS / "gnn_cache"
    if Xva.index.min() >= len(d.L):
        z = np.load(cache / f"final_s{seed}.npz")
    else:
        t_tr, t_va = d.ts[Xtr.index].to_numpy(), d.ts[Xva.index].to_numpy()
        key = f"f{t_tr.min()}-{t_tr.max()}_v{t_va.min()}-{t_va.max()}_s{seed}"
        z = np.load(cache / f"{key}.npz")
    pos_tr = pd.Series(np.arange(len(z["idx_tr"])), index=z["idx_tr"]); pos_va = pd.Series(np.arange(len(z["idx_va"])), index=z["idx_va"])
    i_tr, i_va = pos_tr[Xtr.index.to_numpy()].to_numpy(), pos_va[Xva.index.to_numpy()].to_numpy()
    return z["p_tr"][i_tr], z["h_tr"][i_tr], z["p_va"][i_va], z["h_va"][i_va]

def p_gnn_embed(d):
    Xtr, Xte = _gnn_setup(d)
    def fit_predict(Xtr_, ytr, Xva, seed=0, sample_weight=None):
        _, h_tr, _, h_va = _gnn_load(d, Xtr_, Xva, seed)
        E = [f"emb_{i}" for i in range(h_tr.shape[1])]
        A = pd.concat([Xtr_.reset_index(drop=True), pd.DataFrame(h_tr, columns=E)], axis=1)
        B = pd.concat([Xva.reset_index(drop=True), pd.DataFrame(h_va, columns=E)], axis=1)
        return lgbm_fit_predict(A, ytr, B, seed, sample_weight)
    return dict(Xtr=Xtr, Xte=Xte, fit_predict=fit_predict)

def p_gnn_blend(d, w=0.15):
    Xtr, Xte = _gnn_setup(d)
    def fit_predict(Xtr_, ytr, Xva, seed=0, sample_weight=None):
        p = lgbm_fit_predict(Xtr_, ytr, Xva, seed, sample_weight)
        _, _, q, _ = _gnn_load(d, Xtr_, Xva, seed)
        return (1 - w) * rankdata(p) / len(p) + w * rankdata(q) / len(q)
    return dict(Xtr=Xtr, Xte=Xte, fit_predict=fit_predict)


def p_bag10(d):
    Xtr, Xte = d.features(); return dict(Xtr=Xtr, Xte=Xte, fit_predict=lgbm_fit_predict, final_seeds=tuple(range(10)))


PROBES = {"ref": p_ref, "identifiers_back": p_identifiers_back, "rank7": p_rank7, "local_only": p_local_only,
          "topology": p_topology, "anomaly_10": p_anomaly_10, "anomaly_20": p_anomaly_20, "adv_weights": p_adv_weights,
          "self_train": p_self_train, "bag10": p_bag10, "adv_weights_soft": p_adv_weights_soft,
          "adv_weights_hard": p_adv_weights_hard, "pseudo_train_unlab": p_pseudo_train_unlab, "xgb_blend": p_xgb_blend, "xgb_alone": p_xgb_alone, "gnn_embed": p_gnn_embed, "gnn_blend": p_gnn_blend}


# ------------------------------------------------------------------ runner

def cv_summary(d, spec, name):
    kw = {"sample_weight": spec["weights"]} if spec.get("weights") is not None else {}
    r1, _ = evaluate(spec["fit_predict"], spec["Xtr"], d.y, d.ts, folds=FOLDS_WITH_HORIZON, seeds=SEEDS_CV, name=name, verbose=False, **kw)
    r2, _ = evaluate(spec["fit_predict"], spec["Xtr"], d.y, d.ts, folds=FAR, seeds=SEEDS_CV, name=name, verbose=False, **kw)
    f1 = r1.groupby("fold").auc_pooled.mean(); far = r2.groupby("fold").auc_pooled.mean()
    std = [k for k in f1.index if not k.endswith("22-35")]
    return dict(auc_3fold=f1[std].mean(), auc_hard=f1[[k for k in std if "22-28" in k]].mean(), auc_horizon=f1[[k for k in f1.index if k.endswith("22-35")]].mean(),
                pr_auc=r1.pr_auc.mean(), far_mean=far.mean(), far_min=far.min(), n_features=spec["Xtr"].shape[1])


def final_predict(d, spec):
    y = d.y.to_numpy().astype(int); w = None if spec.get("weights") is None else spec["weights"].to_numpy()
    seeds = spec.get("final_seeds", SEEDS_FINAL)
    return np.mean([spec["fit_predict"](spec["Xtr"], y, spec["Xte"], seed=s, sample_weight=w) for s in seeds], axis=0)


def write_submission(d, pred, name):
    sub = d.sample.copy(); sub["target"] = pred
    assert len(sub) == 15_329 and (sub["index"].to_numpy() == d.sample["index"].to_numpy()).all()
    assert sub["target"].between(0, 1).all() and sub["target"].notna().all()
    path = SUBMISSIONS / f"probe_{name}.csv"; sub.to_csv(path, index=False); return path


def ledger_row(name, path, s):
    return (f"| {date.today()} | {path.name if path else '-'} | {name} | {REFERENCE_FILE} | {s['n_features']} | {s['auc_3fold']:.4f} | "
            f"{s['auc_horizon']:.4f} | {s['far_mean']:.4f} | {s['far_min']:.4f} | _pending_ | | |")

HEADER = ("\n## Probe campaign (plan_step4.md): one change per file against the sub_07 reference\n\n"
          "| date | file | change | reference | n features | CV 3-fold | CV horizon | far-window mean | far-window min | public LB | delta vs ref | verdict |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def run(names, cv_only=False):
    d = Data(); ledger = LEDGER; txt = ledger.read_text()
    if "## Probe campaign" not in txt:
        txt += HEADER
    for name in names:
        print(f"\n=== probe: {name}", flush=True)
        spec = PROBES[name](d)
        s = cv_summary(d, spec, name)
        print(f"  CV 3-fold={s['auc_3fold']:.4f} hard={s['auc_hard']:.4f} horizon={s['auc_horizon']:.4f} PR={s['pr_auc']:.4f} | far mean={s['far_mean']:.4f} min={s['far_min']:.4f} | n={s['n_features']}", flush=True)
        json.dump(s, open(ARTIFACTS / f"probe_{name}.json", "w"), indent=1)
        if cv_only:
            continue
        pred = final_predict(d, spec)
        if name == "ref":
            ref = pd.read_csv(SUBMISSIONS / REFERENCE_FILE)["target"].to_numpy()
            print(f"  reference check: max |diff| vs {REFERENCE_FILE} = {np.abs(pred - ref).max():.2e}", flush=True)
            continue
        path = write_submission(d, pred, name)
        from scipy.stats import spearmanr
        ref = pd.read_csv(SUBMISSIONS / REFERENCE_FILE)["target"].to_numpy()
        print(f"  wrote {path}; Spearman vs reference {spearmanr(pred, ref).correlation:.4f}", flush=True)
        if path.name not in txt:
            txt += ledger_row(name, path, s) + "\n"
    ledger.write_text(txt)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args, cv_only="--cv-only" in sys.argv)
