"""Model factories with the ``fit_predict(X_tr, y_tr, X_val, seed) -> p_val``
signature that ``validation.evaluate`` expects, plus adversarial validation.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

LGBM_PARAMS = dict(
    n_estimators=2000, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.6, reg_lambda=1.0, verbose=-1, n_jobs=-1,
)


def logreg_fit_predict(Xtr, ytr, Xva, seed=0):
    # standardise, then clip to +-10 sd: a few raw features have extreme tails that
    # overflow the logistic loss otherwise
    m = make_pipeline(StandardScaler(), FunctionTransformer(lambda a: np.clip(a, -10, 10)),
                      LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    return m.fit(Xtr, ytr).predict_proba(Xva)[:, 1]


def rf_fit_predict(Xtr, ytr, Xva, seed=0):
    m = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=2,
                               n_jobs=-1, random_state=seed)
    return m.fit(Xtr, ytr).predict_proba(Xva)[:, 1]


class LGBMFitPredict:
    """LightGBM with the boosting-round count chosen on the last 15% of the
    training rows (rows must be time-ordered, so this is the most recent tail),
    then refit on the full training window with that count.

    ``best_iters`` collects the chosen counts so the final model can reuse
    their median.
    """

    def __init__(self, params: dict | None = None, tail: float = 0.15):
        self.params = {**LGBM_PARAMS, **(params or {})}
        self.tail = tail
        self.best_iters: list[int] = []

    def __call__(self, Xtr, ytr, Xva, seed=0):
        p = {**self.params, "random_state": seed,
             "scale_pos_weight": (ytr == 0).sum() / max((ytr == 1).sum(), 1)}
        cut = int(len(Xtr) * (1 - self.tail))
        m = lgb.LGBMClassifier(**p).fit(
            Xtr.iloc[:cut], ytr[:cut], eval_set=[(Xtr.iloc[cut:], ytr[cut:])],
            eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        self.best_iters.append(int(m.best_iteration_))
        final = lgb.LGBMClassifier(**{**p, "n_estimators": int(m.best_iteration_)}).fit(Xtr, ytr)
        return final.predict_proba(Xva)[:, 1]

    @property
    def median_best_iter(self) -> int:
        return int(np.median(self.best_iters))


def lgbm_fixed_fit_predict(n_estimators: int = 300, pos_weight: bool = False, params: dict | None = None):
    """LightGBM with a fixed round count and, by default, no class weight.

    Rolling CV (artifacts/cv_lgbm_rounds.csv) showed that both ingredients of
    ``LGBMFitPredict`` cost AUC under temporal drift: ``scale_pos_weight`` lowers
    the hard middle fold by ~0.006 and early stopping on the most recent tail
    picks a round count for the tail's base rate, not the next period's. AUC is
    rank-based, so the class weight buys nothing for the metric. Round count is
    a hyperparameter chosen on rolling CV instead.
    """
    base = {**LGBM_PARAMS, **(params or {}), "n_estimators": n_estimators}

    def fit_predict(Xtr, ytr, Xva, seed=0, sample_weight=None):
        p = {**base, "random_state": seed}
        if pos_weight:
            p["scale_pos_weight"] = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        return lgb.LGBMClassifier(**p).fit(Xtr, ytr, sample_weight=sample_weight).predict_proba(Xva)[:, 1]

    fit_predict.n_estimators = n_estimators
    fit_predict.pos_weight = pos_weight
    return fit_predict


def bagged_fit_predict(factory, seeds=(0, 1, 2, 3, 4)):
    """Average the probabilities of `factory` fitted with each seed. The seed
    passed by evaluate() is added to every bag seed so folds stay independent."""
    def fit_predict(Xtr, ytr, Xva, seed=0, sample_weight=None):
        ps = [factory(Xtr, ytr, Xva, seed=seed * 100 + s, sample_weight=sample_weight) for s in seeds]
        return np.mean(ps, axis=0)
    return fit_predict


def rf_fit_predict_w(Xtr, ytr, Xva, seed=0, sample_weight=None):
    m = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=2,
                               n_jobs=-1, random_state=seed)
    return m.fit(Xtr, ytr, sample_weight=sample_weight).predict_proba(Xva)[:, 1]


def rank_blend_fit_predict(factories: dict, weights: dict):
    """Rank-average the predictions of several fit_predict callables."""
    from scipy.stats import rankdata
    def fit_predict(Xtr, ytr, Xva, seed=0, sample_weight=None):
        out = np.zeros(len(Xva))
        for k, f in factories.items():
            out += weights[k] * rankdata(f(Xtr, ytr, Xva, seed=seed, sample_weight=sample_weight)) / len(Xva)
        return out / sum(weights.values())
    return fit_predict


# Step 3: random search (artifacts/lgbm_search.csv), then a leaves x rounds grid around
# the winner (artifacts/lgbm_search_confirm.csv), each confirmed with five seeds. The
# optimum is far shallower than the default: 7 leaves, 600 rounds at lr 0.05, no row
# subsampling. Horizon fold 0.9147 -> 0.9420, hard fold 0.9568 -> 0.9675, 3-fold
# 0.9821 -> 0.9854, PR-AUC 0.9180 -> 0.9262 (five seeds, v2b features).
LGBM_TUNED = dict(
    n_estimators=600, learning_rate=0.05, num_leaves=7, min_child_samples=40,
    colsample_bytree=0.6, subsample=1.0, subsample_freq=1, reg_lambda=1.0, reg_alpha=0.0,
    verbose=-1, n_jobs=-1,
)


LGBM_SEARCH_SPACE = {
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [10, 20, 40, 80, 160],
    "colsample_bytree": [0.3, 0.5, 0.6, 0.8],
    "subsample": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 20.0],
    "reg_alpha": [0.0, 1.0, 5.0],
    "learning_rate": [0.02, 0.03, 0.05],
}
ROUNDS_TIMES_LR = 9.0   # the step-1 choice, 300 rounds x 0.03


def sample_lgbm_params(rng, n: int):
    """Random configurations from LGBM_SEARCH_SPACE. Rounds scale with the
    learning rate so every configuration sees the same total shrinkage."""
    out = []
    for _ in range(n):
        p = {k: v[rng.integers(len(v))] for k, v in LGBM_SEARCH_SPACE.items()}
        p["n_estimators"] = int(round(ROUNDS_TIMES_LR / p["learning_rate"]))
        p["subsample_freq"] = 1
        out.append(p)
    return out


def time_order(*frames, time_step: pd.Series):
    """Stable-sort frames/series by time_step so the LightGBM tail is the most recent rows."""
    order = np.argsort(time_step.to_numpy(), kind="stable")
    return tuple(f.iloc[order].reset_index(drop=True) for f in frames)


def adversarial_auc(Xa: pd.DataFrame, Xb: pd.DataFrame, seed: int = 0):
    """Out-of-fold AUC of a classifier separating rows of Xa (0) from Xb (1),
    and normalised gain importance. AUC near 0.5 means the two sets are
    indistinguishable on these features."""
    X = pd.concat([Xa, Xb], ignore_index=True)
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    oof = np.zeros(len(X)); imp = pd.Series(0.0, index=X.columns)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5,
                               random_state=seed, verbose=-1, n_jobs=-1).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        imp += pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    return roc_auc_score(y, oof), (imp / imp.sum()).sort_values(ascending=False)


def single_feature_adversarial_auc(Xa: pd.DataFrame, Xb: pd.DataFrame, cols=None, seed: int = 0) -> pd.Series:
    """Out-of-fold AUC of a one-feature tree model separating Xa (0) from Xb (1),
    for every column. A tree, unlike a rank test, also catches non-monotone
    identifiers such as a component size that takes different values per period.
    Values near 1 mark columns that identify the period on their own."""
    cols = list(Xa.columns if cols is None else cols)
    X = pd.concat([Xa[cols], Xb[cols]], ignore_index=True)
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y))
    out = {}
    for c in cols:
        if X[c].nunique() <= 1:
            out[c] = 0.5; continue
        oof = np.zeros(len(X))
        for tr, va in folds:
            m = lgb.LGBMClassifier(n_estimators=40, learning_rate=0.1, num_leaves=15, min_child_samples=50,
                                   verbose=-1, n_jobs=-1).fit(X.iloc[tr][[c]], y[tr])
            oof[va] = m.predict_proba(X.iloc[va][[c]])[:, 1]
        out[c] = roc_auc_score(y, oof)
    return pd.Series(out, name="single_feature_adversarial_auc").sort_values(ascending=False)


PERIOD_ID_THRESHOLD = 0.95  # single-feature adversarial AUC above this = period identifier
