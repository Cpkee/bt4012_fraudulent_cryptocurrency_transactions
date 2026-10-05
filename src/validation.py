"""Rolling-origin (expanding window) validation by time_step.

The competition split is temporal: train = steps 1-35, test = steps 36-49.
Random K-fold would leak future information across folds, so every model is
evaluated on folds that validate strictly after their training window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Validation blocks of 7 steps, mirroring the 14-step forward gap of the test set.
DEFAULT_FOLDS = [
    (1, 14, 15, 21),
    (1, 21, 22, 28),
    (1, 28, 29, 35),
]

# Step 2: one more fold with the test set's horizon. The test window (36-49) sits
# 14 steps past the end of training; this fold validates 14 steps past step 21.
HORIZON_FOLD = (1, 21, 22, 35)
FOLDS_WITH_HORIZON = DEFAULT_FOLDS + [HORIZON_FOLD]


@dataclass(frozen=True)
class Fold:
    train_lo: int
    train_hi: int
    val_lo: int
    val_hi: int

    @property
    def name(self) -> str:
        return f"train {self.train_lo}-{self.train_hi} | val {self.val_lo}-{self.val_hi}"


def rolling_origin_folds(
    time_step: pd.Series, folds=DEFAULT_FOLDS
) -> Iterator[tuple[Fold, np.ndarray, np.ndarray]]:
    """Yield (fold, train_mask, val_mask) over positional rows of `time_step`."""
    ts = time_step.to_numpy()
    for lo, hi, vlo, vhi in folds:
        f = Fold(lo, hi, vlo, vhi)
        assert hi < vlo, "validation block must start after the training window"
        yield f, (ts >= lo) & (ts <= hi), (ts >= vlo) & (ts <= vhi)


def per_step_auc(y: np.ndarray, p: np.ndarray, ts: np.ndarray) -> pd.Series:
    """ROC AUC within each time step; NaN where a step has a single class."""
    out = {}
    for s in np.unique(ts):
        m = ts == s
        out[int(s)] = roc_auc_score(y[m], p[m]) if len(np.unique(y[m])) == 2 else np.nan
    return pd.Series(out, name="auc")


def score(y: np.ndarray, p: np.ndarray, ts: np.ndarray) -> dict:
    ps = per_step_auc(y, p, ts)
    pooled = roc_auc_score(y, p)
    return {
        "auc_pooled": pooled,
        "auc_step_mean": float(ps.mean()),
        # pooled minus per-step mean: > 0 means scores rank well inside steps but
        # are not comparable across steps (cross-step scale error), < 0 the reverse
        "auc_gap": pooled - float(ps.mean()),
        "pr_auc": average_precision_score(y, p),
        "per_step": ps,
    }


def evaluate(
    fit_predict: Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y: pd.Series,
    time_step: pd.Series,
    folds=DEFAULT_FOLDS,
    seeds=(0,),
    name: str = "model",
    verbose: bool = True,
    sample_weight: pd.Series | None = None,
    weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    postprocess: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run `fit_predict(X_tr, y_tr, X_val, seed) -> p_val` over rolling folds.

    Only labelled rows (y notna) are used for fitting and scoring. Returns
    (summary table with one row per fold x seed, per-step AUC table).

    sample_weight: per-row weights aligned with X; passed to fit_predict as a
        keyword when given. weight_fn: alternative, called with the training
        rows' time steps so weights can depend on the fold's own last step
        (recency). postprocess: called with (p_val, time_step_val) and must
        return adjusted scores; used for score-level corrections.
    """
    labelled = y.notna().to_numpy()
    rows, steps = [], []
    for fold, tr, va in rolling_origin_folds(time_step, folds):
        tr &= labelled
        va &= labelled
        ts_tr, ts_va = time_step[tr].to_numpy(), time_step[va].to_numpy()
        kw = {}
        if weight_fn is not None:
            kw["sample_weight"] = weight_fn(ts_tr)
        elif sample_weight is not None:
            kw["sample_weight"] = sample_weight[tr].to_numpy()
        for seed in seeds:
            p = fit_predict(X[tr], y[tr].to_numpy().astype(int), X[va], seed, **kw)
            if postprocess is not None:
                p = postprocess(p, ts_va)
            s = score(y[va].to_numpy().astype(int), p, ts_va)
            rows.append({"model": name, "fold": fold.name, "seed": seed,
                         "n_train": int(tr.sum()), "n_val": int(va.sum()),
                         "auc_pooled": s["auc_pooled"], "auc_step_mean": s["auc_step_mean"],
                         "auc_gap": s["auc_gap"], "pr_auc": s["pr_auc"]})
            steps.append(s["per_step"].rename(f"{fold.val_lo}-{fold.val_hi}/s{seed}"))
            if verbose:
                print(f"[{name}] {fold.name} seed={seed}  AUC={s['auc_pooled']:.4f}  "
                      f"step-mean={s['auc_step_mean']:.4f}  PR-AUC={s['pr_auc']:.4f}")
    return pd.DataFrame(rows), pd.concat(steps, axis=1)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over folds and seeds per model."""
    cols = [c for c in ("auc_pooled", "auc_step_mean", "auc_gap", "pr_auc") if c in results]
    g = results.groupby("model")[cols]
    return g.agg(["mean", "std"]).round(4)
