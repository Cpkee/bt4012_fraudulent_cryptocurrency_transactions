"""Temporal drift handling: recency weights, per-step feature normalisation,
and score-level prior-shift correction.

Everything here is computed per time_step. Edges never cross steps and every
test step has at least 471 rows, so per-step statistics are stable on both
sides of the train/test boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- recency

def recency_weights(ts: np.ndarray, half_life: float | None, t_ref: int | None = None) -> np.ndarray:
    """Exponential weights 0.5 ** ((t_ref - t) / half_life); t_ref defaults to
    the latest training step. half_life=None gives uniform weights."""
    ts = np.asarray(ts, dtype=float)
    if half_life is None:
        return np.ones_like(ts)
    t_ref = ts.max() if t_ref is None else t_ref
    return 0.5 ** ((t_ref - ts) / half_life)


# ------------------------------------------------- per-step normalisation

class StepNormalizer:
    """Within-time-step rank, z-score or median-deviation of selected columns.

    Statistics are computed on whatever frame is passed to transform(), per
    step, so train and test are normalised against their own steps. No
    training-set statistics are carried across, which is what makes the
    transform valid on a period the model never saw.
    """

    def __init__(self, cols: list[str], method: str = "rank", mode: str = "add", suffix: str | None = None):
        assert method in ("rank", "z", "dev")
        assert mode in ("add", "replace")
        self.cols, self.method, self.mode = list(cols), method, mode
        self.suffix = suffix or f"_{method}"

    def transform(self, df: pd.DataFrame, time_step: pd.Series) -> pd.DataFrame:
        g = df[self.cols].groupby(time_step.to_numpy())
        if self.method == "rank":
            out = g.rank(pct=True)
        elif self.method == "z":
            out = (df[self.cols] - g.transform("mean")) / g.transform("std").replace(0, 1)
        else:  # deviation from the step median
            out = df[self.cols] - g.transform("median")
        out.columns = [f"{c}{self.suffix}" for c in self.cols]
        if self.mode == "replace":
            return pd.concat([df.drop(columns=self.cols), out], axis=1)
        return pd.concat([df, out], axis=1)

    def output_cols(self, base_cols: list[str]) -> list[str]:
        new = [f"{c}{self.suffix}" for c in self.cols]
        if self.mode == "replace":
            return [c for c in base_cols if c not in self.cols] + new
        return list(base_cols) + new


# ------------------------------------------------ score-level corrections

def rank_within_step(p: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Replace each score by its percentile rank inside its own time step."""
    return pd.Series(p).groupby(np.asarray(ts)).rank(pct=True).to_numpy()


def adjust_scores(p: np.ndarray, prior_train: float, prior_new: float) -> np.ndarray:
    """Bayes-adjust posteriors from the training prior to a new prior
    (Saerens, Latinne & Decaestecker 2002, eq. 2)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    w1, w0 = prior_new / prior_train, (1 - prior_new) / (1 - prior_train)
    return w1 * p / (w1 * p + w0 * (1 - p))


def estimate_prior_em(p: np.ndarray, prior_train: float, iters: int = 100, tol: float = 1e-6) -> float:
    """EM estimate of the positive prior in an unlabelled sample from a
    classifier's posteriors (Saerens et al. 2002). Returns the new prior."""
    prior = prior_train
    for _ in range(iters):
        post = adjust_scores(p, prior_train, prior)
        new = float(post.mean())
        if abs(new - prior) < tol:
            return new
        prior = new
    return prior


def prior_shift_correct(p: np.ndarray, ts: np.ndarray, prior_train: float,
                        min_rows: int = 50) -> tuple[np.ndarray, pd.Series]:
    """Per-step EM prior estimate, then per-step Bayes adjustment of the
    scores. Returns (adjusted scores, estimated prior per step)."""
    p, ts = np.asarray(p, dtype=float), np.asarray(ts)
    out = p.copy(); priors = {}
    for s in np.unique(ts):
        m = ts == s
        pr = estimate_prior_em(p[m], prior_train) if m.sum() >= min_rows else prior_train
        pr = float(np.clip(pr, 1e-3, 1 - 1e-3))
        priors[int(s)] = pr
        out[m] = adjust_scores(p[m], prior_train, pr)
    return out, pd.Series(priors, name="prior_est")


def bbse_prior(p: np.ndarray, y_cal: np.ndarray, p_cal: np.ndarray, threshold: float = 0.5) -> float:
    """Black-box shift estimation (Lipton et al. 2018) for a binary classifier
    with a hard threshold: solves C^T q = mu for the new prior, where C is the
    confusion matrix on calibration data and mu the predicted-positive rate on
    the new sample."""
    yhat_cal = (p_cal >= threshold).astype(int)
    tpr = yhat_cal[y_cal == 1].mean() if (y_cal == 1).any() else 0.5
    fpr = yhat_cal[y_cal == 0].mean() if (y_cal == 0).any() else 0.5
    mu = (np.asarray(p) >= threshold).mean()
    if abs(tpr - fpr) < 1e-6:
        return float(np.nan)
    return float(np.clip((mu - fpr) / (tpr - fpr), 1e-3, 1 - 1e-3))
