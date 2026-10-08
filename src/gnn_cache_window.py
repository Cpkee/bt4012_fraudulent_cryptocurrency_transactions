"""Step 9 (docs/plans/plan_step9.md): strictly out-of-fold GraphSAGE embeddings for the two-stage
model, in a PyTorch-only process (LightGBM and PyTorch hang together in one process on macOS).

For a training window [wlo, whi] (labelled nodes only, the "lab" graph variant) and an apply
block [vlo, vhi] (validation steps, or the test file):
  * window rows: 5-fold random out-of-fold inside the window, the same protocol as model A's
    random OOF scores — each fold's model trains with the loss on 4/5 of the window's labels and
    embeds the held-out 1/5 (their labels are never used; their features take part in message
    passing, as any node's do);
  * apply rows: one model trained on the whole window, applied inductively to the apply block's
    own subgraph (edges never cross steps, so the two graphs are disconnected).
families.two_stage(b_gnn_embed=True) reads the cache.

    .venv/bin/python src/gnn_cache_window.py      # writes artifacts/gnn_cache/win<wlo>-<whi>_apply<vlo>-<vhi>_s<seed>.npz
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from gnn import GraphData, train_model, predict, _zscore
from validation import FOLDS_WITH_HORIZON
from paths import ARTIFACTS, DATA

OUT = ARTIFACTS / "gnn_cache"; OUT.mkdir(exist_ok=True)
EPOCHS, HIDDEN, LAST_K, OOF_FOLDS = 200, 64, 14, 5


def key(wlo, whi, vlo, vhi, seed): return f"win{wlo}-{whi}_apply{vlo}-{vhi}_s{seed}"


def build(g: GraphData, mwin: np.ndarray, mapply: np.ndarray, seed: int) -> dict:
    idx_w, ei_w = g.subgraph(mwin); idx_a, ei_a = g.subgraph(mapply)
    X_w, X_a = _zscore(g.X[idx_w], g.X[idx_a]); y_w = g.y[idx_w]
    assert not np.isnan(y_w).any(), "window must hold labelled nodes only"
    p_oof, h_oof = np.full(len(idx_w), np.nan), np.full((len(idx_w), HIDDEN), np.nan, dtype=np.float32)
    for k, (tr_i, ho_i) in enumerate(StratifiedKFold(OOF_FOLDS, shuffle=True, random_state=seed).split(X_w, y_w.astype(int))):
        y_fold = y_w.copy(); y_fold[ho_i] = np.nan                       # held-out labels hidden from the loss
        m = train_model("sage", X_w, ei_w, y_fold, seed=seed * 10 + k, epochs=EPOCHS, hidden=HIDDEN)
        p, h = predict(m, X_w, ei_w); p_oof[ho_i] = p[ho_i]; h_oof[ho_i] = h[ho_i]
    m = train_model("sage", X_w, ei_w, y_w, seed=seed, epochs=EPOCHS, hidden=HIDDEN)
    p_a, h_a = predict(m, X_a, ei_a)
    return dict(tx_oof=g.txId[idx_w], p_oof=p_oof, h_oof=h_oof, tx_apply=g.txId[idx_a], p_apply=p_a, h_apply=h_a.astype(np.float32))


if __name__ == "__main__":
    train = pd.read_csv(DATA / "train.csv"); test = pd.read_csv(DATA / "test.csv"); edges = pd.read_csv(DATA / "txs_edgelist.csv")
    g = GraphData(train, edges, "lab", test=test); ts = g.ts; lab = ~g.is_test
    jobs = []
    for lo, hi, vlo, vhi in FOLDS_WITH_HORIZON:                        # the two-stage CV folds, cv seeds 0-1
        for seed in (0, 1):
            jobs.append((hi - LAST_K + 1, hi, vlo, vhi, seed, lab & (ts >= hi - LAST_K + 1) & (ts <= hi), lab & (ts >= vlo) & (ts <= vhi)))
    for seed in range(5):                                               # final fit: window 22-35 -> test rows
        jobs.append((22, 35, 36, 49, seed, lab & (ts >= 22) & (ts <= 35), g.is_test))
    for wlo, whi, vlo, vhi, seed, mwin, mapply in jobs:
        f = OUT / f"{key(wlo, whi, vlo, vhi, seed)}.npz"
        if f.exists(): continue
        np.savez_compressed(f, **build(g, mwin, mapply, seed)); print("wrote", f.name, flush=True)
    print("done", flush=True)
