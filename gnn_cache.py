"""Precompute GraphSAGE probabilities and embeddings per rolling fold and seed, in a
PyTorch-only process (LightGBM and PyTorch each ship an OpenMP runtime and hang
when used in the same process on macOS). probes.py reads the cache.

    .venv/bin/python gnn_cache.py          # writes artifacts/gnn_cache/<key>.npz
Keys: f{lo}-{hi}_v{vlo}-{vhi}_s{seed} for CV folds; final_s{seed} for the fit on all
labelled train rows applied to the test rows.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from gnn import GraphData, train_model, predict, _zscore
from validation import FOLDS_WITH_HORIZON

FAR = [(1, 7, 15, 21), (1, 14, 22, 28), (1, 21, 29, 35), (1, 10, 18, 24), (1, 17, 25, 31), (1, 7, 8, 21), (1, 14, 15, 28), (1, 21, 22, 35)]
D = Path.home() / ".cache/bt4012/bt-4012-competition-2026"; OUT = Path("artifacts/gnn_cache")
EPOCHS, HIDDEN = 200, 64


def fold_key(lo, hi, vlo, vhi, seed): return f"f{lo}-{hi}_v{vlo}-{vhi}_s{seed}"


def run_fold(g, mtr, mva, seed):
    idx_tr, ei_tr = g.subgraph(mtr); idx_va, ei_va = g.subgraph(mva)
    X_tr, X_va = _zscore(g.X[idx_tr], g.X[idx_va])
    m = train_model("sage", X_tr, ei_tr, g.y[idx_tr], seed=seed, epochs=EPOCHS, hidden=HIDDEN)
    p_tr, h_tr = predict(m, X_tr, ei_tr); p_va, h_va = predict(m, X_va, ei_va)
    return dict(idx_tr=idx_tr, p_tr=p_tr, h_tr=h_tr, idx_va=idx_va, p_va=p_va, h_va=h_va)


if __name__ == "__main__":
    train = pd.read_csv(D / "train.csv"); test = pd.read_csv(D / "test.csv"); edges = pd.read_csv(D / "txs_edgelist.csv")
    g = GraphData(train, edges, "lab", test=test)
    n_lab = int((~g.is_test).sum()); ts = g.ts
    folds = list(dict.fromkeys(FOLDS_WITH_HORIZON + FAR))
    for lo, hi, vlo, vhi in folds:
        for seed in (0, 1, 2):
            key = fold_key(lo, hi, vlo, vhi, seed); f = OUT / f"{key}.npz"
            if f.exists(): continue
            mtr = (~g.is_test) & (ts >= lo) & (ts <= hi); mva = (~g.is_test) & (ts >= vlo) & (ts <= vhi)
            np.savez_compressed(f, **run_fold(g, mtr, mva, seed)); print("wrote", key, flush=True)
    for seed in range(5):
        key = f"final_s{seed}"; f = OUT / f"{key}.npz"
        if f.exists(): continue
        np.savez_compressed(f, **run_fold(g, ~g.is_test, g.is_test, seed)); print("wrote", key, flush=True)
    print("done")
