"""Graph neural networks under a strictly inductive, time-respecting protocol.

For a fold with training steps <= T, the training graph contains only nodes with
step <= T and the edges among them; the validation graph contains only the
validation block's nodes. Edges never cross time steps, so the two graphs are
disconnected by construction (asserted). Node features are the same v2b matrix
the LightGBM uses, z-scored with training-node statistics.

Graph variants: "lab" = labelled nodes only (mirrors the test file, which holds
only labelled nodes); "full" = all train nodes incl. unlabelled ones as message-
passing context (loss on labelled nodes only). Models: MLP (no edges, the
control), SAGE (GraphSAGE + linear skip), and SAGE on rewired edges (ablation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from validation import rolling_origin_folds, score, DEFAULT_FOLDS
from drift import StepNormalizer

NORM9 = ["feat_2", "feat_106", "feat_107", "feat_109", "feat_142", "feat_115", "feat_143", "feat_151", "feat_145"]
PROXY = ["feat_136", "feat_101", "feat_103", "feat_100", "feat_139", "feat_137"]


# ------------------------------------------------------------------ data

class GraphData:
    """Node table (train rows, optionally + test rows) with v2b features and an
    edge index restricted to the given nodes."""

    def __init__(self, train: pd.DataFrame, edges: pd.DataFrame, variant: str = "lab", test: pd.DataFrame | None = None):
        RAW = [c for c in train.columns if c.startswith("feat_")]
        RAW_CLEAN = [c for c in RAW if c not in PROXY]
        nodes = train if variant == "full" else train[train.label.notna()]
        nodes = nodes.reset_index(drop=True)
        if test is not None:
            nodes = pd.concat([nodes, test.assign(label=np.nan)], ignore_index=True)
        n = StepNormalizer(NORM9, "rank", "replace"); cols = n.output_cols(RAW_CLEAN)
        self.X = n.transform(nodes[RAW], nodes["time_step"])[cols].to_numpy(dtype=np.float32)
        self.y = nodes["label"].to_numpy()                       # NaN for unlabelled / test
        self.ts = nodes["time_step"].to_numpy()
        self.txId = nodes["txId"].to_numpy()
        self.is_test = np.r_[np.zeros(len(nodes) - (0 if test is None else len(test)), bool), np.ones(0 if test is None else len(test), bool)]
        pos = pd.Series(np.arange(len(nodes)), index=self.txId)
        e = edges[edges.txId1.isin(pos.index) & edges.txId2.isin(pos.index)]
        self.edge_index = np.stack([pos[e.txId1].to_numpy(), pos[e.txId2].to_numpy()])
        self.cols = cols

    def subgraph(self, mask: np.ndarray):
        """Edge index restricted to nodes in `mask`, re-indexed to 0..k-1."""
        idx = np.flatnonzero(mask); remap = -np.ones(len(mask), dtype=np.int64); remap[idx] = np.arange(len(idx))
        s, d = self.edge_index; keep = mask[s] & mask[d]
        ei = np.stack([remap[s[keep]], remap[d[keep]]])
        return idx, ei


def shuffle_edge_index(edge_index: np.ndarray, ts: np.ndarray, seed: int = 0) -> np.ndarray:
    """Rewire within time step, drop self loops, orient by a random node order (keeps a DAG)."""
    rng = np.random.default_rng(seed); s, d = edge_index.copy(), edge_index[1].copy()
    s = edge_index[0].copy()
    for step in np.unique(ts[s]):
        m = ts[s] == step; d[m] = rng.permutation(d[m])
    keep = s != d; s, d = s[keep], d[keep]
    rank = rng.permutation(len(ts)); flip = rank[s] > rank[d]
    s2, d2 = np.where(flip, d, s), np.where(flip, s, d)
    return np.unique(np.stack([s2, d2]), axis=1)


# ------------------------------------------------------------------ models

class MLP(nn.Module):
    def __init__(self, d_in, hidden=128, dropout=0.3):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(d_in, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 1); self.dp = dropout
    def forward(self, x, edge_index=None):
        h = F.dropout(F.relu(self.l1(x)), self.dp, self.training)
        h = F.dropout(F.relu(self.l2(h)), self.dp, self.training)
        return self.out(h).squeeze(-1), h


class SAGE(nn.Module):
    """Two GraphSAGE layers (mean aggregation) with a linear root/skip connection."""
    def __init__(self, d_in, hidden=128, dropout=0.3):
        super().__init__()
        self.c1, self.c2 = SAGEConv(d_in, hidden), SAGEConv(hidden, hidden)
        self.skip = nn.Linear(d_in, hidden); self.out = nn.Linear(hidden, 1); self.dp = dropout
    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.c1(x, edge_index)), self.dp, self.training)
        h = F.dropout(F.relu(self.c2(h, edge_index)) + self.skip(x), self.dp, self.training)
        return self.out(h).squeeze(-1), h


MODELS = {"mlp": MLP, "sage": SAGE}


# ------------------------------------------------------------------ training

def _tensor(a, dtype=torch.float32): return torch.as_tensor(a, dtype=dtype)

def train_model(model_name, X_tr, ei_tr, y_tr, seed=0, epochs=200, lr=1e-3, wd=5e-4, hidden=128, dropout=0.3):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MODELS[model_name](X_tr.shape[1], hidden, dropout)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    x, ei = _tensor(X_tr), _tensor(ei_tr, torch.long)
    lab = ~np.isnan(y_tr); yt = _tensor(y_tr[lab]); lab_t = torch.as_tensor(lab)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        logit, _ = m(x, ei)
        loss = F.binary_cross_entropy_with_logits(logit[lab_t], yt); loss.backward(); opt.step()
    return m


@torch.no_grad()
def predict(m, X, ei):
    m.eval(); logit, h = m(_tensor(X), _tensor(ei, torch.long))
    return torch.sigmoid(logit).numpy(), h.numpy()


def _zscore(X_tr, X_other):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    return (X_tr - mu) / sd, (X_other - mu) / sd


def evaluate_gnn(g: GraphData, model_name="sage", folds=DEFAULT_FOLDS, seeds=(0,), epochs=200, shuffled=False, name=None, verbose=True, **kw):
    """Rolling-origin evaluation. Only labelled validation nodes are scored."""
    name = name or f"{model_name}{'-shuffled' if shuffled else ''}"
    rows = []
    ts = pd.Series(g.ts[~g.is_test])
    for fold, tr, va in rolling_origin_folds(ts, folds):
        tr_full = np.r_[tr, np.zeros(g.is_test.sum(), bool)]; va_full = np.r_[va, np.zeros(g.is_test.sum(), bool)]
        idx_tr, ei_tr = g.subgraph(tr_full); idx_va, ei_va = g.subgraph(va_full)
        # protocol assertion: no edge joins a training node to a validation node
        s, d = g.edge_index; assert not ((tr_full[s] & va_full[d]) | (va_full[s] & tr_full[d])).any()
        if shuffled:
            ei_tr = shuffle_edge_index(ei_tr, g.ts[idx_tr], seed=0)
        X_tr, X_va = _zscore(g.X[idx_tr], g.X[idx_va])
        y_tr, y_va = g.y[idx_tr], g.y[idx_va]
        lab_va = ~np.isnan(y_va)
        for seed in seeds:
            m = train_model(model_name, X_tr, ei_tr, y_tr, seed=seed, epochs=epochs, **kw)
            p, _ = predict(m, X_va, ei_va)
            sc = score(y_va[lab_va].astype(int), p[lab_va], g.ts[idx_va][lab_va])
            rows.append({"model": name, "fold": fold.name, "seed": seed, "n_train": int((~np.isnan(y_tr)).sum()), "n_val": int(lab_va.sum()),
                         "auc_pooled": sc["auc_pooled"], "auc_step_mean": sc["auc_step_mean"], "auc_gap": sc["auc_gap"], "pr_auc": sc["pr_auc"]})
            if verbose:
                print(f"[{name}] {fold.name} seed={seed}  AUC={sc['auc_pooled']:.4f}  step-mean={sc['auc_step_mean']:.4f}  PR={sc['pr_auc']:.4f}", flush=True)
    return pd.DataFrame(rows)


def embed_all(g: GraphData, model_name="sage", seed=0, epochs=200, **kw):
    """Train on all labelled non-test nodes, return (probabilities, embeddings) for every node
    in g, including test nodes (which are never in the training graph)."""
    tr = ~g.is_test; idx_tr, ei_tr = g.subgraph(tr)
    X_tr, X_all = _zscore(g.X[idx_tr], g.X)
    m = train_model(model_name, X_tr, ei_tr, g.y[idx_tr], seed=seed, epochs=epochs, **kw)
    p, h = predict(m, X_all, g.edge_index)
    return p, h
