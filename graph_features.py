"""Structural graph features from txs_edgelist.csv, computed per period.

Two facts about the data drive the design (verified in main.ipynb):

* Edges never cross the train/test boundary, so each period (train: steps 1-35,
  test: steps 36-49) is its own disconnected graph. Every feature is computed
  inside one period only.
* The test file holds only the *labelled* test-period nodes. The remaining
  46,668 test-period nodes appear in the edge list (their ids and edges) but
  have no feature vector. Topology is therefore fully observed in both periods,
  but neighbour *feature* aggregates on test can only see labelled neighbours.

Hence two families:

* ``topology_features`` - degrees, coverage, component size, DAG depth/height,
  PageRank. Uses every edge of the period, so the semantics match exactly
  across train and test.
* ``aggregate_features`` - neighbour statistics of node features. Only defined
  over neighbours whose features are known; on train this can be run over all
  nodes ("full") or over labelled nodes only ("labelled-only", mirroring test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


# --------------------------------------------------------------------------- #
# period selection
# --------------------------------------------------------------------------- #
def period_edges(edges: pd.DataFrame, train_ids: pd.Index, period: str) -> pd.DataFrame:
    """Edges belonging to one period.

    train period = edges touching a train node (both endpoints are always train).
    test period  = every other edge (test-test, test-dangling, dangling-dangling).
    """
    touches_train = edges["txId1"].isin(train_ids) | edges["txId2"].isin(train_ids)
    return edges[touches_train] if period == "train" else edges[~touches_train]


def _index_nodes(e: pd.DataFrame) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    nodes = pd.Index(np.union1d(e["txId1"].unique(), e["txId2"].unique()))
    return nodes, nodes.get_indexer(e["txId1"]), nodes.get_indexer(e["txId2"])


# --------------------------------------------------------------------------- #
# topology (parity-exact across periods)
# --------------------------------------------------------------------------- #
def _dag_depth_height(n: int, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Longest path from any source (depth) and to any sink (height). Graph is a DAG."""
    indeg = np.bincount(dst, minlength=n)
    succ = [[] for _ in range(n)]
    for u, v in zip(src.tolist(), dst.tolist()):
        succ[u].append(v)
    order, stack = [], [i for i in range(n) if indeg[i] == 0]
    indeg = indeg.copy()
    while stack:
        u = stack.pop(); order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if len(order) != n:
        raise ValueError("graph has a cycle; DAG depth undefined")
    depth = np.zeros(n, dtype=np.int32)
    for u in order:
        for v in succ[u]:
            if depth[u] + 1 > depth[v]:
                depth[v] = depth[u] + 1
    height = np.zeros(n, dtype=np.int32)
    for u in reversed(order):
        for v in succ[u]:
            if height[v] + 1 > height[u]:
                height[u] = height[v] + 1
    return depth, height


def _pagerank(n: int, src: np.ndarray, dst: np.ndarray, alpha=0.85, iters=100, tol=1e-8) -> np.ndarray:
    A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    out = np.asarray(A.sum(axis=1)).ravel()
    inv = np.where(out > 0, 1.0 / np.maximum(out, 1), 0.0)
    P = sp.diags(inv) @ A
    dangling = out == 0
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r_new = alpha * (P.T @ r + r[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(r_new - r).sum() < tol:
            r = r_new; break
        r = r_new
    return r


def topology_features(e: pd.DataFrame, known_ids: pd.Index) -> pd.DataFrame:
    """One row per node in the period graph (including nodes without features).

    ``known_ids`` are nodes whose features exist; neighbours outside it are
    "unknown" (unlabelled train nodes in the full-train view, dangling nodes in
    the test view). Coverage counts refer to them.
    """
    nodes, src, dst = _index_nodes(e)
    n = len(nodes)
    in_deg = np.bincount(dst, minlength=n)
    out_deg = np.bincount(src, minlength=n)
    known = nodes.isin(known_ids)
    unk_pred = np.bincount(dst, weights=~known[src], minlength=n)   # predecessors without features
    unk_succ = np.bincount(src, weights=~known[dst], minlength=n)
    _, comp = connected_components(sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)), directed=False)
    comp_size = np.bincount(comp)[comp]
    depth, height = _dag_depth_height(n, src, dst)
    pr = _pagerank(n, src, dst)
    # relative to the mean PageRank of the node's own component, so the value does
    # not carry the component's (i.e. the time step's) size
    comp_mean = np.bincount(comp, weights=pr) / np.bincount(comp)
    pr_rel = pr / comp_mean[comp]
    deg = in_deg + out_deg
    return pd.DataFrame({
        "g_in_deg": in_deg, "g_out_deg": out_deg, "g_deg": deg,
        "g_in_out_ratio": (in_deg + 1) / (out_deg + 1),
        "g_unk_pred": unk_pred, "g_unk_succ": unk_succ,
        "g_unk_share": (unk_pred + unk_succ) / np.maximum(deg, 1),
        "g_wcc_size": comp_size, "g_wcc_log": np.log1p(comp_size),
        "g_dag_depth": depth, "g_dag_height": height,
        "g_pagerank": pr_rel,          # 1.0 = average node of its component
    }, index=nodes)


# --------------------------------------------------------------------------- #
# neighbour feature aggregates (only over neighbours with known features)
# --------------------------------------------------------------------------- #
def aggregate_features(e: pd.DataFrame, feats: pd.DataFrame,
                       stats=("mean", "max", "std")) -> pd.DataFrame:
    """Neighbour statistics of ``feats`` (index = txId) for every node in ``feats``.

    pred_*: over predecessors (txId1 of edges into the node)
    succ_*: over successors  (txId2 of edges out of the node)
    nb2_mean_*: 2-hop smoothing = mean over undirected neighbours of their 1-hop undirected mean
    delta_*: node value minus its undirected 1-hop mean
    Edges whose other endpoint has no features are ignored.
    """
    cols = list(feats.columns)
    e = e[e["txId1"].isin(feats.index) & e["txId2"].isin(feats.index)]

    def agg(group_key: str, other_key: str, prefix: str) -> pd.DataFrame:
        m = e[[group_key, other_key]].merge(feats, left_on=other_key, right_index=True, how="left")
        g = m.drop(columns=[other_key]).groupby(group_key)[cols].agg(list(stats))
        g.columns = [f"{prefix}_{s}_{c}" for c, s in g.columns]
        g[f"{prefix}_n"] = m.groupby(group_key).size()
        return g

    pred = agg("txId2", "txId1", "pred")
    succ = agg("txId1", "txId2", "succ")

    # undirected 1-hop mean, then 2-hop mean via a second pass
    und = pd.concat([e.rename(columns={"txId1": "a", "txId2": "b"}),
                     e.rename(columns={"txId2": "a", "txId1": "b"})])
    m1 = und.merge(feats, left_on="b", right_index=True, how="left").drop(columns="b").groupby("a")[cols].mean()
    m2 = und.merge(m1, left_on="b", right_index=True, how="left").drop(columns="b").groupby("a")[cols].mean()
    m1.columns = [f"nb1_mean_{c}" for c in cols]
    m2.columns = [f"nb2_mean_{c}" for c in cols]
    delta = (feats[cols] - m1.reindex(feats.index).to_numpy())
    delta.columns = [f"delta_{c}" for c in cols]

    out = pd.concat([pred, succ, m1, m2], axis=1).reindex(feats.index)
    out[["pred_n", "succ_n"]] = out[["pred_n", "succ_n"]].fillna(0)
    return pd.concat([out, delta], axis=1)


def build_period_features(edges: pd.DataFrame, train_ids: pd.Index, period: str,
                          feats: pd.DataFrame, known_ids: pd.Index | None = None) -> pd.DataFrame:
    """Topology + aggregates for one period, indexed by txId (only ids in ``feats``).

    ``feats``: node features to aggregate (index txId). Nodes in ``feats`` are the
    "known" set for coverage unless ``known_ids`` is given explicitly.
    """
    e = period_edges(edges, train_ids, period)
    known = feats.index if known_ids is None else known_ids
    topo = topology_features(e, known)
    aggr = aggregate_features(e, feats)
    return pd.concat([topo.reindex(feats.index), aggr], axis=1)


def shuffle_edges(edges: pd.DataFrame, train_ids: pd.Index, seed: int = 0) -> pd.DataFrame:
    """Ablation: randomly rewire edges within each period.

    Destinations are permuted, then every edge is oriented from the lower to the
    higher position in a random node order, so the rewired graph is still a DAG
    (the real graph is one) and DAG depth/height remain defined.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for period in ("train", "test"):
        e = period_edges(edges, train_ids, period).copy()
        e["txId2"] = rng.permutation(e["txId2"].to_numpy())
        e = e[e["txId1"] != e["txId2"]]
        nodes = np.union1d(e["txId1"].unique(), e["txId2"].unique())
        rank = pd.Series(rng.permutation(len(nodes)), index=nodes)
        flip = rank[e["txId1"]].to_numpy() > rank[e["txId2"]].to_numpy()
        a, b = e["txId1"].to_numpy().copy(), e["txId2"].to_numpy().copy()
        a[flip], b[flip] = e["txId2"].to_numpy()[flip], e["txId1"].to_numpy()[flip]
        parts.append(pd.DataFrame({"txId1": a, "txId2": b}).drop_duplicates())
    return pd.concat(parts, ignore_index=True)
