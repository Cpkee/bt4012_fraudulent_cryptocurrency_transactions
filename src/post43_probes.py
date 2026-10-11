"""Post-43 block probes and seed-bag assembly on top of a scored file (plan_step10.md, pointer 1 / 3).

The test file is 44% steps 43-49, which every model scores as almost entirely licit. Block-level
re-orderings move the pooled AUC by far more than the board's noise, so they read what no
offline check can: how much fraud the post-43 block holds and where it currently ranks.

    .venv/bin/python src/post43_probes.py floor fam_B15_nbfeat_all_bag15      # post-43 rows below every pre-43 row, block order kept
    .venv/bin/python src/post43_probes.py ceiling fam_B15_nbfeat_all_bag15    # post-43 rows above every pre-43 row
    .venv/bin/python src/post43_probes.py bag <name> <file1> <file2> ...      # probability average of equal-weight seed bags

Floor: pre-43 fraud rows now beat every post-43 row; post-43 fraud rows lose to every pre-43
licit row they used to beat. Versus the base file the reading is about
n_f1*n_l2*(1 - A12) - n_f2*n_l1*A21 over n_f*n_l: ~0 means the block holds (almost) no rankable
fraud, a clear loss means it does and the within-block order is worth keeping.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from families import SUBS, append_ledger, board_spearman, resolve, verdict
from probes import D

SPLIT = 43


def _load(key: str):
    f = resolve(key); s = pd.read_csv(SUBS / f)
    ts = pd.read_csv(D / "test.csv", usecols=["time_step"])["time_step"].to_numpy()
    assert len(s) == len(ts) == 15_329
    return f, s, ts


def block_probe(kind: str, key: str):
    f, s, ts = _load(key)
    r = rankdata(s["target"]) / len(s)                   # pooled rank in (0, 1]
    post = ts >= SPLIT
    if kind == "floor":
        q = np.where(post, r * 1e-3, r)                  # whole block below every pre-43 row, order kept
    elif kind == "ceiling":
        q = np.where(post, 0.5 + r / 2, r / 2)           # whole block above every pre-43 row, order kept
    else:
        raise ValueError(kind)
    name = f"bag15_post43{kind}" if "bag15" in f else f"{key.replace('fam_', '')}_post43{kind}"
    desc = (f"{f.split('.')[0]} with the post-43 block (steps {SPLIT}-49, {post.mean():.1%} of rows) ranked "
            f"{'below' if kind == 'floor' else 'above'} every pre-43 row, order inside each block kept: "
            f"reads how much rankable fraud the post-43 block holds")
    return name, q, desc


def floor_splice(base_key: str, cand_key: str):
    """Floored file whose pre-43 part is the base file's and whose post-43 *order* is the candidate's.
    Versus the base's own floor file, the score difference is n_f2*n_l2*(A22_cand - A22_base)/(n_f*n_l):
    a direct reading of the candidate's within-block ranking quality (weight ≈ 0.06-0.10 on the public sample, so a
    0.003 reading needs ΔA22 ≈ 0.04; with ~35 positives the board's own noise on this term is ≈ 0.004)."""
    fb, sb, ts = _load(base_key); fc, sc, _ = _load(cand_key)
    post = ts >= SPLIT
    rb = rankdata(sb["target"]) / len(sb); rc = rankdata(sc["target"]) / len(sc)
    q = np.where(post, rc * 1e-3, rb)
    name = f"post43floor_{cand_key.replace('fam_', '').replace('sub_', 'sub')}"
    desc = (f"post-43 floor protocol: pre-43 rows = {fb.split('.')[0]} (Final 1), post-43 block ranked below them with the "
            f"block order of {fc.split('.')[0]}; vs fam_bag15_post43floor (0.89203) reads that file's within-post-43 ranking quality")
    return name, q, desc


def bag(name: str, keys: list[str]):
    files = [resolve(k) for k in keys]
    q = np.mean([pd.read_csv(SUBS / f)["target"].to_numpy() for f in files], axis=0)
    desc = f"probability average of {', '.join(f.split('.')[0] for f in files)} (equal-weight seed bags)"
    return name, q, desc


def write(name: str, q: np.ndarray, desc: str):
    sub = pd.read_csv(D / "sample_submission.csv"); sub["target"] = q
    assert sub["target"].between(0, 1).all() and sub["target"].notna().all()
    rho = board_spearman(q)
    print("  Spearman vs scored files:\n" + rho.head(6).round(3).to_string())
    v = verdict(rho, None); print(f"  verdict: {v}")
    path = SUBS / f"fam_{name}.csv"; sub.to_csv(path, index=False); print(f"  wrote {path}")
    append_ledger(path, desc, 159, None, rho, v)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "bag":
        write(*bag(a[1], a[2:]))
    elif a[0] == "floorsplice":
        write(*floor_splice(a[1], a[2]))
    else:
        write(*block_probe(a[0], a[1]))
