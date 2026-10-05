# BT4012 — Fraudulent Cryptocurrency Transactions

Working reference for the `bt-4012-competition-2026` Kaggle competition.

> **Scope of this document.** *Data* and *Environment* were measured directly from the
> competition files and the Kaggle REST API — reproducible by re-running the notebook.
> *Background*, *Deliverables* and *Evaluation* are transcribed from the competition
> Overview page, which is not reachable programmatically
> (see [Remaining gaps](#remaining-gaps)).

---

## Background
Bitcoin transactions can be used to move funds for illicit purposes, from money laundering to funding illegal marketplaces. In this competition, you'll build a machine learning model that classifies transactions on the Bitcoin network as licit or illicit, using a graph-structured dataset where each transaction is described by a set of anonymized features and connected to other transactions through payment flows.

Only a fraction of transactions carry a known label, so part of the challenge is making the most of a large pool of unlabeled data alongside a highly imbalanced set of confirmed illicit cases. Strong solutions will need thoughtful feature engineering, sound handling of class imbalance, and a model that generalizes to transactions from future time periods it has never seen. This task mirrors a real-world anti-money-laundering problem: catching bad actors early, without drowning investigators in false alarms.

## Environment

| Item | Value |
|---|---|
| Interpreter | `.venv/bin/python` (Python 3.9.6) |
| Jupyter kernel | `Python (bt4012)` — select it in the notebook's kernel picker |
| Notebook | `main.ipynb` |
| Data loader | `kaggle_data.py` |
| Data cache | `~/.cache/bt4012/bt-4012-competition-2026` (outside the repo) |

Installed from `requirements.txt` (Python 3.9 pins): `scikit-learn`, `lightgbm`, `xgboost`,
`networkx`, `scipy`, `matplotlib`, `seaborn`, plus `nbformat`/`nbconvert` to execute the
notebook from the shell. LightGBM and XGBoost need `brew install libomp` on macOS.

| Module | Role |
|---|---|
| `validation.py` | rolling-origin folds by `time_step`, pooled / per-step AUC, PR-AUC |
| `graph_features.py` | per-period topology and neighbour-aggregate features from the edge list |
| `models.py` | `fit_predict` factories (logreg, RF, LightGBM), adversarial validation and pruning |
| `drift.py` | recency weights, per-step rank/z/deviation normaliser, EM and BBSE prior estimation, score adjustment |
| `results.md` | submission ledger: CV numbers next to public LB |
| `report_notes.md` | methodology and results write-up for the report, with every table |
| `plan_step1.md`, `plan_step2.md` | the step plans, each with a status block at the top |
| `artifacts/` | CV tables, importances, cached graph features (gitignored) |
| `submissions/` | submission CSVs |

`.vscode/settings.json` sets `.venv` as the default interpreter and hides
`/usr/bin/python3` from the kernel picker, so the notebook cannot silently attach to
system Python (a missing-module failure that is easy to misread).

## Credentials

`.env` (gitignored) holds a single value:

```
KAGGLE_KEY=KGAT_...
```

**Keep the `KGAT_` prefix.** This token type authenticates only as
`Authorization: Bearer <token>`; without the prefix the API returns 401.

The `kaggle` and `kagglehub` packages **cannot** use it. Both authenticate with HTTP
Basic (username + the 32-character `key` from a legacy `kaggle.json`), and neither
supports Bearer tokens outside a Kaggle-hosted notebook. Verified against
`competitions/list`:

| Scheme | Result |
|---|---|
| Basic, username + token | 401 |
| Basic, prefix stripped | 401 |
| **Bearer, prefix intact** | **200** |

`kaggle_data.py` therefore calls the REST API directly. Consequences:

- `KAGGLE_USERNAME` is unused. If it is set alongside `KAGGLE_KEY`, any call into
  `kagglehub` will take the Basic path and fail with 401 — remove it to avoid the trap.
- `kaggle competitions submit` will not work. Submit through the website.

## Getting the data

```python
from kaggle_data import download_competition, list_files
DATA_DIR = download_competition("bt-4012-competition-2026")
```

Downloads a 115 MB zip, extracts ~515 MB, and caches via a `.complete` marker.
Re-running is a no-op; pass `force=True` to refresh.

---

## Data

Four files. Train and test are disjoint in both transaction ID and time.

| File | Shape | Notes |
|---|---|---|
| `train.csv` | 141,772 × 168 | `txId`, `time_step`, `feat_1`…`feat_165`, `label` |
| `test.csv` | 15,329 × 168 | same but `label` replaced by `index` |
| `sample_submission.csv` | 15,329 × 2 | `index`, `target` — all `0.5` |
| `txs_edgelist.csv` | 234,355 × 2 | `txId1`, `txId2` — directed transaction graph |

All 168 columns are numeric (166 `float64`, 2 `int64`) with **no missing values except
in `label`**.

### The label is mostly missing

| `label` | Rows | Share |
|---|---|---|
| `NaN` (unlabelled) | 110,537 | 77.97% |
| `0.0` (licit) | 27,591 | 19.46% |
| `1.0` (illicit) | 3,644 | 2.57% |

**Only 31,235 of 141,772 training rows are usable for supervised learning.** The other
78% carry features and graph edges but no target. Either drop them or exploit them
semi-supervised — the edgelist is what makes the second option viable.

Among labelled rows the fraud rate is **11.67%**.

### The split is temporal

- `train.csv`: `time_step` 1–35
- `test.csv`: `time_step` 36–49
- Overlap: **none**, in either `time_step` or `txId`

**Do not use random K-fold cross-validation.** It leaks future information across folds
and will inflate local scores relative to the leaderboard. Split by `time_step` — for
example, train on 1–28 and validate on 29–35 to mimic the real forward gap.

### Class balance drifts hard over time

Per `time_step`, among labelled rows:

| | Min | Max |
|---|---|---|
| Labelled rows | 206 | 2,147 |
| Fraud rate | **0.43%** | **35.97%** |

An **83× swing** in base rate across time steps. A model tuned on the pooled 11.67%
average will be miscalibrated for individual periods, and the test period's true rate is
unknown. This is the single most important modelling constraint in the dataset.

### The transaction graph

`txs_edgelist.csv` is a payment-flow graph over **transactions only** — wallets and
addresses never appear as nodes. An edge `txId1 -> txId2` means an output of `txId1` was
spent as an input of `txId2`.

A transaction appears once per spent output and once per consumed input, so a transaction
with many inputs (an exchange sweeping customer deposits) or many outputs (a payout
distributing to many recipients) generates correspondingly many edges. This is why some
transactions recur dozens or hundreds of times in the file.

**Degree is not the input/output count.** In-degree counts predecessor transactions
*present in the observed graph*; out-degree counts successors. The true number of Bitcoin
inputs and outputs is captured separately among the 165 local features. Degree therefore
reflects observed connectivity, not raw transaction shape.

Wallet identity is unrecoverable by design — feature engineering has to work with
transaction-level structure alone.

#### Verified properties

| Property | Measured |
|---|---|
| Edges | 234,355 |
| Nodes appearing in the edgelist | 203,769 |
| Both endpoints present in train or test | 175,918 (75.1%) |
| Acyclic (Kahn's algorithm) | **confirmed — full topological sort** |
| Edges joining transactions in the same `time_step` | **175,918 / 175,918 = 100.0000%** |
| Isolated nodes (degree 0) | **0 in train, 0 in test** |

Degree, restricted to each split:

| | In-degree mean | In max | Out-degree mean | Out max |
|---|---|---|---|---|
| train (141,772 nodes) | 1.15 | 284 | 1.15 | 472 |
| test (15,329 nodes) | 2.02 | 241 | 1.22 | 99 |

Every transaction has at least one edge, so graph features are defined for every row —
no fallback needed for isolated nodes. About 25% of edges have an endpoint outside both
splits, so features must still tolerate dangling references.

#### The consequence that matters: no train–test edges

Because connected transactions always share a `time_step`, and the splits partition
`time_step` (1–35 vs 36–49), **the graph contains zero train-test edges**:

| | → train | → test |
|---|---|---|
| **train →** | 163,194 | 0 |
| **test →** | 0 | 12,724 |

The train and test graphs are entirely disconnected components. Therefore:

- **Label propagation from train into test is impossible.** No semi-supervised technique
  can carry a known label across the boundary through the graph — there is no path.
- Graph features must be **structural and computed within each split independently**
  (degree, local neighbourhood feature aggregates, component size, depth in the DAG),
  never label-derived from neighbours.
- The 110,537 unlabelled train rows are still useful — they thicken the train graph, so
  neighbourhood aggregates over them are legitimate features — but they help only by
  improving the learned feature→label mapping, not by reaching test nodes.
- The upside: a whole class of leakage is structurally impossible here.

Note the test graph is sparser in absolute terms (12,724 edges over 15,329 nodes) yet has
a higher mean in-degree (2.02 vs 1.15), so neighbourhood-size distributions differ between
splits — worth checking before relying on degree-sensitive features.

Since edges never cross `time_step`, **`time_step` cannot be used to order an edge's
endpoints.** Bitcoin's causal ordering guarantees the source precedes the destination in
real time, but the dataset's coarse snapshot does not record that ordering.

---

## Submission

`index` must match `test.csv` row for row — verified identical to
`sample_submission["index"]`, range 0–15,328.

```python
submission = sample_submission.copy()
submission["target"] = model.predict_proba(X_test)[:, 1]
submission.to_csv("submission.csv", index=False)
```

Upload at the competition's **Submit Predictions** page. The CLI cannot submit with a
`KGAT_` token.

**Limit: 3 submissions per day.**

### Evaluation

Area under the ROC curve, between the predicted probability and the observed target.
Submit **probabilities**, not thresholded labels.

## Leaderboard

11 entries as of 2026-09-16, via the API. These are **public** leaderboard scores,
computed on the remaining 30% of the test set; grading uses the **private** leaderboard
(70%). With roughly 4,600 rows behind each public score, expect meaningful shuffling
between the two — do not tune hard against the public number.

| | Score |
|---|---|
| Top | 0.95333 |
| Median | 0.94802 |
| Bottom | 0.92119 |

A narrow 0.032 spread. Entry names are NUS student numbers (`e1122376`, `E1303010`, …),
which suggests individual rather than team entries — not a stated rule, just an
observation from the API.

---

## Deliverables

Two graded components, 15 points total.

### 1. Final ranking — 10 points

Scored from the **private** leaderboard — **70% of the test set** — on a non-linear
curve:

```
Final Score = 10 × (your AUC / highest AUC) ^ a        (a is a scaling parameter, e.g. a = 5)
```

Worked example from the brief: AUC 0.92 against a top of 0.95 with a = 5 gives
10 × (0.9684)^5 ≈ **8.5177**.

What that curve implies, using the current public top of 0.95333:

| Your AUC | a = 1 | a = 3 | a = 5 |
|---|---|---|---|
| 0.95333 (top) | 10.00 | 10.00 | 10.00 |
| 0.94802 (median) | 9.94 | 9.83 | 9.73 |
| 0.92119 (bottom) | 9.66 | 9.02 | 8.42 |
| 0.90000 | 9.44 | 8.41 | 7.50 |
| 0.85000 | 8.92 | 7.09 | 5.64 |

**Budget your effort accordingly.** Even at a = 5, +0.001 AUC near the top is worth about
**0.05 points out of 15**. Closing the gap from 0.90 to the leaderboard pack is worth
~2.2 points; grinding from 0.950 to 0.953 is worth ~0.16. The report is 5 points — a third
of the total — for far less work than the last decimal of AUC.

> **Inconsistency worth clarifying with the teaching team.** The brief says the
> lowest-ranked participant receives 0 points, but the formula gives 8.42 for the current
> bottom score. The two rules disagree; which one governs changes how much the ranking
> component is actually worth.

### 2. Report — 5 points

| Requirement | Detail |
|---|---|
| Deadline | **25 Oct 2026, 23:59 Singapore time** |
| Filename | Your NUS student number, e.g. `e1234567.pdf` |
| Length | Maximum 4 pages |
| Code | Must be hosted on GitHub, with the repository link in the report |

Must be understandable to a reader unfamiliar with the competition. Required sections:

1. **Introduction** — the competition objective and its relevance to fraud analytics; the
   nature of the fraudulent Bitcoin transaction detection problem and why it matters; your
   high-level approach, goals, and key strategies.
2. **Methodology** — the main section, in three subsections:
   - **2.1 Data Preprocessing and Feature Engineering** — cleaning, transformation, missing
     values, graph-based features (node degree, neighbourhood aggregates), feature
     selection/extraction, augmentation. Justify choices with domain insight or experimental
     evidence.
   - **2.2 Model Development and Validation** — models implemented, training, validation,
     cross-validation, hyperparameter tuning, class imbalance, overfitting.
   - **2.3 Model Exploration and Final Strategy** — the range of models explored (logistic
     regression, decision trees, ensembles, graph neural networks), how you compared them,
     and why you chose the final submission.
3. **Results and Discussion** — performance using appropriate metrics (accuracy, precision,
   recall, F1, ROC AUC), tables or charts where helpful, strengths and limitations, and
   reflection on model behaviour.
4. **Conclusion and Reflection** — main findings, what you learned, what worked, what could
   be improved, and how this informs your understanding of AI-driven fraud detection.

Layout is free as long as these elements are covered.

### Notes for the report, from this repo

The measured findings above map onto specific required subsections — use them:

- **2.1** — the 77.97% unlabelled rate is a genuine preprocessing decision (drop vs.
  semi-supervised), and `txs_edgelist.csv` is exactly the graph-feature source the brief
  names (node degree, neighbourhood aggregates). The zero-train-test-edges finding is the
  strongest single justification you can give for *which* graph features you built: it
  rules out label propagation on structural grounds, not preference. That every node has
  degree ≥ 1 means no isolated-node fallback is needed — worth stating.
- **On "making the most of unlabeled data"** (from the Background) — the honest version is
  that unlabelled train rows improve the learned feature→label mapping and thicken train
  neighbourhood aggregates, but cannot reach test nodes through the graph. Saying so, with
  the edge counts as evidence, is a stronger result than an unexamined semi-supervised
  claim.
- **2.2** — the disjoint temporal split is your justification for time-based rather than
  random cross-validation. The 83.7× drift in base rate across time steps is concrete
  evidence for how you treat class imbalance.
- **Results** — report ROC AUC to match the competition metric, but the brief invites
  precision/recall/F1 too, which are more informative at a 11.67% positive rate.

## Remaining gaps

The Overview page returns **404** to every programmatic request — with a Bearer token,
without auth, and via external fetch. Kaggle's web UI authenticates by browser session
cookie, which an API token cannot supply, and this is a private class competition. The
REST API exposes file listings, leaderboard, and submissions, but not page prose. The
sections above were transcribed by hand.

Still unconfirmed:

- [ ] **Kaggle submission close date** — the report is due 25 Oct 2026, but the date the
      leaderboard freezes was not stated
- [ ] **Team rules** — maximum team size, or whether entries are individual (leaderboard
      names suggest individual)
- [ ] **External data and pretrained model policy**
- [ ] **The value of `a`** — given as "e.g., a = 5", so not necessarily final
