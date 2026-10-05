# Step 1 — Baseline, graph features, honest CV, first submission

Scope of this step: options **A** (GBDT backbone) and **B** (structural graph features) from the
brainstorm, on top of a **rolling-origin time CV** protocol. Output is a first submission and the
infrastructure every later step (drift correction, GNN, collective classification) reuses.

Hard rule: only the four files from `download_competition()` are ever read. No public Elliptic files.

---

## Status (2026-09-26)

Done and executed in `main.ipynb` sections 5–10b: environment, `validation.py`, `graph_features.py`,
`models.py`, rolling CV, raw baselines, period-identifier removal, graph features with parity
check, adversarial validation, edge-shuffle ablation, rounds/weight comparison, three submission
files in `submissions/`, ledger in `results.md`. Not done: uploading to Kaggle (manual) and filling
the public LB column.

Findings that changed the plan:
- Six raw columns (`feat_100/101/103/136/137/139`) are period identifiers (near-linear in
  `time_step`, 99% of test values outside the train range). Removed from every model.
- No graph-feature group beats the cleaned raw set on rolling CV; shuffled edges score the same
  as the real graph. Graph features are recorded as an ablation, not used in the submission.
- `scale_pos_weight` and early stopping on the recent tail both lose AUC under drift. Fixed 300
  rounds, unweighted, is the primary model (CV 0.982 vs 0.973).

---

## 0. Environment (once)

Machine: Apple Silicon, 26 GB RAM, 15 cores, `.venv` on Python 3.9.6 with numpy 2.0.2 / pandas 2.3.3.

1. `brew install libomp` — LightGBM and XGBoost wheels need it on macOS; currently missing.
2. Install into `.venv` (last versions that support Python 3.9):
   `scikit-learn<1.7  lightgbm  xgboost  networkx<3.3  scipy  matplotlib  seaborn  tqdm`
   Pin these in a `requirements.txt` so the GitHub repo linked from the report is reproducible.
3. Update the Environment table in `instructions.md` (the "No ML libraries yet" line).

Fallback if libomp is a problem: `sklearn.ensemble.HistGradientBoostingClassifier` needs nothing extra.

## 1. Code layout

Keep `main.ipynb` as the narrative notebook, move reusable code into small modules so later steps
(GNN, drift) import the same functions instead of copy-pasting cells:

| File | Contents |
|---|---|
| `validation.py` | rolling-origin fold generator, `evaluate()` that returns pooled AUC, mean per-step AUC, PR-AUC, per-step AUC table |
| `graph_features.py` | build split-restricted graph from `edges`, all feature builders in section 3 |
| `main.ipynb` sections 5–9 | EDA additions, baseline, graph features, adversarial validation, final fit + submission |

## 2. Validation protocol (before any model)

- Folds by `time_step`, expanding window, validation block of 7 steps (mirrors the 14-step test gap):
  `train 1–14 → val 15–21`, `train 1–21 → val 22–28`, `train 1–28 → val 29–35`.
  Only **labelled** rows enter the loss and the metric; unlabelled rows are still used to build graph features.
- Report per fold: pooled ROC AUC (competition metric), mean of per-step AUC, PR-AUC, and the per-step AUC row.
  The gap between pooled AUC and mean per-step AUC is the cross-step calibration diagnostic that step 2 (drift) will target.
- Per-step fraud rate table for steps 1–35 goes into the notebook now; it is a report figure and it justifies the protocol.
- Fixed seeds; every CV number is a mean over 3 seeds once models are cheap enough (LightGBM is).

## 3. Models for A (tabular baseline), raw 165 features only

Train on labelled rows of the training folds only. Three models, all through the same `evaluate()`:

1. Logistic regression (standardised features) — the floor, required by the report's model range.
2. Random Forest (500 trees, `class_weight="balanced_subsample"`) — the literature's strongest Elliptic baseline.
3. LightGBM (`scale_pos_weight`, early stopping on the fold's validation AUC, learning rate ~0.03) — the workhorse.

Record fold results in a results table cell (model × fold × metric). This table is the control for everything after.

## 4. Graph features for B

Build one directed graph per split from `edges`, keeping only edges whose endpoints are in that split
(the instructions confirm there are zero cross-split edges, so this loses nothing).

Features per node, identical code path for train and test:

| Group | Features |
|---|---|
| Degree | in-degree, out-degree, total, in/out ratio |
| Coverage | **unlabelled-neighbour count** = neighbours with NaN label (train) / dangling endpoints (test); also as a share of degree |
| 1-hop aggregates | mean, max, std of the top-20 raw features (by LightGBM importance from section 3) over predecessors and over successors, computed separately |
| 2-hop aggregates | mean of the same top-20 over the 2-hop neighbourhood |
| Deltas | node value minus 1-hop mean for the top-20 |
| Structure | weakly-connected component size, DAG depth (longest path from a source) and height (longest path to a sink), PageRank within the split graph |

Parity variants: compute the aggregate groups **twice** on train — once on the full train graph (all 141,772 nodes)
and once on the labelled-only train subgraph — because test contains only labelled nodes. Section 5 decides which variant survives.

Verify first: every dangling edge (endpoint missing from both splits) has its other endpoint in **test**, never train.
That confirms the coverage feature has the same meaning on both sides.

## 5. Adversarial validation (parity check)

- Fit LightGBM to distinguish train rows from test rows, 5-fold stratified (random folds are fine here; the target is split identity).
- Run three times: raw features only, raw + full-graph variant, raw + labelled-only variant.
- AUC near 0.5 is the goal. If a variant pushes AUC up materially, look at the top importances and drop or replace those features.
- Keep the result table; it is a report exhibit for section 2.1.

## 6. A + B model and ablations

- LightGBM on raw + surviving graph features, same folds, same seeds. Compare against the section-3 table.
- Quick **edge-shuffle ablation**: rewire `edges` randomly within each split (degree-preserving is nicer, plain shuffle is enough for now),
  recompute graph features, retrain. If CV AUC barely moves, the graph is not contributing yet. Record either way.
- Feature importance plot for the top-30 features; note how many are graph features.

## 7. Final fit and first submission

- Fit on all labelled rows of steps 1–35 with the boosting-round count chosen by CV (median of the folds' best iterations).
- Predict `test`, write `submissions/sub_01_lgbm_graph.csv` with `index` and `target`.
- Sanity checks before upload: 15,329 rows, `index` identical to `sample_submission["index"]`, targets in [0, 1], no NaN.
- Upload by hand at the Submit Predictions page (the CLI cannot submit with the `KGAT_` token). Three submissions per day.
- Log in a `results.md`: submission name, CV pooled AUC, CV mean per-step AUC, public LB score. This is the tuning ledger for the whole project.

## Verification checklist for this step

- [x] `pip list` shows sklearn, lightgbm, xgboost, networkx importable inside `.venv`
- [x] Rolling-origin folds print their step ranges and labelled row counts; no validation step ever appears in its own training set
- [x] Section-3 results table filled for all three models × three folds
- [x] Dangling-edge check passes (all dangling edges touch test only)
- [x] Adversarial validation reported by feature group; single-feature rule (AUC > 0.95) replaces the planned delta rule because the raw floor is already 0.997
- [x] A + B CV table and edge-shuffle ablation recorded (shuffled edges score the same: graph features add no signal on this cut)
- [ ] Submission files pass the four sanity checks (done); public LB scores still to be filled in `results.md` after manual upload

## What this step deliberately leaves out

Drift and prior-shift correction (step 2), GNN (step 3), collective classification and blending (step 4),
self-supervised pretraining (step 5). Do not tune against the public leaderboard; it is ~4.6k rows.

## What changed during execution (deviations from the plan above)

- **Six raw features are period identifiers** (`feat_100/101/103/136/137/139`): single-feature
  adversarial AUC ≥ 0.99, monotone in `time_step`, 99% of test values out of range for three of them.
  They are removed from every model; the all-165 model is kept only as a reference and a
  controlled leaderboard comparison.
- **The planned pruning rule (delta over raw AV AUC) was moot** because the cleaned raw features
  already reach 0.997. Replaced by one rule for raw and graph columns alike: drop any column whose
  single-feature tree adversarial AUC exceeds 0.95. That drops component size (and its log) among
  the graph features, nothing else.
- **PageRank is normalised within its component** so it does not carry the time step's size.
- **Graph features did not improve rolling CV** in any variant, and randomly rewired edges scored
  the same. Consistent with arXiv 2604.19514. They stay in the codebase for the GNN comparison and
  the report's ablation, not in the submission.
- **Final-model rule fixed before seeing the leaderboard:** best CV among configurations without
  period identifiers.
