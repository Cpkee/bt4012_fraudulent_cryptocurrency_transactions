# AGENTS.md — BT4012 fraudulent cryptocurrency transactions

Kaggle competition `bt-4012-competition-2026`: score Bitcoin transactions by probability of being
illicit; metric = ROC AUC on the pooled test set. Deadline for report and final selection:
**25 Oct 2026**. Priority set by the user: the leaderboard first, the report once settled.

## Rules for any session (read before touching anything)

1. **Evidence comes from the Kaggle files only.** Papers may be cited for methods; never as facts
   about this data. Say "predicted fraud drops at step 43", not "the dark-market shutdown".
2. **One session at a time.** Several sessions have collided on this repo. Check `git status` and
   file mtimes before writing; never `cat >` a module that exists; extend, do not replace.
3. **Reference model = `sub_06_lgbm_clean_bagged5.csv`** (public 0.94374): LightGBM, 31 leaves,
   300 rounds, lr 0.03, 80% rows / 60% columns, unweighted, 5-seed bag, 159 cleaned features.
   `families.py ref` reproduces it exactly.
4. **Upload gate = distinctness, not CV.** A file is uploaded only if its Spearman rank
   correlation with every scored file is < 0.98 (`families.py` checks this). Historical CV is a
   sanity check (3-fold ≥ 0.95) and report material; it has been wrong about the test period on
   every large decision (see "What we learned").
5. **One change per upload**, 3 uploads/day. Reading rule vs the reference: > +0.005 signal,
   within ±0.005 tie (simpler model wins), < −0.005 rejected. Public noise ≈ 0.005–0.006.
6. **Ledger**: every file goes in `results.md` before upload with CV numbers and the nearest scored
   file; the public score is filled in after. `BOARD` in `families.py` mirrors scored files.
7. Never use `time_step` as a model input; it is for splitting and per-step statistics only.

## Data facts (measured)

- train: 141,772 rows, steps 1–35, 165 features, 78% unlabelled (31,235 labelled, 11.7% illicit);
  test: 15,329 labelled rows, steps 36–49. Edge list 234,355 edges; edges never cross steps, so
  there are no train–test edges. Test holds labelled nodes only; 46,668 test-period nodes exist
  as ids in the edge list without features.
- Labelled fraud rate per step swings 0.43%–35.97% inside train. From step 43 on, every model we
  built scores almost everything as licit (mean predicted rate 0.022 vs 0.085 for 36–42); the data
  cannot say why.
- Six columns (`feat_100, 101, 103, 136, 137, 139`) identify the time period (near-linear in step,
  99% of test values out of training range). Removed everywhere.

## Feature engineering: what was tried and what it did

| technique | where | historical CV | public board | status |
|---|---|---|---|---|
| Remove 6 period identifiers (adversarial validation) | all subs | neutral | +0.002 when kept (sub_02 vs sub_01), inside noise | kept removed; `fam_all_time` (kept + time_step) ranks 0.995 like sub_06, unreadable |
| Within-step percentile rank of 9 drifting continuous features (`drift.StepNormalizer`) | sub_04, 05, 07 | +0.02 horizon fold | −0.002 vs sub_06 | not in reference; re-test on the recent-window base |
| Graph topology: in/out degree, unlabelled-neighbour counts, DAG depth/height, component-relative PageRank (`graph_features.py`) | step 1, probe | no gain; rewired edges score the same | ranks 0.994 like reference, unreadable | rejected |
| Neighbour feature aggregates (mean/max/std over predecessors, successors, 1- and 2-hop; deltas) | step 1 | −0.008 | not uploaded | rejected |
| Local block only (drop the 72 aggregated columns 94–165) | probe | −0.023 far windows | not uploaded (distinct, 0.91) | rejected on CV; optional board read |
| Period identifiers back as step-median deviations | step 2 | −0.004 | – | rejected |
| Recency sample weights (half-life 5/10/20), window start 5/8/12 | step 2 | monotone loss | see below: window helps on the board | **CV was wrong here** |
| Adversarial importance weights (3 strengths) | probes | ≈ 0 | −0.003 (hard) | no effect |
| Self-training on test rows / pseudo-labels from unlabelled train rows | probes | −0.018 / −0.024 | – | rejected |
| Isolation-Forest anomaly blend 10–30% | step 2, probe | −0.007 and worse | – | rejected |
| GraphSAGE embeddings → LightGBM (in-sample) | probe | −0.065 (label leakage) | – | rejected as built; unsupervised version untested |
| GraphSAGE / MLP as models (`gnn.py`, strict inductive) | step 3 | MLP > SAGE; rewired edges > real; all ≈ 0.03 below trees | – | graph carries no forward signal |
| LightGBM tuning (7 leaves, 600 rounds, lr 0.05, no row subsampling) | step 3 | +0.027 horizon fold | −0.003 vs sub_06 | **CV was wrong here**; not in reference |
| Score-level: within-step rank, EM prior, BBSE, step-mean equalisation | step 2, sub_08 | all hurt or neutral | +0.001 (sub_08) | rejected |
| Model families: RF, ExtraTrees, logistic regression, shallow XGBoost | step 5 | far below LightGBM | RF 0.9285 | rejected |
| **Training window = last 14 steps (22–35) only** | `fam_recent14` | −0.016 horizon fold | **0.94836, +0.0046 vs sub_06** | **best result; being mapped** |

## Best result and standing

- Best public score: **fam_recent14.csv = 0.94836** (sub_06 recipe trained on steps 22–35).
  Before it, sub_03 = 0.94510 (single seed of the reference recipe; seeds spread ±0.003).
- Board (2026-09-29): 20 other entries from 0.9475 to 0.9636; recent14 would rank 19th. The
  brief gives the lowest-ranked participant 0 points, so staying off the bottom matters.
- Finals must be selected by hand on Kaggle (two files; the better private score counts).
  Current choice: fam_recent14 + sub_06, to be revised as the window map comes in.

## What we learned (the through-line for the report)

- Under drift, capacity that fits the training period fails ahead of it: neighbour aggregates,
  deep trees and long history all lost on far-horizon validation.
- But the test period is unlike any historical window: the changes that improved the 14-step
  horizon fold (normalisation, shallow trees) scored *lower* on the board, and the change it
  rejected most (recent window) scored highest. Historical CV cannot see steps 43–49.
- The public board is noisy (±0.006) and compressed; files with rank correlation ≥ 0.98 to a
  scored file return that file's number plus noise.

## Direction (current)

1. **Map the training window** on the board: `fam_recent7` (29–35), `fam_recent10` (26–35),
   `fam_recent21` (15–35) are built and pass the gate; `fam_recency_hl5` (soft weights) next.
2. **Then drift-aware feature engineering on the best window, one change per upload:**
   within-step rank normalisation re-test; **two-stage neighbour predictions** (model A scores
   all rows; mean/max predicted probability of incoming and outgoing neighbours → model B;
   labelled-only neighbours on both sides for parity); trailing-period detrending (feature minus
   median of the preceding periods); stability-based feature selection over recent steps.
3. Re-check depth and normalisation on the short window (they may behave differently with less
   history).
4. Freeze by ~18 Oct: 10-seed bag of the best recipe; select two finals; write the
   student-number notebook (`eXXXXXXX.ipynb`, end to end from raw files to CSV) and the report
   from `report_notes.md`.

## Future implementation notes

- New variants go in `families.py` (`FAMILIES` dict; `lgbm_recent`, `lgbm_recency_weighted`
  patterns) so the CV sanity check, distinctness gate, file naming and ledger row are automatic.
- Two-stage neighbour features: build model-A predictions out-of-fold in time (predict a step
  from a model trained on earlier steps) or the trees will over-trust them; test neighbours are
  the labelled test rows only. `graph_features.period_edges` gives per-period edges.
- GNN outputs must be precomputed in a PyTorch-only process (`gnn_cache.py`): LightGBM and
  PyTorch hang together in one process on macOS.
- Full notebook execution (`main.ipynb`, 55 cells) takes ~2 h; it re-runs every experiment.

## File map

| file | role |
|---|---|
| `main.ipynb` | full experiment notebook, sections 5–16 (steps 1–3), executed |
| `validation.py` | rolling-origin folds, horizon fold, per-step AUC, gap metric |
| `models.py` | LightGBM factories (`LGBM_PARAMS` = reference, `LGBM_TUNED`), RF/logreg, adversarial validation, search space |
| `drift.py` | recency weights, `StepNormalizer`, EM/BBSE prior estimation, score adjustment |
| `graph_features.py` | per-period topology and neighbour aggregates, edge rewiring |
| `gnn.py`, `gnn_cache.py` | inductive GraphSAGE/MLP and the cache builder |
| `probes.py` | step-4 probe factory (sub_07 reference; superseded by `families.py`) |
| `families.py` | current campaign: sub_06 reference, distinctness gate, ledger rows |
| `results.md` | ledger of every file with CV and public score |
| `report_notes.md` | methodology and results write-up with every table; `figures/` diagram |
| `plan_step1..5.md` | step plans with status blocks; `instructions.md` competition reference |
| `submissions/` | every CSV; `artifacts/` (gitignored) CV tables and caches |
