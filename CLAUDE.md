# CLAUDE.md — BT4012 fraudulent cryptocurrency transactions

Kaggle competition `bt-4012-competition-2026`: score Bitcoin transactions by probability of being
illicit; metric = ROC AUC on the pooled test set. Deadline for report and final selection:
**25 Oct 2026**.

**Objective (set by the user, 2026-10-03): keep improving the public leaderboard score, continuously,
until the deadline.** Do not plan around "freezing" or hand-in; the notebook and report are
maintained alongside, not instead of, the search for the next gain. Every day with unused upload
slots is a day of lost information: there should always be readable candidates built and gated.

## Rules for any session (read before touching anything)

1. **Evidence comes from the Kaggle files only.** Papers may be cited for methods; never as facts
   about this data. Say "predicted fraud drops at step 43", not "the dark-market shutdown".
2. **One session at a time.** Several sessions have collided on this repo. Check `git status` and
   file mtimes before writing; never `cat >` a module that exists; extend, do not replace.
3. **Current best / reference for new probes = `fam_B15_nbfeat.csv`** (public 0.96069,
   2026-10-08: the 0.95849 two-stage plus the means of the top-10 raw features over labelled
   in- and out-neighbours as 20 extra model-B columns; before it random-OOF scoring 0.95849,
   time-ordered OOF 0.95621, 7 leaves 0.95504, plain two-stage 0.95260).
   The base recipe underneath it is `sub_06` (LightGBM, 31 leaves, 300 rounds, lr 0.03, 80% rows /
   60% columns, unweighted, 5-seed bag, 159 cleaned features; `families.py ref` reproduces it).
4. **Upload gate = distinctness, not CV.** A file is uploaded only if its Spearman rank
   correlation with every scored file is < 0.98 (`families.py` checks this). Historical CV is a
   sanity check (3-fold ≥ 0.95) and report material; it has been wrong about the test period on
   most large model decisions but right on the one large feature decision (see "What we learned").
5. **One change per upload**, 3 uploads/day. Reading rule vs the file it changes: > +0.005 signal,
   within ±0.005 tie (simpler model wins), < −0.005 rejected. Noise: a 5-seed bag re-seeded moves
   the public score by ≈ 0.001 (measured 2026-10-02); differences between *different* models of
   0.002–0.003 are therefore more meaningful than the ±0.005 band assumed earlier.
6. **Ledger**: every file goes in `docs/results.md` before upload with CV numbers and the nearest scored
   file; the public score is filled in after. `BOARD` in `families.py` mirrors scored files.
7. Never use `time_step` as a model input; it is for splitting, windows, weights and per-step
   statistics only.
8. **Submission standard (set by the user 2026-10-02): the code behind any submission lives in a
   notebook.** `final_model.ipynb` is the self-contained end-to-end notebook (raw files →
   preprocessing → feature engineering → validation → prediction → CSV) for the current best
   model, with a `CONFIG` cell for variants; it imports no project modules. When the best model
   changes, update the notebook's `CONFIG` (or its code, if the pipeline changes) and re-execute
   it; it checks itself against the uploaded CSV. `families.py` remains the fast experiment
   harness, but a model is not final until the notebook reproduces it. Rename the notebook to
   the NUS student number (`eXXXXXXX.ipynb`) for the final hand-in.

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

## The current model (0.96069), in order

1. 159 cleaned features, untouched (no scaling, no engineered columns).
2. Training rows = labelled rows of the last 14 steps (22–35), ~11,700 rows.
3. **Model A** (sub_06 recipe) scores every transaction. Training rows are scored out-of-fold by
   5-fold random splits *within the window* (each scorer saw 80% of the window, including the
   row's own period); test rows by a model trained on the whole window. Time-ordered OOF
   (step s scored by a model on steps < s) was the previous protocol: 0.95621 vs 0.95849.
4. **Six neighbour numbers** per transaction from the edge list: mean, max and count of model-A
   scores over the transactions that paid into it, and separately over those it paid out to.
   Only labelled rows count as neighbours on both sides (parity with the labelled-only test file).
5. **Twenty neighbour feature means** (2026-10-08, +0.0022): the mean of each of the top-10 raw
   features by step-1 gain (`feat_53, 55, 47, 90, 5, 2, 163, 132, 41, 14`) over the same labelled
   in-neighbours and, separately, out-neighbours (`two_stage(nb_feats=10)`).
6. **Model B** (LightGBM, 15 leaves, 300 rounds, lr 0.05, no row subsampling) on 159 features +
   6 neighbour numbers + 20 neighbour feature means → final probability.
7. Average of 5 seeds. Everything lives in `families.py` (`two_stage`) and `final_model.ipynb`.

## Feature engineering and modelling: what was tried and what it did

| technique | historical CV | public board | status |
|---|---|---|---|
| Remove 6 period identifiers (adversarial validation) | neutral | +0.002 when kept, inside noise; kept+time_step ranks 0.995 like sub_06 (unreadable) | kept removed |
| Within-step percentile rank of 9 drifting features (`drift.StepNormalizer`) | +0.02 horizon | −0.002 vs sub_06; on recent14 ranks 0.981 (unreadable) | not used |
| Graph topology features (degree, unlabelled-neighbour counts, DAG depth, PageRank) | no gain; rewired edges same | ranks 0.994 like reference (unreadable) | rejected |
| Neighbour *feature* aggregates (mean/max/std, 1- and 2-hop, deltas) on the single-stage | −0.008 | – | rejected (step 1) |
| **Neighbour feature means inside the two-stage** (top-10 raw features over labelled in/out neighbours, 20 model-B columns) | −0.004 3-fold, −0.02 horizon | **+0.0022 → 0.96069** | **adopted; best** (2026-10-08) |
| Local block only (drop aggregated columns 94–165) | −0.023 | **−0.012 vs recent14** (2026-10-01) | rejected; board and CV agree |
| Period identifiers back as step-median deviations | −0.004 | – | rejected |
| Recency: hard window / soft weights | monotone loss | **window 14 = +0.005; half-life 5 = +0.007 vs all history** | **CV wrong; adopted** |
| Adversarial importance weights (3 strengths) | ≈ 0 | −0.003 | no effect |
| Self-training on test rows / pseudo-labels from unlabelled train rows | −0.018 / −0.024 | – | rejected |
| Isolation-Forest anomaly blend | −0.007 and worse | – | rejected |
| GraphSAGE embeddings → LightGBM (in-sample) | −0.065 (label leakage) | – | rejected as built |
| GraphSAGE / MLP as models (`gnn.py`, strict inductive) | MLP > SAGE; rewired > real; ≈ 0.03 below trees | – | graph shape carries no forward signal |
| LightGBM tuning (7 leaves, 600 rounds, lr 0.05) | +0.027 horizon | −0.003 vs sub_06 alone; **+0.0024 as model B of the two-stage** | adopted as model B |
| Score-level corrections (within-step rank, EM/BBSE prior, step-mean equalisation) | hurt or neutral | +0.001 | rejected |
| Model families: RF, ExtraTrees, logistic regression, shallow XGBoost | far below | RF 0.9285 | rejected |
| **Two-stage neighbour predictions** (model-A scores of in/out neighbours → model B) | −0.03 horizon | **+0.004 vs recent14 → 0.95260** | **adopted; best** |
| Two-stage elaborations: min/std/2-hop stats, unlabelled rows as neighbours, 21-step window, second pass (model C), model A on all history | all ≈ −0.01 to −0.05 | all rank 0.981–0.997 like the two-stage file: unreadable | **saturated**: the six basic numbers carry everything model B can use |

## Board history (public, 30% of test; noise ±0.006)

| file | change | score |
|---|---|---|
| sub_01 | cleaned raw, early stop + class weight | 0.9390 |
| sub_02 | + period identifiers kept | 0.9409 |
| sub_03 / seed 1 | cleaned raw, 300 fixed rounds, unweighted, single seed | 0.9451 / 0.9420 |
| sub_06 | same, 5-seed bag (reference recipe) | 0.9437 |
| sub_04 / sub_05 | + rank normalisation (9 / 12 features) | 0.9420 / 0.9398 |
| sub_07 / sub_08 | tuned shallow trees / + step-mean equalisation | 0.9403 / 0.9413 |
| probe_adv_weights_hard | adversarial importance weights | 0.9377 |
| fam_rf_clean | random forest | 0.9285 |
| fam_recent7 / 10 / **14** / 21 | training window = last k steps | 0.9283 / 0.9339 / **0.9484** / 0.9462 |
| fam_recent14_local | recent14 on local columns only | 0.9367 |
| fam_recency_hl5 | soft weights, half-life 5, all history | 0.9510 |
| fam_recent14_2stage | recent14 + two-stage neighbour predictions | 0.9526 |
| fam_hl5_2stage | two-stage on the soft-weighted base | 0.9523 |
| fam_recent14_2stage_seedsB | same two-stage model, seeds 5–9 (noise reading) | 0.9536 |
| fam_recent14_2stage_tunedB | + model B = tuned shallow LightGBM (7 leaves, 600 rounds) | 0.9550 |
| fam_hl5_2stage_tunedB | weighted base + shallow model B | 0.9552 |
| **fam_recent14_2stage_B15** | + model B = 15 leaves, 300 rounds | **0.9562** |
| fam_recent14_2stage_B23 | model B = 23 leaves | 0.9550 |
| fam_B15_interact / smooth / lambdarank | interaction features / edge smoothing / ranking objective | 0.9562 / 0.9446 / 0.9450 |
| fam_B15_xgb / cat / et | model B = XGBoost / CatBoost / ExtraTrees | 0.9540 / 0.9464 / 0.9178 |
| fam_B15_nb_labels / nb_mix50 | training neighbour numbers from labels / half labels | 0.9462 / 0.9542 |
| fam_B15_Aoof_random | training rows scored by random 5-fold OOF within the window | 0.9585 |
| **fam_B15_nbfeat** | + means of the top-10 raw features over labelled in/out neighbours (20 model-B columns) | **0.9607** |

Standing (2026-10-05): **6th of 20**; top 0.9681; five entries above us (0.9609–0.9681). Model-B depth
curve (under time-ordered OOF): 31 → 0.9526, 23 → 0.9550, **15 → 0.9562**, 7 → 0.9550, 5 ≈ 7.
Label curve for training neighbour numbers: 0% labels 0.9562, 50% 0.9542, 100% 0.9462 (closed).
**Current best = fam_B15_nbfeat** (0.96069, 2026-10-08, 7th on the board; `final_model*.ipynb` updated, re-execution pending). The brief gives
the lowest-ranked participant 0 points. Finals must be selected by hand on Kaggle (two files;
the better private score counts). Standing choice, revised whenever the best changes: the best
file plus the strongest file that ranks < 0.98 like it (currently `fam_B15_Aoof_random`, 0.973).

## What we learned (the through-line for the report)

- On historical windows, capacity that fits the training period fails ahead of it: neighbour
  aggregates, deep trees and long history all lost on far-horizon validation.
- The test period is unlike any historical window. Model choices that improved the horizon fold
  (normalisation, shallow trees) scored lower on the board; the change it rejected most (recent
  window) scored highest. But on the one large *feature* decision (local vs all columns) the
  board and CV agreed. Historical CV is a guide for features, not for temporal choices.
- Both real gains come from one principle: let the model lean on what is recent and on what its
  neighbours look like, through predictions rather than engineered features. The graph's usable
  content is first-order and saturated: the six neighbour numbers capture all of it.
- The public board is noisy (±0.006) and compressed; files with rank correlation ≥ 0.98 to a
  scored file return that file's number plus noise. Most "improvements" are unreadable.

## Direction (current): continuous improvement

Closed axes (each read to its peak on the board; do not re-spend slots on them): training window
(14), base weighting (flat), model A (window / history / depth all leave the ranking unchanged),
neighbour *score* statistics (saturated at the six numbers; neighbour *feature* means are the open axis since 2026-10-08), model B depth (peak at 15 leaves), feature
block (all 159), normalisation (unreadable), graph topology and GNNs, score-level corrections,
other model families.

Read 2026-10-03 on `fam_recent14_2stage_B15` (0.95621), all three closed:
- interaction features for model B (56 products/ratios of the top-8 raw features): **0.95620**,
  an exact tie — the 15-leaf trees already form the interactions they need;
- edge smoothing of the final score (λ = 0.3): **0.94462**, −0.012 — the graph helps at the
  input (neighbour scores for model B) and hurts at the output;
- lambdarank objective for model B (random groups ≤ 5,000 rows): **0.94502**, −0.011 — the
  log-loss classifier orders better than a pure ranking loss here.
Blends of the best file with any other scored file rank 0.994–0.998 like it: unreadable, finals only.

Read 2026-10-04: model B's learner closed (XGBoost −0.002, CatBoost −0.010, ExtraTrees −0.038).
Read 2026-10-05: **random 5-fold OOF scoring of training rows = +0.0023, new best**; label
information in training neighbour numbers hurts monotonically (closed). The new axis: make the
training-row scorer and the test-row scorer as alike as possible. Plan with candidates, reading
rules and follow-ups: **`docs/plans/plan_step6.md`**. Gate results 2026-10-06: 10-fold OOF ranks
0.996 like the best and the fold-average test scorer 1.000 (the scorer-similarity axis is
exhausted); 7 leaves under random OOF ranks 0.985 like the old 7-leaf file (unreadable); the
**21-step window under random OOF passes (0.970)** and has the best CV of the family. The window
is the one open readable lever under the new protocol: 10 / 21 / 28 steps are built and gated
(0.935 / 0.970 / 0.933 vs the best). Read 2026-10-07: 21 steps 0.95413 (−0.004), 28 steps 0.95298
(−0.006): **the 14-step window holds under random OOF**; 10 steps not uploaded (no decision value).
The window is closed under both protocols. Step 7 (new mechanisms, `docs/plans/plan_step6.md`):
fold scorers with 7 extra steps of history rank 0.994 like the best (scorer-strength axis closed);
`B15_stack_logreg` (logistic-regression score as an extra model-B column) read **0.95485, −0.004**:
a linear stacked score hurts; `B15_pseudo_unlab` (pseudo-labelled unlabelled window rows as extra
model-B training rows) gated 0.915, next upload.** Read 2026-10-07: `B15_stack_knn` 0.95589 (−0.003) and `B15_pseudo_unlab` 0.95278 (−0.006): stacking
and extra training rows closed; the MLP stack is held. **Next plan: `docs/plans/plan_step7.md`**
(input-scale axis). Built 2026-10-07: `B15_nb_steprank` (within-step rank of model-A scores before
aggregation) gated 0.964, uploaded as the day's third slot; `B15_step_ctx` (per-step mean score as a
model-B column) ranks 0.988 like the best, unreadable: model B ignores a per-period level.
Read 2026-10-08: `B15_nb_steprank` 0.95560 (−0.003): input-scale axis closed. **Next plan:
`docs/plans/plan_step8.md`** (bridged neighbours through unlabelled nodes; transductive
self-training of model B on the test rows; private-board finals track; exit condition). Diagnostics 2026-10-07 argued against both step-8 candidates as written (recorded in the plan);
built instead with approval: **`B15_nbfeat`** (means of the top-10 raw features over labelled in/out
neighbours as 20 extra model-B columns, `two_stage(nb_feats=10)`), gated 0.973 vs the best, CV 3-fold
0.9689. **Read 2026-10-08: 0.96069, +0.0022, new best.** Neighbour *feature* information helps model B
once it sees neighbour scores (CV rejected it on the single-stage in step 1: CV is a guide for features
only when the feature enters the same model). Axis reopened; strength being mapped one change at a time:
`B15_nbfeat20` (top-20) and `B15_nbfeat_max` (mean + max of the top-10) built and gated 2026-10-08. Keep the ledger and this file current after every reading; re-execute both
`final_model*.ipynb` whenever the best changes. Note (2026-10-04): `final_model_kaggle.ipynb`
(GPU switch, Kaggle paths) was added outside this session. Repo reorganised 2026-10-05: modules
in `src/`, `main.ipynb` in `notebooks/`, ledger / report / plans / figures in `docs/` (see file map).

## Future implementation notes

- Run scripts from anywhere as `.venv/bin/python src/families.py <name>` (also `src/probes.py`,
  `src/gnn_cache.py`). Repo locations (`submissions/`, `artifacts/`, the ledger, the data cache)
  come from `src/paths.py`; never hard-code a cwd-relative path in a module.
- New variants go in `families.py` (`FAMILIES` dict). Patterns: `lgbm_recent`,
  `lgbm_recency_weighted`, `two_stage(last_k, half_life, b_params, a_all_history, hop2, unlab)`,
  `two_stage_iter2`. The CV sanity check, distinctness gate, file naming and ledger row are
  automatic; `_PHASE` tells two-stage whether `Xva` holds validation or test rows.
- Two-stage model-A predictions must be out-of-fold in time for training rows or the trees
  over-trust them; test neighbours are the labelled test rows only.
- GNN outputs must be precomputed in a PyTorch-only process (`gnn_cache.py`): LightGBM and
  PyTorch hang together in one process on macOS.
- A two-stage build takes ~30 min (model A once per training step, × folds × seeds); queue builds
  behind each other rather than running them concurrently.
- Full notebook execution (`notebooks/main.ipynb`, 55 cells) takes ~2 h; it re-runs every experiment.

## File map

| path | role |
|---|---|
| `final_model.ipynb` / `final_model_kaggle.ipynb` | **the submission notebooks** (local / Kaggle with GPU switch): self-contained, raw files → CSV for the current best model; rename to the student number for hand-in. Stay at the root: they write `submission.csv` here and check against `submissions/` |
| `submission.csv` | output of the last `final_model*.ipynb` run |
| `src/families.py` | current campaign: reference recipe, windows, weights, two-stage, distinctness gate, ledger rows |
| `src/probes.py` | step-4 probe factory (sub_07 reference; superseded) — `Data` class still used by `families.py` |
| `src/validation.py` | rolling-origin folds, horizon fold, per-step AUC, gap metric |
| `src/models.py` | LightGBM factories (`LGBM_PARAMS` = reference, `LGBM_TUNED`), RF/logreg, adversarial validation, search space |
| `src/drift.py` | recency weights, `StepNormalizer`, EM/BBSE prior estimation, score adjustment |
| `src/graph_features.py` | per-period topology and neighbour aggregates, edge rewiring |
| `src/gnn.py`, `src/gnn_cache.py` | inductive GraphSAGE/MLP and the cache builder |
| `src/kaggle_data.py` | competition download (KGAT_ token from `.env`) into `~/.cache/bt4012` |
| `src/paths.py` | repo root, `submissions/`, `artifacts/`, ledger and data-cache locations |
| `notebooks/main.ipynb` | experiment notebook, sections 5–16 (steps 1–3), executed; its first code cell moves the kernel to the repo root and puts `src/` on the path. Steps 5–6 live in `src/families.py` + ledger |
| `docs/results.md` | ledger of every file with CV and public score |
| `docs/report_notes.md` | methodology and results write-up with every table; `docs/figures/` diagrams |
| `docs/plans/plan_step1..6.md` | step plans (`plan_step6.md` = the current round) |
| `docs/instructions.md` | competition reference |
| `submissions/`, `artifacts/` | (gitignored) every CSV; CV tables, caches and `family_*.json` |
| `AGENTS.md` | pointer to this file for other agents |
