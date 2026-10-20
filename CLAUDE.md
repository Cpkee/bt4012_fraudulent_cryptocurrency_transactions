# CLAUDE.md — BT4012 fraudulent cryptocurrency transactions

Kaggle competition `bt-4012-competition-2026`: score Bitcoin transactions by probability of being
illicit; metric = ROC AUC on the pooled test set. Deadline for report and final selection:
**25 Oct 2026**.

**Objective (set by the user, 2026-10-03): keep improving the public leaderboard score, continuously,
until the deadline.** Do not plan around "freezing" or hand-in; the notebook and report are
maintained alongside, not instead of, the search for the next gain. Every day with unused upload
slots is a day of lost information: there should always be readable candidates built and gated.

**Target (set by the user, 2026-10-18): beat the current first place, public 0.97033.** Our best is the xt + blocks_xt blend 0.96828
(gap +0.0021; before it xt 0.96678 / 0.96736; the Final 1 blend 0.96539, gap +0.0049). Feasibility and route: `docs/plans/plan_step11.md` (round 14).

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
   the public score by ≈ 0.001 on the plain two-stage (2026-10-02) and by **0.0015 on the current
   model** (2026-10-08: seeds 0–4 0.96177, seeds 5–9 0.96323); a single-file difference below ≈ 0.003 is
   therefore not a reading, and the finals file is a 10-seed bag.
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
   harness, but a model is not final until the notebook reproduces it. **Hand-in notebook
   (2026-10-16): `submission_notebook.ipynb`**, the same pipeline stripped to the final model and
   laid out as the report's §2.1–2.3 (preprocessing & features / model & validation / exploration &
   final strategy; exploration results quoted from the ledger, not re-run; `FAST` / `USE_GPU`
   flags). When the final model changes, change both notebooks. Rename `submission_notebook.ipynb`
   to the NUS student number (`eXXXXXXX.ipynb`) for the final hand-in.

## Data facts (measured)

- train: 141,772 rows, steps 1–35, 165 features, 78% unlabelled (31,235 labelled, 11.7% illicit);
  test: 15,329 labelled rows, steps 36–49. Edge list 234,355 edges; edges never cross steps, so
  there are no train–test edges. Test holds labelled nodes only; 46,668 test-period nodes exist
  as ids in the edge list without features.
- Labelled fraud rate per step swings 0.43%–35.97% inside train. From step 43 on, every model we
  built scores almost everything as licit (mean predicted rate 0.022 vs 0.085 for 36–42); the data
  cannot say why.
- Six columns (`feat_100, 101, 103, 136, 137, 139`) identify the time period (single-feature
  adversarial AUC 0.99–1.00, Spearman with step 0.90–1.00; `feat_100, 101, 103` also have 99% of
  test values out of training range, the other three 0%). Removed everywhere.

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
| fam_B15_nbfeat_all | means of all 159 features over labelled in/out neighbours (318 columns) | 0.9618 (tie band, +0.0011) |
| fam_B15_nbfeat_all_seedsB | the same model, seeds 5–9 (noise reading) | **0.9632** (+0.0015 = seed noise; highest public score) |
| fam_B15_nbfeat_all_bag10 | the 10-seed bag of the two above (Final 1 until the 15-seed bag) | 0.9626 |
| fam_B15_nbfeat_gnn | + out-of-fold GraphSAGE embeddings as model-B columns | 0.9557 (−0.005, closed) |
| fam_B15_nbfeat_comm | + mean/max/count of model-A scores over the row's Louvain community (3 model-B columns) | 0.9573 (−0.0034, closed) |
| **fam_B15_nbfeat_all_bag15** | the 15-seed bag (seeds 0–14) of the all-159 model: **Final 1** | **0.9631** (2026-10-10) |
| fam_B15_Aoof_random_bag10 | the 10-seed bag of the random-OOF two-stage: **Final 2** (ranks 0.964 like Final 1) | 0.9597 (2026-10-10) |
| fam_bag15_steprank | Final 1 replaced by its within-step percentile rank (cross-period calibration question) | 0.9412 (−0.022: the period level is right; closed) |
| fam_bag15_levelamp25 | Final 1 with each period's level amplified by 0.25 in log-odds (the mirror probe) | 0.9617 (−0.0015, tie band: the level is used about right; axis closed both ways) |
| fam_bag15_post43floor | Final 1 with the post-43 block ranked below every pre-43 row (information probe) | 0.8920 (−0.071: the block holds ≈ 1.3–2.4% fraud that Final 1 ranks above most pre-43 licit rows) |
| fam_B15_nbfeat_all_bag30 | the 30-seed bag (seeds 0–29) of the all-159 model | 0.9631 (= bag15: seed variance flat from 15 seeds; Final 1 stays bag15) |
| fam_B15_nbfeat_pseudo_pre43 | confident test rows of steps 36–42 as pseudo-labelled model-B training rows | 0.9555 (−0.0052: transductive axis closed) |
| fam_post43floor_sub06 | floorsplice: Final 1's pre-43, sub_06's post-43 block order | 0.8851 (−0.0069 vs the floor: all-history orders the block worse) |
| fam_post43floor_lowrate | floorsplice: the regime-matched (low-rate steps) block order | 0.8780 (−0.0141: much worse; the offline pre-screen misranked both donors) |
| fam_B15_nbfeat_all_goof | all-159 model with model-A OOF folds grouped by connected component (label-leak fix) | 0.9621 (tie vs 0.9618 / 0.9632, as gated 0.995; selectable) |
| fam_B15_nbfeat_all_goof_bag15 | the 15-seed bag of the grouped-OOF model | 0.9631 (= Final 1 0.96313; leak-free, selectable) |
| fam_bag15_post43floorrev | the floor probe with Final 1's post-43 order reversed (calibration) | 0.8384 (−0.054 vs the floor → within-block AUC ≈ 0.87–0.96: post-43 campaign closed) |
| **fam_B15_nbfeat_all_combo** | all-159 + grouped OOF + top-10 means into A + max/delta/argmax-neighbour of the top-10 + step-mean score (5 seeds) | **0.9636** (2026-10-14: highest public score; +0.0011 vs the 5-seed parents' mean, inside seed noise, CV-consistent) |
| fam_B15_nbfeat_combo | the same stacked changes on the top-10 model | 0.9595 (−0.0012 vs 0.96069: the two board readings of the extras average ≈ 0) |
| fam_B15_nbfeat_all_combo_bag15 | the 15-seed bag of the all-159 combo | 0.9632 (tie with both bag15s: the 5-seed 0.9636 was seed noise; extras closed) |
| fam_B15_nbfeat_all_soof / _bag15 | all-159 with leave-one-step-out OOF (5 / 15 seeds) | 0.9632 / 0.9629 (ties: 15-seed bags of all four OOF protocols read 0.9629–0.9632) |
| **fam_B15_nbfeat_all_goof_blocks** | Final 1 model with model B = mean of fits on all / local+nb / aggregated+nb columns (5 seeds) | **0.9638** (2026-10-15: highest public score; +0.001 vs the 5-seed family, tie band, but ranks 0.920 like Final 1: a blend partner) |
| fam_B15_nbfeat_all_goof_blocks_bag15 | the 15-seed bag of the block-ensemble model | 0.9639 (+0.0008 vs Final 1: tie) |
| **fam_blend_goof_blocks_bag15** | rank average of Final 1 (goof bag15) and the 15-seed block-ensemble bag (0.921 apart) | **0.9654** (2026-10-16: highest public score, +0.0023 vs Final 1: diversity between equally good models pays) |
| fam_B15_nbfeat_all_goof_xgb | Final 1's inputs with model B = XGBoost (5 seeds; ranks 0.944 like Final 1) | 0.9619 (−0.001 vs the 5-seed family: a tie; CV had it −0.003 to −0.014) |
| fam_blend3_goof_blocks_xgb_bag15 | rank average of the goof, blocks and XGBoost 15-seed bags | 0.9653 (= the 2-way blend: tie) |
| fam_B15_nbfeat_all_goof_dropadv | goof without the 5 drifting columns with adversarial AUC > 0.85 (`feat_2, 106, 107, 109, 142`), raw + neighbour means | 0.9596 (2026-10-18: −0.0025 vs goof 5-seed although CV +0.006 / +0.015 / +0.032; drift removal closed) |
| **fam_B15_nbfeat_all_goof_xt** | goof with model B = LightGBM `extra_trees=True` (random split thresholds), 5 seeds | **0.9668** (2026-10-18: highest public score; +0.0047 vs goof 5-seed, CV tied) |
| fam_B15_nbfeat_all_goof_xt_r600 | xt with 600 model-B rounds | 0.9636 (2026-10-18: −0.0032 vs xt, rejected) |
| **fam_B15_nbfeat_all_goof_xt_seedsBC** | xt, seeds 5–14 (replicate, no seed shared) | **0.9674** (2026-10-19: highest public score; xt confirmed, mean of the two seed sets 0.9671; rank 6) |
| fam_B15_nbfeat_all_goof_blocks_xt | block-ensemble model B with `extra_trees` (5 seeds) | 0.9663 (2026-10-19: +0.0025 vs blocks; blend member for xt) |
| **fam_blend_..._xt_bag15_..._blocks_xt_bag15** | rank average of the 15-seed xt and blocks_xt bags | **0.9683** (2026-10-19: highest public score; +0.0016 over the members' mean; **Final 1**) |
| fam_B15_nbfeat_all_goof_xt_w21 | xt with a 21-step window | 0.9676 (2026-10-20: tie vs xt 0.9671; 14 steps stays) |
| fam_B15_nbfeat_all_goof_xt_all | xt trained on all 35 steps (0.927 like xt) | 0.9649 (2026-10-20: below the 0.9655 member bar; third-member search closed) |
| fam_B15_nbfeat_all_goof_xt_w21_seedsBC | xt with 21 steps, seeds 5–14 (replicate) | 0.9668 (2026-10-20: 21-step mean 0.9672 = 14-step mean 0.9671; window closed) |

Standing (2026-10-05): **6th of 20**; top 0.9681; five entries above us (0.9609–0.9681). **2026-10-18: first place 0.97033.** Model-B depth
curve (under time-ordered OOF): 31 → 0.9526, 23 → 0.9550, **15 → 0.9562**, 7 → 0.9550, 5 ≈ 7.
Label curve for training neighbour numbers: 0% labels 0.9562, 50% 0.9542, 100% 0.9462 (closed).
**Reference = fam_B15_nbfeat** (0.96069; `final_model.ipynb` and `final_model_kaggle.ipynb` reproduce it exactly);
**highest public score = fam_B15_nbfeat_all_seedsB** (0.96323, 2026-10-08: the all-159 model re-seeded; the
0.96069 → 0.96177 → 0.96323 steps are all inside seed noise). **Final 1 = `fam_B15_nbfeat_all_bag10`** (the
10-seed bag = probability average of the two scored 5-seed files; must be uploaded to be selectable). The brief gives
the lowest-ranked participant 0 points. Finals must be selected by hand on Kaggle (two files;
the better private score counts). Standing choice, revised whenever the best changes: the largest seed bag of the
highest-scoring model plus a seed bag of the strongest file that ranks < 0.97 like it (step 10: `fam_B15_nbfeat_all_bag15` = **0.96313** +
`fam_B15_Aoof_random_bag10` = **0.95970**; both scored 2026-10-10: **select these two on Kaggle**). **Revised 2026-10-14: Final 1 = `fam_B15_nbfeat_all_goof_bag15` (0.96310, leak-free grouped OOF, ties the old
Final 1 and is CV-better on both seed pairs) + Final 2 = `fam_B15_Aoof_random_bag10`; `final_model*.ipynb` switched to grouped OOF and re-executed 2026-10-15: both reproduce it exactly (4.4e-16).**

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
- **The ordering between periods is worth 0.022 of public AUC** (2026-10-10: Final 1 with every score
  replaced by its within-step rank fell from 0.9631 to 0.9412). The model's low scores after step 43 are
  right as far as the board can tell: the collapse is a base-rate drop, not blindness, and any change
  that flattens period levels will lose. Amplifying the level (×0.25 in log-odds) read −0.0015 (2026-10-11):
  the model uses the period level about right; the axis is closed in both directions.

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
`B15_nbfeat20` (top-20) and `B15_nbfeat_max` (mean + max of the top-10) rank 0.989 / 0.994 like the
best: unreadable, the amount of neighbour feature information is saturated at the top-10 means. Next,
changing the kind of information: `B15_nbfeat_delta` (row minus neighbour mean) and `B15_nbfeat_all`
(means of all 159 features), gated 2026-10-08: delta 0.992 (unreadable), **nbfeat_all 0.979 passed and read 0.96177 (+0.0011, tie band;
highest public score)**: the amount of neighbour feature information is closed. **Next plan: `docs/plans/plan_step9.md`**
(neighbour feature means into model A as well: `B15_nbfeat_A` ranks 0.991, unreadable, entry point closed;
`B15_nbfeat_hop2` ranks 0.988, unreadable). **Step-9 exit condition reached 2026-10-08: the neighbour-feature axis
is closed on amount, statistic, entry point and reach.** Finals = `fam_B15_nbfeat_all` + `fam_B15_Aoof_random`; one slot
a day for an idea that changes the information set and passes the gate; report work resumes.
Round 2 (user-approved 2026-10-08, `plan_step9.md`): seed replicate of the leader uploaded as a noise reading;
per-step balanced weights 0.986 (unreadable); **`B23_nbfeat` passes the gate (0.972)**; OOF GraphSAGE embeddings as
model-B columns (`gnn_cache_window.py`, `two_stage(b_gnn_embed=True)`) **passes the gate at 0.932** (CV 3-fold 0.9611). **Seed replicate read 0.96323**: seed noise
0.0015 on this model. **Read 2026-10-09: `fam_B15_nbfeat_gnn` 0.95570 (−0.005, clear loss: the trees + embeddings
recipe is closed on the board as the in-sample version was on CV); `fam_B15_nbfeat_all_bag10` 0.96263** (between its
parents: Final 1 selectable). `B23_nbfeat` (0.972) deferred for good: a depth re-check's delta would sit inside seed
noise. **Next plan: `docs/plans/plan_step10.md`** (variance, finals, report): Final 1 = the 15-seed bag
`fam_B15_nbfeat_all_bag15` (seeds 10–14 built 2026-10-09), Final 2 = `fam_B15_Aoof_random_bag10` (seeds 5–9); reading
rule revised to ±0.003. `final_model*.ipynb` reproduce **Final 1, the 15-seed all-159 bag, exactly (4.4e-16, 2026-10-10)**. Keep the ledger and this file current after every reading; re-execute both
`final_model*.ipynb` whenever the best changes. Note (2026-10-04): `final_model_kaggle.ipynb`
(GPU switch, Kaggle paths) was added outside this session. Repo reorganised 2026-10-05: modules
in `src/`, `main.ipynb` in `notebooks/`, ledger / report / plans / figures in `docs/` (see file map).

**Step 10, SNA track (2026-10-09, `docs/plans/plan_step10_sna.md`): Lecture 8 (Social Network Analysis) mapped onto the
pipeline.** Measured: all 35 train steps are homophilic (cross-edge rate r ≈ 0.25 × the chance level 2xy in
the window), the data-grounded justification of the two-stage; fraud PageRank with the row's own score removed
from the restart adds little over the one-hop mean (AUC 0.896 vs 0.877, not built); **Louvain communities on
the full per-step graph reach 99.5% of rows where labelled one-hop neighbours reach 78%**, and the community
mean carries more label signal (0.92 vs 0.85). Built `B15_nbfeat_comm` (`two_stage(nb_comm=True)`: mean / max /
count of model-A scores over the other scored rows of the row's community, 3 model-B columns; partition cached
at `artifacts/communities_louvain_s0.csv`, label- and feature-free): **passes the gate at 0.963 with the family's
best CV (3-fold 0.9769, horizon 0.8511)**; took slot 3 of 2026-10-09 (the 15-seed finals bag moves to 2026-10-10).
**Read 2026-10-09: 0.95732, −0.0034 vs 0.96069: a loss.** The community unit reaches more rows and has the better CV,
yet the board rejects it like every other multi-hop / wider-neighbourhood variant (2-hop, bridge, smoothing, GNN):
the graph's usable content on the test period is the labelled one-hop neighbourhood, and CV is again wrong about a
neighbour-information change in the direction opposite to before. **Community / SNA axis closed** (homophily curve and
the PageRank self-score trap remain report material).

**2026-10-11 (user decision, two slots; `plan_step10.md` tail):** slot 1 = `fam_bag15_post43floor` (`src/post43_probes.py`:
Final 1 with the post-43 block, 43.6% of rows, ranked below every pre-43 row, block order kept; Spearman 0.456): reads
how much rankable fraud the post-43 block holds (≈ 0 → closed for good; a clear loss → the floor becomes a protocol for
comparing post-43 scorers with the cross-block terms held fixed). Slot 2 = `fam_B15_nbfeat_all_bag30` (seeds 15–29
building as `B15_nbfeat_all_seedsD`, averaged with Final 1): variance only, unreadable by construction, uploaded to be
selectable; notebooks go to 30 seeds only if it is selected.
**Read 2026-10-11: `fam_bag15_post43floor` = 0.89203, −0.071.** The post-43 block holds fraud (≈ 1.3–2.4% of its public
rows, 27–49 positives) that Final 1 ranks above most pre-43 licit rows; the within-block ordering term has weight ≈ 0.06–0.10,
so a better post-43 ordering is worth up to ≈ 0.01–0.02, with board noise ≈ 0.004 on that term. **Open lever: the post-43
ordering, read through the floor protocol** (`post43_probes.py floorsplice`: pre-43 = Final 1, post-43 floored with a
candidate's order; difference vs 0.89203 = the candidate's within-block quality). Existing files order the block 0.84–0.96
like Final 1; first candidate `sub_06` (all-history, 0.844) is built (`fam_post43floor_sub06`). A candidate that reads positive
becomes "Final 1 with its post-43 order, mapped onto Final 1's post-43 value range" (the level stays: the level probes say it is right).

**Step 11 (2026-10-12, user-approved: all 3 slots/day this week; `docs/plans/plan_step11.md`): the post-43 ordering
campaign.** Measured: the only training steps with a post-43-like fraud rate (< 3%) are **1–6 and 10** (9,322 labelled
rows, 107 positives; < 5% adds 12 and 23, 176 positives) — the recent-14 window contains no low-rate regime at all.
Offline pre-screen (`src/post43_prescreen.py`, approved as a ranker only; low-rate band E = steps 1–6, 10 as the in-data
analogue of the block): the win14 model orders E at pooled AUC **0.810**, all-history 0.96–0.97, and a 7-leaf LightGBM
trained on the low-rate steps (leave-one-step-out) **0.992** with a very different order (Spearman 0.37–0.50 vs win14);
kNN fraud-similarity 0.73 (deprioritised). Era-control (train on early high-rate steps 7–9, 11): 0.93–0.95, so part of
the gain is era proximity that cannot transfer to post-43, but the rate-matching increment (+0.04–0.06) and the ranking
regime-matched > all-history > window survive. Built: **`fam_lowrate`** (7 leaves, steps 1–6, 10, 12, 23, bagged ×10;
splice donor only, CV sanity gate does not apply) and **`fam_post43floor_lowrate`** (its block order spliced under
Final 1's pre-43 rows; within-block Spearman **0.687** vs Final 1, the most different order yet). Slots of 2026-10-12:
1 = `fam_post43floor_sub06`, 2 = `fam_B15_nbfeat_pseudo_pre43`, 3 = `fam_post43floor_lowrate`. Reading rule for splices
vs 0.89203: ≥ +0.005 → promote to the mapped full candidate; ≤ −0.005 → next donor; between → unreadable, next donor.
Next donors in order (plan_step11.md): post-43 confident-licit pseudo-negatives, pseudo_pre43's block order, zero-build
splices (sub_04, recent14_local). Exit: three losses/ties from different mechanism families closes the campaign.

**Read 2026-10-12 (all three slots): `fam_post43floor_sub06` 0.88509 (−0.0069), `fam_post43floor_lowrate` 0.87796
(−0.0141), `fam_B15_nbfeat_pseudo_pre43` 0.95553 (−0.0052).** All losses. Final 1's own block order beats both motivated
challengers — more history and regime-matched training each order the block *worse* — and the transductive axis is
closed. The offline pre-screen misranked the two donors even relative to each other (it had regime-matched ≫
all-history; the board read the reverse): **discredited as a ranker for temporal questions, like every offline proxy
before it.** pseudo_pre43's block order is 0.936 like Final 1 (never readable; dropped). The open question is now
calibration: Final 1's *absolute* within-block AUC. **Slot 1 of 2026-10-13 = `fam_bag15_post43floorrev`** (Final 1's
block order reversed under the floor; built, gated): Δ vs 0.89203 = −w·(2·A22−1), w ≈ 0.06–0.10, so A22 = 0.5 − Δ/(2w).
A22 ≥ ~0.85 → headroom < ~0.01, the campaign closes with evidence; A22 low → it continues. The pseudo-negatives donor
`B15_nbfeat_pseudo_p43neg` (`two_stage(b_pseudo_post43_neg=True)`) was built 2026-10-12: CV = the parent's exactly
(wiring check), but its within-block Spearman is **0.926** vs Final 1 — the splice is unreadable and dropped. The pattern
is now measured three ways: **every variant of the current pipeline orders the block essentially like Final 1**
(pseudo_pre43 0.936, p43neg 0.926, scored files 0.84–0.96); only outside models differ (sub_06 0.844, lowrate 0.687) and
both ordered it worse. Slot 2 of 2026-10-13 is contingent on the calibration: if the campaign continues, the
`B15_nbfeat_pseudo_p43neg` *full file* vs 0.96069 (passes the whole-file gate at 0.960; weakly motivated, sibling read
−0.0052); if it closes, finals stand and report work resumes with one exploratory slot a day.

**Step 11 round 2 (user-approved 2026-10-13, `plan_step11.md`): the neighbour-feature axis re-opened for the two
untried kind-changes, gate-only.** `B15_nbfeat_scorew` (score-weighted neighbour feature means: the
suspicious-neighbour profile, `two_stage(nb_feat_scorew=True)`) and `B15_nbfeat_top1` (the argmax-neighbour's raw
features, `two_stage(nb_feat_top1=True)`). **Both failed the gate — 0.993 / 0.992 vs `fam_B15_nbfeat`, the same
0.988–0.994 band as every step-9 variant.** Model B reaches the same test ranking whatever shape the one-hop
neighbour information takes: **the axis is closed for good on amount, statistic, entry point, reach and kind**; no
slot spent.

**Step 11 round 3 (user-approved 2026-10-13): label leak in random OOF.** Under random 5-fold OOF the scorer of a
neighbour saw the row's own label 80% of the time (`nb_out_mean` AUC 0.902 random vs 0.889 grouped). Built
`two_stage(a_oof="group")` (StratifiedGroupKFold, groups = connected components of the labelled window subgraph):
`B15_nbfeat_goof` CV 0.9739 / 0.9293 / 0.8443 vs parent 0.9689 / 0.9139 / 0.8215 (largest gain at the horizon), and
`B15_nbfeat_all_goof` 0.9656 / 0.9056 / 0.8113 vs 0.9627 / 0.8963 / 0.8018. **Both fail the gate (0.994 / 0.995 vs
their parents)**: the leak is real but does not move the test ranking. OOF-protocol axis closed; no slot spent. **User decision 2026-10-13: near the top a
gate failure is not a veto for a principled fix** — the gate says whether the board can read a change, not whether it
is better. `fam_B15_nbfeat_all_goof` is uploaded as a finals candidate (its score = "selectable", not a reading) and
bagged to 15 seeds (`B15_nbfeat_all_goof_seedsBC` = seeds 5–14) to compete with Final 1 like for like. Read: 0.96209 (5 seeds) and **0.96310 (15 seeds) = Final 1**.
`fam_bag15_post43floorrev` read **0.83841** (−0.054 vs the floor): Final 1's within-post-43 AUC ≈ 0.87–0.96, headroom
≤ 0.002–0.009: **post-43 campaign closed with evidence**.

**Step 11 round 4 (2026-10-14, user goal: 7–8 more days of gains; `plan_step11.md`):** all CV-positive gate-failed
changes stacked (`two_stage(a_oof="group", nb_feats_into_A=True, nb_extra_k=10, nb_extra_stats=("max",),
nb_feat_delta=True, nb_feat_top1=True, step_ctx=True)`). `B15_nbfeat_combo` CV 0.9746 / 0.9311 / 0.8491 (best top-10
CV), gate **0.982** vs `fam_B15_nbfeat` (the singles were 0.988–0.994: the changes add up); `B15_nbfeat_all_combo`
0.9671 / 0.9097 / 0.8150, 0.992 vs the goof bag15. **`fam_B15_nbfeat_all_combo` read 0.96362 (2026-10-14), the highest public score** (+0.0011 vs the 5-seed parents' mean); seeds 5–14 building for the 15-seed bag = the Final 1 candidate.

**Step 11 round 6 (2026-10-15, `plan_step11.md`): one model-B change each on Final 1.** Colsample 1.0: CV lower, 0.992
(dropped). DART (300 rounds): clear CV loss (−0.009 / −0.024 / −0.017), ranks < 0.93 like older weaker files (held).
**Feature-block ensemble `B15_nbfeat_all_goof_blocks`** (`two_stage(b_blocks=True)`: model B = mean of fits on all /
local + neighbour / aggregated + neighbour columns): CV 0.9657 / 0.9069 / 0.8166 (≥ parent), **0.920 vs Final 1: the
first readable candidate with non-negative CV since nbfeat_all.** Read 0.96382 (highest public; tie band but distinct): seeds 5–14 building for `blocks_bag15`; next = `blocks_bag15` and the blend (Final 1 + blocks_bag15)/2; finals decided by those two readings.

**Step 11 rounds 7–8 (2026-10-16):** blocks_bag15 read 0.96385 (tie); **the blend (Final 1 + blocks_bag15, rank average)
read 0.96539, the highest public score**. XGBoost model B (0.944 vs Final 1) failed the pre-set CV inclusion rule (−0.005 /
−0.015 / −0.024 on seeds 5–6); held. `src/blend_cv.py` scores blends on CV (cached member predictions): CV does **not**
confirm the blend's board gain (goof + blocks 0.9674 / 0.9097 / 0.8155 vs goof 0.9675 / 0.9112 / 0.8199); soof is the
CV-best single model on both seed pairs. **Finals: Final 1 = `fam_blend_goof_blocks_bag15` (board-best) + Final 2 =
`fam_B15_nbfeat_all_soof_bag15` (CV-best).** `final_model*.ipynb` carry `CONFIG["members"]` (default `("lgbm", "blocks")`) and a pooled-rank
blend; re-executed 2026-10-16, both reproduce Final 1 (2.2e-16) and each member bag (≤ 5.6e-16) exactly. `fam_blend3_goof_blocks_xgb_bag15` built (members + `"xgb"`), unreadable vs the 2-way blend; finals choice at the end.

**Step 11 round 10 (2026-10-18, `plan_step11.md`): drift-aware feature removal.** `dropadv` (5 columns with adversarial
AUC > 0.85 removed) had the largest CV gain since the two-stage, growing with the horizon, and read **0.95957** (−0.0025 vs
goof 5-seed): CV wrong again on a drift decision. Not a level flattening: its probability-mean pre/post ratio fell
(11.1 → 8.8) but in ranks it puts *fewer* post-43 rows in the top 5% (0.096 vs 0.113, like levelamp25 0.093); probability
means mislead here (low-end calibration), use rank shares. Cause not identified. Drift handling is closed on
removal as on normalisation; `dropunstable` (23 direction-unstable columns, 0.976 vs goof, CV optimistic) is built, not uploaded.

**Finals (2026-10-19, user-approved): Final 1 = `fam_blend_B15_nbfeat_all_goof_xt_bag15_B15_nbfeat_all_goof_blocks_xt_bag15`
(0.96828: rank average of the 15-seed xt and blocks_xt bags), Final 2 = `fam_blend_goof_blocks_bag15` (0.96539, the same
blend without extra_trees: the hedge if the extra_trees gain is public-only).** `final_model.ipynb` / `final_model_kaggle.ipynb`
(`CONFIG["members"]` default `("xt", "blocks_xt")`; `("lgbm", "blocks")` = Final 2) and the hand-in notebook `e1130084.ipynb`
(`extra_trees=True` in `CONFIG["model_b"]`; `False` = Final 2) re-executed 2026-10-19: all three reproduce Final 1 and both
member bags exactly (≤ 4.4e-16).

**Rounds 16–18 (2026-10-19/20): window and third member under extra_trees, closed.** `inout_xt` 0.987 like xt (held);
window curve with extra_trees 14 → 0.9671 (two seed sets), 21 → 0.9672 (two seed sets: 0.96755 / 0.96676), 35 → 0.9649:
14 and 21 are equal, all history still costs ≈ 0.002 (≈ 0.005 without extra_trees); `xt_all` ranks 0.927 like xt but sits
below the 0.9655 member bar. No readable route to 0.9700 remains in this architecture; Final 1 (0.96828) stands.

**Round 15 (2026-10-19): third-member search closed.** `soof_xt` 0.996 and `xtA` 0.993 like xt (model-A changes converge
onto xt once model B uses extra_trees; xtA has the best xt-family CV but is unreadable); `nbonly_xt` (neighbour columns only)
fails the sanity gate (CV 0.874) and ranks 0.39 like xt. Only column-block structure gives an equally strong, different member.

**Rounds 11–13 (2026-10-18, `plan_step11.md`, `plan_step12_dro.md`):** **`xt` (model B `extra_trees=True`) read 0.96678,
the highest public score** (+0.0047 vs goof 5-seed, CV tied); replicate `xt_seedsBC` (seeds 5–14) is the next upload, then
`xt_bag15` / `blocks_xt`. Its neighbourhood is mapped: 600 rounds 0.96363 (rejected), max_bin 63 (0.993) and 31 leaves (0.981,
CV lower) unreadable; strongly regularised model B CV-lower, held. Group DRO over window steps (`two_stage(b_dro=...)`) closed
before a build: model B scores every window step 0.978–1.000 even held out (τ² = 0, weights stay 1); step reweighting closed.

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
- **Column order is part of the model.** LightGBM samples 60% of columns per tree by position, so the
  neighbour-feature block must be added in the same order everywhere (the step-1 gain order from
  `artifacts/lgbm_importance_raw_1to28.csv`); the notebooks carry that order as an explicit list in
  `CONFIG`. A file-order version of the same model differed by up to 0.066 in probability (2026-10-08).
- GNN outputs must be precomputed in a PyTorch-only process (`gnn_cache.py`): LightGBM and
  PyTorch hang together in one process on macOS.
- A two-stage build takes ~30 min (model A once per training step, × folds × seeds); queue builds
  behind each other rather than running them concurrently.
- Full notebook execution (`notebooks/main.ipynb`, 55 cells) takes ~2 h; it re-runs every experiment.

## File map

| path | role |
|---|---|
| `e1130084.ipynb` | **the hand-in notebook** (was `submission_notebook.ipynb`, renamed to the student number): report-structured (§2.1–2.3), self-contained, final model only, raw files → `submission.csv`, checks against `submissions/`; runs locally and on Kaggle (`USE_GPU`) |
| `final_model.ipynb` / `final_model_kaggle.ipynb` | the variant notebooks (local / Kaggle with GPU switch): self-contained, raw files → CSV for the current best model, with `CONFIG` switches for variants (OOF protocol, xgb member). Stay at the root: they write `submission.csv` here and check against `submissions/` |
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
| `docs/plans/plan_step1..11.md`, `plan_step10_sna.md` | step plans (`plan_step11.md` = the current round: the post-43 ordering campaign, floorsplice donors, pre-screen protocol; `plan_step10_sna.md` = the Lecture 8 SNA mapping, closed 2026-10-09) |
| `src/post43_probes.py`, `src/post43_prescreen.py` | post-43 block probes (floor / ceiling / floorsplice / bag) and the offline low-rate-band pre-screen (ranker only) |
| `docs/instructions.md` | competition reference |
| `submissions/`, `artifacts/` | (gitignored) every CSV; CV tables, caches and `family_*.json` |
| `AGENTS.md` | pointer to this file for other agents |
