# Step 7 — Plan (no code yet): what to change next, built on `fam_B15_Aoof_random`

Date: 2026-10-07. Best: **`fam_B15_Aoof_random.csv` = 0.95849**, 6th of 20. Everything below is a
one-change probe on that file; it stays the reference until something beats it by ≥ 0.002.

## Where we are

The last eleven readings: one gain (random OOF scoring, +0.0023), one exact tie (interactions),
nine losses. Every setting-level axis of the pipeline is closed on the board:

| axis | closed by |
|---|---|
| training window | 7/10/14/21/28/35 on the single-stage; 21/28 again under random OOF |
| base weighting | soft weights flat on plain and shallow two-stage |
| model A | window, all-history, depth, 3-seed bag, extra history, fold count, fold-average test scorer: all rank ≥ 0.994 like the best |
| neighbour numbers | more statistics, 2-hop, unlabelled neighbours, second pass: unreadable; labels in training numbers: monotone loss |
| model B | depth curve (15 peak), learner (XGBoost/CatBoost/ExtraTrees lose), pruning / monotone / regularisation unreadable, interactions tie, ranking objective −0.011 |
| model-B training rows | pseudo-labelled unlabelled rows −0.006 |
| extra columns for model B | logistic −0.004, k-NN −0.003 (MLP held) |
| output side | edge smoothing −0.012, step-mean equalisation flat |

What has never been changed: **the scale on which neighbour numbers and model-A scores enter
model B across time steps.** Every model scores steps 43–49 at a tenth of the earlier steps'
level, so a test row's neighbour numbers there are tiny in absolute terms, and model B, trained
on steps 22–35 where the level was higher, reads "low neighbour suspicion" where it should read
"average for this period". The two candidates below attack that from opposite sides.

## Intended changes (one per upload)

### 1. Within-step rank of model-A scores before neighbour aggregation — `B15_nb_steprank`
Replace each model-A score by its percentile rank within its own time step (train and test
alike) *before* computing the six neighbour numbers. The numbers then say "how suspicious are my
neighbours relative to their period", which is the same question in every period. Model B's own
inputs are unchanged; only the six neighbour columns move.
- Why it might work: removes the period level from the neighbour numbers, which is exactly the
  thing that differs most between the training window and the post-43 test steps.
- Why it might not: the final-score equalisation (sub_08) was flat, and within-step ranking of
  final scores lost; but those acted on the output, this acts on an input model B still weighs.
- Readability: it rescales every neighbour column non-linearly, so the ranking should move
  (expect ≈ 0.95–0.97 vs the best).

### 2. Step-context feature — `B15_step_ctx`
Add one column for model B: the mean model-A score over the labelled rows of the row's time
step (computed within the window for training rows, within the test file for test rows).
Model B can then condition on "how risky this period looks" explicitly instead of inferring it
from the raw neighbour levels. The opposite move to candidate 1: keep the level, but hand it to
model B as its own feature.
- Why it might work: lets model B treat a low neighbour number in a low-risk period differently
  from the same number in a high-risk period.
- Why it might not: a per-step constant is close to a period identifier; the trees may use it to
  learn the training steps' base rates and extrapolate badly. The board decides.
- Readability: one strong column; likely readable (expect ≈ 0.95–0.98).

### 3. Both together — `B15_nb_steprank_ctx` (only if 1 or 2 reads positive)
Rank-normalised neighbour numbers plus the step-context column: period-free neighbour
information with the period level supplied separately.

## What is deliberately not on the list

- Finals-level blends (unreadable on the public board; they belong to the final selection).
- Any further model-A, model-B-setting, window, or neighbour-statistic variant.
- The MLP stack (built; held after two stacking losses).

## Reading rule and exit

vs 0.95849: ≥ +0.002 gain, ≤ −0.002 loss, between is a tie. If both 1 and 2 lose, the
input-scale axis closes too, and the honest position is that this pipeline is at its public-board
optimum. The remaining effort then goes to the private board: select the two finals
(`fam_B15_Aoof_random` + the strongest file ranking < 0.97 like it), keep `final_model*.ipynb`
reproducing the best, and finish the report, while keeping one slot a day for any genuinely new
idea that passes the gate.

## Today's remaining slot

Hold it. Nothing built is both readable and informative (the MLP stack is held; the 10-step
window has no decision value).

## Status (2026-10-07)

Built. `B15_nb_steprank` gated 0.964 → uploaded today. `B15_step_ctx` 0.988, unreadable (candidate 2 closed
without a board reading; candidate 3 therefore reduces to candidate 1). Next step depends on the steprank reading.
