# Step 8 — Plan (no code yet): after the input-scale axis closed

Date: 2026-10-08. Best: **`fam_B15_Aoof_random.csv` = 0.95849**, 6th of 20 (5th 0.96085, top 0.96805).
`fam_B15_nb_steprank` read 0.95560 (−0.003): removing the period level from the neighbour numbers
hurts, and handing the level to model B as a column is unreadable. The period level inside the
neighbour numbers is information model B uses correctly.

## Honest position

Since the random-OOF gain (2026-10-05) there have been twelve readings: one exact tie and eleven
losses, across every axis the pipeline has. Under one-change probes this pipeline is at its
public-board optimum. What remains are mechanisms that change *which information* the model sees,
not how it is configured. Two such mechanisms have never been tried, and both are readable by
construction because they change the neighbour set or the training set for the test rows.

## Candidates (one change each on `fam_B15_Aoof_random`)

### 1. Bridged neighbours through unlabelled nodes — `B15_bridge`
Today a neighbour must be a labelled transaction directly connected by an edge. Unlabelled
transactions are dropped, and with them every path that runs through one: two labelled
transactions joined by an unlabelled one are currently strangers. The change: count as a
neighbour any labelled transaction reachable through at most one unlabelled transaction
(direction kept: in via in, out via out). The unlabelled node contributes no score itself; it is
a bridge only, so train and test are treated identically (unlabelled train nodes bridge in train,
unlabelled test-period nodes bridge in test). The six neighbour numbers are then computed over the
enlarged neighbour sets.
- Why it might work: the test graph loses 24,245 edges to unlabelled nodes; bridging restores
  some of the connectivity those edges carried, for exactly the rows whose neighbour numbers are
  currently blank or thin.
- Why it might not: earlier 2-hop aggregates (through labelled nodes) were unreadable; the graph's
  usable content may still be first-order.
- Readability: changes the neighbour set for a large share of rows → expected 0.90–0.97.

### 2. Transductive self-training of model B on the test rows — `B15_selftrain_test`
After the final model B scores the test rows, the most confident test rows (score > 0.9 or < 0.02)
are added to model B's training set with their predicted labels and model B is refitted once.
Uses the test rows' *features* (allowed by the competition; declared as transductive), never any
test label. The single-stage version lost 0.018 on historical CV and was never board-read.
- Why it might work: it is the only large source of information untouched by any probe, the
  15,329 test feature vectors, and the test period is exactly where the model is weakest.
- Why it might not: the model's confident test predictions after step 43 are "licit" almost
  everywhere, so the pseudo-labels may just reinforce the blind spot (as the unlabelled-row
  pseudo-labels did, −0.006).
- Readability: refitting model B on a different training set → expected 0.93–0.97.

### 3. Only if 1 or 2 reads positive
Combine the winner with the other, or map its strength (bridge depth 2; self-training confidence
thresholds 0.8 / 0.05).

## Private-board track (runs regardless; not readable on the public board)

The grade uses the private 70%. A seed bag re-seeded moves the public score by 0.001; a different
30% sample moves it by ~0.006. Two files are selected by hand; the better private score counts.
- Final 1: `fam_B15_Aoof_random` (best public).
- Final 2: the strongest file ranking < 0.97 like it — currently `fam_recent14_2stage_B15`
  (0.95621, Spearman 0.980 → too similar) or `fam_recent14_2stage_tunedB` (0.95504, 0.96). Decide at
  the end; if a candidate from this step wins, it becomes Final 1 and the current best Final 2.

## Exit condition

If 1 and 2 both lose, stop single-change probing of this pipeline: keep the finals above, keep
`final_model*.ipynb` reproducing the best, write the report, and use at most one slot a day on an
idea that changes the information set and passes the gate. Reading rule unchanged (vs 0.95849:
≥ +0.002 gain, ≤ −0.002 loss).

## Diagnostics (2026-10-07, read-only, single seed; scripts in the session scratchpad)

Measured on the actual neighbour-number construction (labelled-only neighbours, window 22–35,
random 5-fold OOF for training rows, full-window model A for test rows) before approving any code.

| quantity | train window | test |
|---|---|---|
| rows with ≥ 1 labelled in-neighbour / out-neighbour / any | 0.528 / 0.542 / 0.780 | 0.397 / 0.615 / 0.746 |
| mean labelled in-count (= out-count) | 0.80 | 0.83 |
| share of a row's in-edges that lead to a labelled node | 0.47 | 0.41 |
| rows whose neighbour set changes under bridging (≤ 1 unlabelled hop) | 0.077 | 0.121 |
| rows that gain a first neighbour under bridging | 0.035 | 0.055 |
| nb_in_mean q90 (rows with a neighbour) | 0.54 | 0.013 |
| nb_out_max q90 (rows with a neighbour) | 0.99 | 0.017 |
| test rows "confident" for self-training (pA > 0.9 or < 0.02) | – | 0.94 (post-43: 100% of them below 0.02) |

- Count parity holds approximately: the labelled-only neighbour counts are distributed alike in
  the window and the test file. Counts carry label signal (0 labelled neighbours → 18% fraud in the
  window; ≥ 2 → 2.5–8%), but their gain share in model B is small. No corrective candidate.
- Bridging reaches 12% of test rows and gives a first neighbour to 5.5%. The train-only
  unlabelled-neighbour variant, which touched more rows, ranked 0.988 (unreadable). Expect the
  bridge file to fail the gate; build only when nothing else is queued.
- Self-training on test rows would add ~14,000 pseudo-licit rows and a few hundred pseudo-illicit
  rows, all pre-43; post-43 it reinforces the blind spot exactly as the unlabelled-row version did
  (−0.006). Evidence against; not built.
- The post-43 collapse is not driven by the features that shift most inside the test period
  (`artifacts/adversarial_post43.csv`: feat_88 0.80, feat_52 0.72, feat_53 0.68): dropping the top
  5 / 10 / 20 / 40 shifting features leaves the post-43 predicted rate at 1.1–1.4% (vs 0.95% with
  all features; 6.6–7.1% pre-43). Feature removal by within-test drift: closed without a slot.
- Model A's in-window per-step mean tracks the labelled fraud rate almost exactly under random
  OOF (0.088 vs 0.090 at step 22, 0.298 vs 0.299 at step 28, 0.043 vs 0.052 at step 33). Model B's
  test level: 0.023–0.134 for 36–42, 0.003–0.017 for 43–49 (single seed).
- Model B gain shares (single seed): feat_53 0.19, nb_in_mean 0.18, feat_55 0.09, nb_out_max 0.08;
  the six numbers together 0.30. feat_53 / feat_55 medians fall from 0.58 in the window to 0.21 /
  −0.01 in the test file (both halves), i.e. towards their fraud-side values, while predicted rates
  stay low.

## Status (2026-10-07, evening)

Candidates 1 and 2 as written were not built: the diagnostics above argue against both (bridging
reaches too few rows to pass the gate; self-training on test rows reinforces the post-43 blind
spot). Built instead, approved by the user: **`B15_nbfeat`** = means of the top-10 raw features
(step-1 gain order) over labelled in- and out-neighbours, 20 extra model-B columns, built exactly
like the six score numbers (`two_stage(..., nb_feats=10)`). The one input-side change never
board-read.

| file | CV 3-fold | hard | horizon | PR | Spearman vs best | verdict |
|---|---|---|---|---|---|---|
| fam_B15_nbfeat.csv | 0.9689 | 0.9139 | 0.8215 | 0.9099 | 0.973 | passes the gate: first upload of 2026-10-08 |

CV is below the best (0.9728 / 0.8420), as it was for every neighbour-information change that
later won on the board; it is a sanity pass, not a veto. Reading rule vs 0.95849: ≥ +0.002 gain,
≤ −0.002 loss. Queue behind it: `B15_bridge` only if built and gated; the third slot held.
