# Step 6 — Next round on the random-OOF two-stage model (plan, no code yet)

Date: 2026-10-05. Best: `fam_B15_Aoof_random.csv` = **0.95849**, 6th of 20 (top 0.96805; five
entries above us, 0.9609–0.9681). Objective unchanged: keep improving the public score.

## What the last three readings established

| training-row neighbour numbers made from | score | reading |
|---|---|---|
| model-A scores, time-ordered OOF (previous best) | 0.95621 | – |
| model-A scores, **random 5-fold OOF within the window** | **0.95849** | +0.0023, new best |
| half true labels, half scores | 0.95423 | −0.0020 |
| true labels only | 0.94620 | −0.0100 |

- Label information in the training rows' neighbour numbers hurts monotonically: at test time
  those numbers are scores, so training on anything closer to labels than a score mismatches.
- Random OOF beats time-ordered OOF. Interpretation: a time-ordered scorer for step *s* has seen
  only steps < *s* and is weaker (older data, less of it); a random-OOF scorer has seen 4/5 of the
  window including step *s*'s own period, so its scores resemble what the test-row scorer (trained
  on the whole window) produces. The closer the training-row scores are to the test-row scores in
  *quality*, the better model B transfers.

That interpretation gives the next axis: **make the training-row scorer and the test-row scorer
as alike as possible.**

## Candidates for the next three uploads (one change each on `B15_Aoof_random`)

| name | change | why | expected readability |
|---|---|---|---|
| `B15_Aoof10` | 10-fold random OOF instead of 5-fold | each scorer sees 90% of the window, closer still to the test-row scorer (100%) | borderline (0.98–0.99); build and gate |
| `B7_Aoof_random` | model B at 7 leaves under the new protocol | the depth optimum was mapped under time-ordered scores; sharper neighbour numbers may shift it | readable (depth moved the ranking to 0.96 before) |
| `recent21_Aoof_random` | 21-step window under the new protocol | the window curve was mapped on the single-stage model; with random OOF, more rows means better scorers | readable (window moved the ranking to 0.97 before) |

Reading rule as before: vs 0.95849, +0.002 or more is a gain (seed noise 0.001), −0.002 or worse
a loss, between is a tie that keeps the simpler model.

## After those readings

- If 10-fold gains: try 20-fold / leave-one-step-out hybrid; if unreadable, stop at 5.
- If 7 leaves gains under the new protocol: re-map 5 / 11 leaves; if 15 holds, depth stays closed.
- If 21 steps gains: 28 and 35; if not, 14 stays.
- Two further ideas on the same axis, held in reserve: (a) test-row scorer = average of the 5
  random-OOF fold models (so test rows are scored exactly like training rows, by models that
  saw 80% of the window); (b) training-row scores from a *bag* of random partitions (3 different
  5-fold splits averaged) — likely unreadable, build only if (a) is.

## Housekeeping done with this plan

- `final_model.ipynb` and `final_model_kaggle.ipynb` switched to the random-OOF protocol
  (`CONFIG["oof"] = "random"`, 5 folds); each checks itself against `fam_B15_Aoof_random.csv`.
- Ledger, `BOARD` and CLAUDE.md updated; the label curve recorded as a closed axis.
