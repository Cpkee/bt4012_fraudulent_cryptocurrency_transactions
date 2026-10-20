# Plan step 12 — Group DRO for model B (worst-step training)

Status: **approved 2026-10-18; code in `two_stage(b_dro=...)`, family `B15_nbfeat_all_goof_xt_dro`.** Builds on the `xt` recipe
(`B15_nbfeat_all_goof_xt`, 0.96678, replicate `xt_seedsBC` to be read 2026-10-19).

## 1. The problem

Every model so far minimises the **average** loss over the labelled rows of the 14 window steps (22–35). The test
period is not an average window step:

- from step 43 every model predicts almost no fraud (mean predicted rate 0.022 vs 0.085 for 36–42); no window step
  looks like that;
- the window itself is heterogeneous. Steps differ in size and fraud rate:

| step | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| labelled rows | 1763 | 1187 | 1126 | 594 | 517 | 206 | 284 | 1174 | 524 | 710 | 1323 | 441 | 515 | 1341 |
| illicit | 158 | 53 | 137 | 118 | 96 | 24 | 85 | 329 | 83 | 106 | 342 | 23 | 37 | 182 |
| fraud rate % | 9.0 | 4.5 | 12.2 | 19.9 | 18.6 | 11.7 | 29.9 | 28.0 | 15.8 | 14.9 | 25.9 | 5.2 | 7.2 | 13.6 |

Under average-loss training, steps 22, 29, 32 and 35 (5,601 rows, 1,011 frauds) dominate the gradient; steps 27,
33 and 34 (1,162 rows, 84 frauds) barely register. Whatever pattern separates fraud in the big, high-fraud steps is
what the trees learn, whether or not it holds in the small, low-fraud ones.

Measured in this data (2026-10-18): 79 of 153 kept features have a per-step AUC on the "wrong" side of 0.5 in at
least 20% of window steps; `feat_22` has pooled AUC 0.37 but 0.53–0.55 in steps 24, 31, 32. A model can lean on such
a feature and still score well on average, because the steps where it fails are few or small.

**The problem Group DRO solves:** average-case training is allowed to be bad on a minority of conditions. When the
deployment conditions are not a re-draw of the training mix, which is our case, the minority conditions are the ones
that matter.

## 2. Intuition

Think of the 14 steps as 14 different "environments". Ordinary training asks: *which model is best on the pooled
data?* Group DRO (distributionally robust optimisation over groups; Sagawa et al. 2020, cited for the method) asks:
*which model is best on the hardest environment?* It does this by repeatedly finding the steps where the current
model ranks worst and giving them more weight, so the next fit must also work there.

The effect is a pressure toward **patterns that hold in every step** and away from patterns that only hold in the
dominant ones. A pattern that held in all 14 recent steps is a better bet for steps 36–49 than one that held in 10.
That is the same goal `extra_trees` may be serving from a different angle: `extra_trees` stops the model trusting
exact cut-points, DRO stops it trusting step-specific patterns.

What it does **not** do: it cannot invent a regime the window never shows. It reweights among the 14 seen steps.
If the hardest steps turn out to be the low-fraud ones (23, 33, 34: 4.5–7.2%), DRO moves the model toward the
window's closest analogue of the post-43 regime, but only by reweighting rows the model already trains on.

## 3. Method

Groups g = the 14 window steps. Row weight w_i = q_g(i) · N / n_g(i), with q a distribution over steps.

1. **Start** from q_g = n_g / N, i.e. every row weight 1: round 0 is exactly the `xt` model (wiring check).
2. **Measure per-step risk** with out-of-fold model-B predictions on the window rows: 5 inner folds grouped by
   connected component of the labelled window graph (the same grouping as `a_oof="group"`, so a row's neighbours'
   labels do not leak into its own fold). Model A and every model-B input are computed once beforehand, so each
   round only refits model B (15 leaves, 300 rounds; a few seconds per fit).
   Risk R_g = 1 − AUC_g (the metric is ranking; a log-loss risk would be dominated by each step's base rate).
   Shrinkage by measured noise (empirical Bayes): R_g ← R̄ + (τ² / (τ² + se_g²)) · (R_g − R̄), se_g = the Hanley–McNeil
   standard error of step g's AUC, τ² = max(0, var(R) − mean(se²)) the between-step variance beyond noise. Measured
   se at AUC 0.95: 0.008 (steps 29, 32, ~335 frauds) to 0.031 (steps 27, 33, 23–24 frauds); if τ² = 0 the steps are
   indistinguishable from noise and DRO leaves the weights at 1 (the build reports this).
3. **Update** q_g ← q_g · exp(η · (R_g − R̄) / sd(R)), renormalise. Standardising R makes η a strength, not a scale.
4. **Cap:** clip q so that max_g w / min_g w ≤ ρ = 8, and require effective sample size
   ESS = (Σw)² / Σw² ≥ 65% of the rows; stop early if a round would break either. Both limits come from weightings
   already read on the board (§4a): equal-per-step weights reached 8.6× at ESS 69%, recency weights 6.1× at 73%.
5. Repeat steps 2–3 for T = 3 rounds.
6. **Final fit:** model B on all window rows with the final weights, seed-bagged x5 as usual; test rows predicted
   exactly as now. Only model B's row weights change; model A, the inputs, column order and `extra_trees` stay.

One strength: **ρ = 8, η = 1.0**. A milder cap would be pointless: equal-per-step weights already moved rows 8.6×
and ranked 0.986 like their parent (unreadable). If the ρ = 8 build also ranks ≥ 0.98 like `xt`, step reweighting
does not move model B's ranking and the axis closes without a slot.

## 4a. Where the limits come from (computed 2026-10-18 on the window)

| weighting | max/min row weight | ESS | board |
|---|---|---|---|
| unweighted (`xt`) | 1.0 | 100% | 0.96678 |
| equal weight per step (`b_step_balance`) | 8.6 | 69.3% | 0.986 vs parent, unreadable |
| recency half-life 5 | 6.1 | 72.8% | tie |
| DRO cap 2 / 4 / 8 (worst case) | 2 / 4 / 8 | ≥ 88.9% / 64.0% / 39.6% | – |

The first draft's 2× cap and 60% floor were unanchored defaults: a 2× cap can never bring ESS below 89%, so the floor
never bound, and 2× is far milder than weightings that already failed to move the ranking.

## 4. How this differs from what was tried

| tried | what it did | result | why DRO is different |
|---|---|---|---|
| `b_step_balance` (step 9) | every step the same total weight, fixed | 0.986 vs parent, unreadable | weights not fixed by size but driven by where the model fails; can push past equal |
| recency weights (half-life 5) | old steps down-weighted by age | tie with the window | weights by difficulty, not by age |
| adversarial importance weights | test-like rows up-weighted | −0.003 | uses no test rows at all; label-driven |
| low-rate splice (step 11) | trained on low-fraud steps only | −0.014 (splice) | keeps every window row; capped reweighting, not a different training set |

The last row is the main warning: regime-matched training ordered the post-43 block *worse*. Hence the cap, the
ESS floor and the mild strength first.

## 5. Risks and mitigations

- **Noise chasing.** The worst step may be worst because it is small. Shrinkage by each step's measured AUC standard
  error, standardised updates, the cap and only 3 rounds.
- **No effect.** Step reweighting at 8.6× did not move the ranking before (0.986). The gate decides; no slot is spent
  on an unreadable file.
- **Period level.** Up-weighting low-fraud steps lowers the effective base rate. A uniform shift is harmless to AUC,
  but the level *between* periods is worth 0.022 (steprank probe). Check: post-43 share of the top 5% by rank must
  stay near `xt`'s 0.111 (the scored range is 0.093–0.132).
- **Fewer effective rows.** ESS floor of 65% (inside the tested 69–73% envelope's neighbourhood).
- **CV cannot judge it.** The rolling folds share the training period's mix; CV is a sanity check only (3-fold ≥
  0.95). The board decides, as for `xt`.

## 6. Checks before any upload

1. η = 0 reproduces `fam_B15_nbfeat_all_goof_xt.csv` exactly (max |diff| = 0).
2. Log per round: per-step OOF AUC, q, max/min ratio, ESS. Report which steps gained weight.
3. CV 3-fold / hard / horizon vs `xt` (0.9652 / 0.9048 / 0.8124).
4. Spearman vs every scored file < 0.98 (automatic gate), and vs the blend members (goof15, blocks15, xgb15).
5. Post-43 top-5% share (level check above).

## 7. Reading rule and follow-ups

Read against `xt` (or against the mean of `xt` and `xt_seedsBC` if the replicate disagrees with 0.96678):

- **≥ +0.003:** signal. Seeds 5–14 → `xt_dro_bag15`; add the option to both notebooks; then try DRO on `blocks_xt`.
- **within ±0.003:** tie. The simpler model (`xt`) stays; DRO kept as a blend-member candidate if it ranks < 0.95
  like each member and scores ≥ 0.962 alone.
- **≤ −0.003:** rejected. If mild was readable and lost, strong is not tried; the step-weighting axis closes
  (size, age, test-likeness and difficulty all read).

## 8. Code changes (on approval)

- `src/families.py`, `two_stage`: new arguments `b_dro: float = 0.0` (η), `b_dro_rho: float = 8.0`, `b_dro_ess: float = 0.65`,
  `b_dro_rounds: int = 3`. A block next to `b_step_balance` (≈ line 539) that computes `wB` by the loop in §3,
  reusing the component grouping already built for `a_oof="group"` and `fit_B` for the inner fits. Per-round
  diagnostics printed and saved to `artifacts/dro_<name>.csv`.
- `FAMILIES`: `B15_nbfeat_all_goof_xt_dro` (η 1.0, ρ 8, ESS ≥ 65%).
- No notebook change until a DRO file wins on the board.
- Cost: model A as now (~25 min) plus 3 rounds × 5 inner fits × 5 seeds of model B, a few minutes. The CV run
  repeats the loop inside each fold (inner folds within that fold's training window).

## 9. Decisions for approval

1. Base recipe: `xt` (proposed), whatever the replicate says, since it is the highest-scoring model.
2. Risk measure: 1 − per-step AUC, shrunk by its measured standard error (proposed) vs normalised log-loss.
3. Limits: cap ρ = 8, ESS ≥ 65%, one build (proposed; revised from 2× / 60%, see §4a).

## 10. Re-tuning later

The step weights are recomputed from the data on every build (saved per round to `artifacts/dro_<name>.csv`); only
the settings that govern them are chosen: `b_dro` (η), `b_dro_rho` (cap), `b_dro_ess` (floor), `b_dro_rounds`. A
variant is one `FAMILIES` line; if adopted, the same four go into the notebooks' `CONFIG`. Tuning only after the
ρ = 8 build reads positive, as the window was tuned: neighbours (ρ 4 / 16, η 0.5 / 2) through the gate first (free),
then the board one change at a time in the direction that pays, reading rule fixed before each upload. A worst-step
CV score (each fold's lowest per-step AUC) may be logged as a sanity check; it does not decide. If the first build
fails the gate or loses, there is nothing to tune.

## 11. First run (2026-10-18): wiring passes, risk measure finds nothing to reweight

- **Wiring check passed:** the DRO loop with η = 0 reproduces `fam_B15_nbfeat_all_goof_xt.csv` to 1.1e-16.
- **Design flaw found before the build finished:** with grouped 5-fold OOF *inside* the window, model B scores every
  step almost perfectly (pooled OOF AUC 0.9973–0.9979 over seeds 0–4; worst step 23 at 0.980–0.983). The between-step
  spread is within noise (τ² = 0 on all five seeds), so shrinkage sets every risk to the mean and the weights stay 1:
  the build would return `xt` unchanged. The build was stopped (no ledger row, no file).
- **Cause:** in-period folds put rows of the same step in the training folds, so no step looks hard. The difficulty
  DRO is meant to target — transfer to a period the model has not seen — is invisible to that risk measure.
- **Proposed fix (needs approval, changes decision 2 of §9):** risk from **leave-one-step-out** model-B scores — each
  window step scored by model B trained on the other 13 steps (the `a_oof="step"` pattern). That measures how well
  model B carries over to an unseen period, the test-time situation. Same shrinkage, cap, floor and rounds; a
  one-seed diagnostic first (does τ² become > 0, which steps come out worst) before any full build.

## 12. Leave-one-step-out diagnostic (seed 0, 2026-10-18): closed

Per-step AUC of model B when the step is held out (trained on the other 13 window steps):

| step | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.998 | **0.978** | 0.995 | 1.000 | 0.994 | 0.998 | 1.000 | 0.999 | 0.996 | 0.998 | 1.000 | **0.989** | 1.000 | 0.998 |
| s.e. | 0.003 | 0.014 | 0.004 | 0.002 | 0.006 | 0.006 | 0.002 | 0.001 | 0.005 | 0.003 | 0.001 | 0.015 | 0.004 | 0.002 |

Pooled 0.9971; τ² = 0 again, weights stay 1. The two worst steps are the two lowest-fraud steps (23: 4.5%, 33: 5.2%),
but each sits within about 1.5 standard errors of the rest. **Model B carries over to an unseen window step almost
perfectly; the window steps are equally easy.** The difficulty of the test period is distance in time (CV horizon fold
0.81 for steps 14 ahead), not which window step a row resembles, and reweighting by distance is the recency weighting
already read (tie). Caveat: model A's inputs for a held-out step still come from in-window folds, so this is an upper
bound on transfer; a fully held-out version would be the soof pipeline (`a_oof="step"`), whose 15-seed bag tied.

**Step reweighting is closed on size, age, test-likeness and difficulty.** No DRO file is built or uploaded. The code
stays (`two_stage(b_dro=..., b_dro_risk=...)`, default off; η = 0 reproduces `xt` to 1.1e-16).
