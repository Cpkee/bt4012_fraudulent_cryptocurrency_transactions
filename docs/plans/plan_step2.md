# Step 2 — Temporal drift and prior shift

Scope: option **F** from the brainstorm, plus the free stability gains (seed bagging, RF blend)
from **H** that any later step would want anyway. The GNN (step 3) and collective
classification (step 4) are out of scope here.

Starting point (from `results.md`): sub_03, LightGBM on 159 cleaned raw features, 300 rounds,
unweighted. Rolling CV 0.9819, public LB 0.9451, rank 19 of 20. Top is 0.9573, median 0.9494.

## Status (2026-09-26)

Executed in `main.ipynb` sections 11–15; module `drift.py` added; `validation.py` gained the
14-step horizon fold, the pooled-minus-per-step gap and sample-weight / postprocess hooks;
`models.py` gained bagging, weighted RF and rank-blend helpers. Submissions sub_04–06 written,
ledger rows appended under "Step 2" in `results.md`; public LB pending upload.

Outcome per section:
- §0 horizon fold: adopted. Step-1 model scores 0.894 there with gap −0.026 (0.982 on 7-step folds).
- §1 recency: **rejected**, every setting hurts, monotonically in history removed.
- §2 normalisation: **adopted**, rank-replace 9 continuous drifting features; horizon 0.894 → 0.917.
- §3 score correction: **rejected**; within-step rank, EM and BBSE all hurt; oracle ceiling +0.019.
- §4 novelty blend: **rejected**, hurts at every weight.
- §5 stability: bagging ×5 adopted (+0.001 everywhere); RF blend rejected.

Final: LightGBM 300 rounds, unweighted, 5-seed bag, 159 features (150 raw + 9 within-step ranks).
Rolling CV 0.9815 / hard fold 0.9575 / horizon 0.9173. Methodology and tables: `report_notes.md`.

---

## Why drift is the next lever

- The CV-to-leaderboard gap is 0.037. Our folds validate 7 steps ahead; the test window runs
  14 steps ahead and contains the step-43 dark-market shutdown.
- The hard fold (train 1–21, val 22–28) is already 0.03 below the other two. Whatever fixes it
  is what the test set needs.
- Adversarial validation still scores 0.997 on the cleaned raw features. `feat_2`, `feat_106`,
  `feat_109`, `feat_112` and others trend with time without being pure identifiers.
- Keeping the six period identifiers *gained* 0.002 on the public LB (sub_02 vs sub_01). Inside
  noise, but it says the time signal is not purely harmful: it should be transformed, not deleted.
- The competition pools steps 36–49 into one AUC. Fraud rate per step swings 0.4% to 36%, so the
  model's score scale has to be comparable across steps, not only its ranking within a step.

## 0. Make CV look more like the test set (do first)

Add a fourth fold with the test's horizon: **train 1–21 → val 22–35** (14 steps ahead). Report it
separately, and use it as the tie-breaker. Keep the three 7-step folds for continuity with step 1.

Also add to `validation.score()`: the gap `auc_pooled − auc_step_mean`. Positive gap means the
model ranks well inside steps but its scores are not comparable across steps. This number is the
target of section 3 below.

Noise floor: with ~550 positives on the public set, LB differences below 0.005 are noise.
CV differences below 0.002 on the 14-step fold are noise (check with 3 seeds).

## 1. Training window and recency (cheap, run first)

| Experiment | What changes |
|---|---|
| Drop early steps | train windows starting at step 5, 8, 12 instead of 1 |
| Exponential recency weights | `sample_weight = 0.5 ** ((t_max − t) / h)` for half-life h in {5, 10, 20} steps |
| Both | window start × half-life grid, coarse |

Expect: modest gain on the hard fold, possible loss on fold 1 (small training set). Pick by the
14-step fold.

## 2. Per-step feature normalisation

Test steps have 471 to 2,154 rows each, so per-step statistics are stable on both sides.

- For every raw feature compute within-step **rank** (0–1) and within-step **z-score**; add as
  new columns, or replace the original, as two variants.
- Restrict to drifting features first (single-feature adversarial AUC > 0.75 from
  `artifacts/single_feature_adversarial_auc_raw.csv`), then try all 159.
- **Bring the six period identifiers back as deviations**: `feat_101 − median_step(feat_101)`
  etc. Their per-step level is time; their within-step spread may be signal (they are neighbour
  aggregates of the time step, so the deviation says whether a transaction's neighbours are
  older or newer than its own step).
- Re-run adversarial validation on each variant. The goal is a lower AV AUC *and* a higher
  14-step-fold AUC. A variant that lowers AV AUC and loses CV is not adopted.

## 3. Score-level correction across steps

The pooled AUC on steps 36–49 depends on scores being comparable across steps.

- **Diagnostic**: on each validation fold, compare pooled AUC of raw scores with pooled AUC
  after replacing each score by its within-step rank. If ranking helps, scales are inconsistent.
  If it hurts, the model is legitimately encoding the per-step base rate and ranking destroys it.
- **Prior-shift correction** (Saerens et al. 2002, EM): estimate each test step's fraud prior
  from the model's own predictions, then Bayes-adjust the scores so that step-level prior
  differences are reflected consistently. Validate on the 7-step and 14-step folds, where the
  true per-step priors are known: compare the EM estimate against the truth per step, and the
  adjusted pooled AUC against the raw one.
- Simpler alternative for comparison: BBSE (Lipton et al. 2018) with a confusion matrix from the
  training tail.
- Implementation: `drift.py` with `estimate_priors_em(p, train_prior)`, `adjust_scores(p, prior_train, prior_new)`, `rank_within_step(p, step)`.

## 4. Novelty component (only if 1–3 leave the hard fold weak)

Isolation Forest, or a one-class model, trained on licit training rows only; rank-blend its
anomaly score with the LightGBM score at weights {0.1, 0.2, 0.3}. Hypothesis: post-shutdown
illicit activity looks anomalous even when it does not look like old fraud. Keep only if it
improves the 14-step fold by more than the noise floor.

## 5. Stability gains (free)

- 5-seed bagging of the final LightGBM (average probabilities).
- Rank-average blend with Random Forest at weights {0.2, 0.3}; RF had the best PR-AUC in step 1.
- Both evaluated on the same folds; adopt if the 14-step fold improves.

## 6. Submissions for this step (3 per day)

| File | Content | Purpose |
|---|---|---|
| sub_04 | best config from sections 1–3 | main candidate |
| sub_05 | sub_04 + section 5 bagging and blend | stability on top |
| sub_06 | one control: sub_03 + recency weights only | prices a single change |

Log every row in `results.md` with the 14-step-fold AUC as an extra column.

## 7. Report exhibits this step produces

- Per-step validation AUC curve for sub_03 vs the drift-corrected model (section 2.2 evidence).
- Table: AV AUC and 14-step-fold AUC per normalisation variant.
- EM prior estimate vs true prior per validation step (shows the correction is grounded).
- Public LB ledger with CV next to it, showing the CV-to-LB gap shrinking or not.

## Code changes

| File | Change |
|---|---|
| `validation.py` | add the 14-step fold, add the pooled-minus-step-mean gap to `score()`, accept `sample_weight` in `evaluate()` |
| `drift.py` (new) | recency weights, per-step normaliser (fit on any split, transform per step), EM prior estimation, Bayes score adjustment, within-step rank |
| `models.py` | LightGBM factory that accepts sample weights; RF blend helper |
| `main.ipynb` | section 11: CV update, 12: recency, 13: normalisation, 14: score correction, 15: stability and submissions |

## Verification checklist

- [ ] 14-step fold added and reported for sub_03 (the new baseline row)
- [ ] Recency grid table with 3 seeds; chosen setting stated with the fold numbers
- [ ] Normalisation variants: AV AUC and CV side by side; period identifiers as deviations tested
- [ ] Score-rank diagnostic reported per fold; EM priors compared with true priors
- [ ] sub_04 to sub_06 written, pass the four sanity checks, logged in `results.md`
- [ ] Public LB filled in; per-change deltas summarised in one paragraph

## What this step leaves out

GNN (step 3), collective classification (step 4), self-supervised pretraining (step 5).
Do not tune to the public LB: use it only to confirm direction.
