# Step 5 — Model families on the leaderboard, then freeze

Replaces the step-4 probe campaign. Deadline for the report and final selection: 25 Oct 2026.

## Status (2026-09-27)

- §0 ledger fixes done (results.md, report_notes.md §3 and line ~149, plan_step4.md closed).
- `families.py` written; `ref` reproduces sub_06 exactly (max |diff| 1.1e-16).
- All candidates built (`submissions/fam_*.csv`, ledger table "Step 5" in `results.md`):
  - pass the upload gate: `rf_clean` (closest scored file 0.873) and `et_clean` (0.783);
    the two differ from each other at 0.827;
  - too close to sub_06: `lgbm_deep_raw` (0.983) and `lgbm_raw_ids` (0.995). Neither the
    depth nor the period identifiers change the LightGBM ranking much;
  - `logreg` fails the sanity gate (3-fold 0.909);
  - `blend_06_07` fails the gate at 0.985 against sub_04, sub_06 and sub_07. A 50/50 blend of
    two files is always close to both, so blends belong to the finals decision, not the probes.
- 2026-09-28: `fam_rf_clean.csv` scored 0.92847 on the public board, −0.0153 vs sub_06 (about
  3x the noise). By the reading rule the forest family is rejected and LightGBM on the cleaned
  raw features stands. `et_clean` has CV close to the random forest (3-fold 0.965, horizon
  0.825), so it is not uploaded.
- CV was wrong about small changes within LightGBM but right about this large gap between
  families (−0.019 on the 3-fold mean, −0.078 on the horizon fold).
- Finals by the rule: **sub_06** (cleaned raw LightGBM, bagged x5) and **sub_07** (v2b tuned
  LightGBM, bagged x5; Spearman 0.942 with sub_06). Both must be selected by hand on Kaggle.

---

## Step 6 (2026-09-29): hypotheses that historical CV rejects, read on the board

Reference re-based on sub_06 (0.94374); distinctness gate kept. First result:
**fam_recent14** (sub_06 recipe trained on steps 22–35 only) scored **0.94836**, +0.0046 vs
sub_06, the largest positive delta of the project and opposite to its −0.016 on the horizon
fold. Moves us off last place (bottom of board 0.9475). Next: map the window (recent7, recent10,
recent21 built; recency_hl5 built) one upload each; then combine the best window with the
untested two-stage neighbour-prediction features. `fam_all_time` (165 columns + time_step) was
too close to sub_06 (0.995) to upload: LightGBM barely uses time-carrying columns.

---

## Why this step

- **No change selected on CV has beaten the simplest model on the public board.** sub_03 (cleaned
  raw features, default 31-leaf LightGBM, one seed) scores 0.9451. Normalisation, tuning and the
  drift probes all landed between 0.938 and 0.944. Every entry on the public board is above
  0.9462, and the top is 0.9636.
- **Every upload so far is LightGBM.** CV cannot see steps 43–49, where all our models score
  almost every row as licit, so CV cannot choose between model families. Only the board can.
- **Step-4 probes were too similar to the reference to read.** They had Spearman ≥ 0.98 against
  it, and the adoption rule asked for +0.008, so the uploads could only come back as noise.
- **Measured distinctness of what is already on the board** (Spearman between test predictions):

| pair | Spearman | reading |
|---|---|---|
| sub_03 vs sub_06 (same recipe, 1 vs 5 seeds) | 0.996 | same model; public delta 0.0014 = pure noise |
| sub_06 vs sub_04 (raw vs v2b ranks) | 0.988 | near-identical |
| sub_06 vs sub_07 (raw default vs v2b tuned) | 0.942 | genuinely different |
| sub_02 vs sub_07 (raw + identifiers vs v2b tuned) | 0.909 | the most different pair on the board |

So re-bagging sub_03 (the earlier "lgbm_raw_bag10" idea) is dropped: it would be sub_06 again.

## Rules

- **Reference bar**: sub_06 = 0.94374 (the bagged version of sub_03; a 5-seed bag is what a
  final submission would use).
- **Upload gate: distinctness, not a CV veto.** Upload a candidate only if its Spearman against
  every file already scored is < 0.98. Anything above that is invisible to the public board.
- **Sanity gate**: mean 3-fold CV AUC ≥ 0.95 (two seeds). This catches broken pipelines only;
  horizon and far-window CV are reported for the write-up but decide nothing.
- **Reading rule**, fixed before any score is seen:
  - public delta vs sub_06 > +0.005 → signal: the next day's uploads go to variants of that family;
  - within ±0.005 → tie; ties go to the simpler, bagged model;
  - paired noise for pairs with Spearman ≈ 0.94 was measured at ~0.005, so no single delta
    under that counts.
- **Finals**: two submissions from different families. Chosen by 5 Oct, then modelling freezes.

## 0. Ledger fixes (before any new work)

| file | fix |
|---|---|
| `results.md` rows for sub_05 / sub_06 | swap public LB: sub_06 = 0.94374, sub_05 = 0.93977 (Kaggle API) |
| `report_notes.md` §3 (lines ~575–576) | normalisation −0.0017 vs sub_06 (not +0.002); discrete ranks −0.0022 vs sub_04 (not +0.002); bagging −0.0014 vs single seed (not −0.005) |
| `report_notes.md` line ~149 header | the board has more than 20 entries and the top is 0.9636 as of 27 Sep |
| `plan_step4.md` | line 17: only `probe_adv_weights_hard` was uploaded (0.93765); mark the campaign closed and point to this plan |

## 1. Candidates

All of them use the 159 cleaned raw features (period identifiers removed, no rank columns)
unless stated otherwise, and are unweighted.

| name | model | notes |
|---|---|---|
| `blend_06_07` | rank average 50/50 of sub_06 and sub_07 files | no training; the two differ most among bagged files |
| `rf_clean` | RandomForest, 500 trees, `balanced_subsample`, min_samples_leaf 2 (`models.rf_fit_predict`), 3 seeds averaged | first non-LightGBM family on the board |
| `et_clean` | ExtraTrees, same settings | more randomised splits than RF |
| `lgbm_deep_raw` | sub_06 recipe with 127 leaves, min_child_samples 20, bagged ×5 | the opposite direction to the step-3 tuning |
| `lgbm_raw_ids` | sub_06 recipe + the six period identifiers (165 features), bagged ×5 | sub_02 beat sub_01 with identifiers kept; this is the fair, bagged version |
| `logreg` | `models.logreg_fit_predict` | low priority; only if the tree families tie |

## 2. Schedule

| day | uploads | decided by |
|---|---|---|
| 28 Sep | `blend_06_07`, `rf_clean`, and whichever of `et_clean` / `lgbm_deep_raw` / `lgbm_raw_ids` has the lowest max-Spearman against the board | distinctness gate |
| 29 Sep | winner signalled → two variants of it + one confirmation (another seed set); no signal → the remaining candidates | reading rule |
| 30 Sep – 4 Oct | at most one follow-up per day (e.g. 3-way blend of the best families); remaining uploads kept spare | reading rule |
| 5 Oct | choose the two finals; freeze modelling; write the report | finals rule |

## 3. Code

- **New `families.py`** (about 120 lines). Leave `probes.py` untouched; it records the finished
  campaign, and other sessions may be using it. It reuses `probes.Data`, `validation.evaluate`
  and `models`.
  - `.venv/bin/python families.py rf_clean et_clean ...` builds each candidate. For each one it
    reports CV on the four standard folds (two seeds), fits on all labelled rows, writes
    `submissions/fam_<name>.csv` and prints Spearman against every uploaded file. It also appends
    a row to a new "Step 5: model families" table in `results.md`, with the public score left
    `_pending_`.
  - `.venv/bin/python families.py blend sub_06 sub_07` rank-averages existing files.
- **No change to `models.py`**. ExtraTrees and the deep LightGBM are defined inside
  `families.py`.
- **Notebook**: after the finals are chosen, add one section that reproduces them end to end.
  The brief requires this; it becomes the student-number notebook.

## 4. Verification checklist

- [ ] §0 ledger fixes made; `results.md` public scores match the Kaggle API for every file
- [ ] `families.py` reproduces sub_06 exactly (max |diff| < 1e-9) as its reference check
- [ ] Each candidate: sanity gate passed, Spearman table printed, ledger row written
- [ ] Day-1 uploads logged with public score and delta vs sub_06
- [ ] Finals chosen by the rule above and recorded with their family and reasoning
- [ ] Report §2.3 gains a "families on the board" table: CV columns next to public LB

## What this step leaves out

More LightGBM tuning, drift probes, GNN variants, semi-supervised variants. These are finished
results for the report, not levers for the score.
