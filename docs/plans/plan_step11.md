# Step 11 — Plan: the post-43 ordering campaign

Date: 2026-10-12 (user-approved: all 3 slots/day this week; offline pre-screen approved as a
ranker only; regime-matched trainer first, then the other candidates if no improvement).

## Position

The floor probe (2026-10-11, `fam_bag15_post43floor` = 0.89203, −0.071) showed the post-43 block
holds ≈ 27–49 public fraud rows (1.3–2.4%) that Final 1 ranks above most pre-43 licit rows. The
within-block ordering term has weight ≈ 0.06–0.10 of pooled AUC, so a better post-43 ordering is
worth up to ≈ +0.01–0.02 — about the size of the 0.005 gap to the leader. Board noise on the term
is ≈ 0.004 (~35 positives): a floorsplice reading is a signal only at |Δ| ≥ 0.005 vs 0.89203,
i.e. a within-block ΔA22 ≳ 0.05. Only large ordering changes are readable; candidates must order
the block very differently from Final 1 (within-block Spearman well below ~0.9).

## The new measurement (2026-10-12)

Labelled fraud rate per training step: the only steps below 3% are **1–6 and 10**
(0.43–2.08%; 9,322 labelled rows, 107 positives); below 5% adds steps 12 and 23 (11,015 rows,
176 positives). Every step of the recent-14 window runs 4.5–36% except 23 (4.5%) and 33 (5.2%).
**Final 1's post-43 order comes from a model that never saw a period with a post-43-like base
rate, while the training data contains such periods — in the early history the window drops.**
This is the data-grounded case for regime-matched candidates, and independently for the queued
`fam_post43floor_sub06` (all-history) splice.

## Protocol

Every candidate is read through the floorsplice (`post43_probes.py floorsplice
fam_B15_nbfeat_all_bag15 <candidate>`): pre-43 rows = Final 1, post-43 block floored with the
candidate's order. Difference vs 0.89203 = the candidate's within-block quality, cross-block
terms held fixed. A candidate that reads ≥ +0.005 becomes "Final 1 with its post-43 order,
mapped onto Final 1's post-43 value range" (the level stays: both level probes said it is
right), which is then read as a full Final-1 candidate.

**Donor files are not upload candidates themselves** — they may fail the CV sanity gate (a
low-rate trainer has no high-rate competence) and that is fine; only their post-43 order is
used. The whole-file distinctness gate does not apply to floorsplice files (they are ~0.99 like
the floor file by construction); their readability condition is the within-block Spearman.

## Offline pre-screen (ranker only — the board remains the only evidence)

`src/post43_prescreen.py`: the only in-data analogue of the post-43 block is the low-rate band
E = steps {1–6, 10} (107 positives). Candidates that train without E are scored by their pooled
AUC within E; the regime-matched trainer is scored leave-one-step-out within the band (train on
the other low-rate steps, score the held-out step, pool). Recorded caveat: the proxy is
anti-chronological (a future-trained model ordering the past) and historical CV has been wrong
on temporal choices; the pre-screen only decides which candidates get slots first.

## Candidates, in order (one change per upload)

1. **`fam_post43floor_sub06`** (built 2026-10-11, queued): all-history single-stage order.
   Reads whether more history alone improves the block.
2. **`fam_B15_nbfeat_pseudo_pre43`** (gated 2026-10-12): the transductive candidate, read as a
   full file vs 0.96069 (its own axis); its block order is also a later floorsplice donor.
3. **Regime-matched trainer** (`lowrate` family): sub_06-style LightGBM trained on the labelled
   rows of the low-rate steps only ({1–6, 10, 12, 23}, 176 positives; depth per pre-screen),
   bagged. Donor for `fam_post43floor_lowrate`.
4. **kNN fraud-similarity order**: rank post-43 rows by distance-weighted similarity to window
   fraud rows — the most different order available, so the most readable.
5. **Post-43 confident-licit pseudo-negatives** into model B (teaches model B what post-43
   licit looks like); block order via floorsplice.
6. **Zero-build splices** of the most-different scored files (sub_04 0.845, recent14_local
   0.853) if the pre-screen ranks their orders well.

## Reading rules

- Floorsplice file vs 0.89203: ≥ +0.005 → the candidate orders the block better; promote to the
  mapped full candidate. ≤ −0.005 → worse; next candidate. Between → unreadable; next candidate.
- Mapped full candidate vs 0.96313 (Final 1): standard ±0.003 rule.
- Exit: three floorsplice losses/ties in a row from different mechanism families (history,
  regime-matched, similarity, transductive) → the block order is as good as the data allows;
  close the campaign, finals stand, report takes the time.

## Slot schedule (3/day, week of 2026-10-12)

- 2026-10-12: slot 1 `fam_post43floor_sub06`, slot 2 `fam_B15_nbfeat_pseudo_pre43`,
  slot 3 `fam_post43floor_lowrate` (if gated in time).
- Then one new donor per day through the floorsplice, ordered by the pre-screen; remaining
  slots for promotions of any winner and for the pseudo-pre43 axis if it opens.

## Status after the 2026-10-12 readings (all three losses)

| file | read | delta | meaning |
|---|---|---|---|
| fam_post43floor_sub06 | 0.88509 | −0.0069 | all-history orders the block *worse* than Final 1 (ΔA22 ≈ −0.07 to −0.12) |
| fam_post43floor_lowrate | 0.87796 | −0.0141 | regime-matched orders it much worse (ΔA22 ≈ −0.14 to −0.23) |
| fam_B15_nbfeat_pseudo_pre43 | 0.95553 | −0.0052 | transductive axis closed with a board reading |

Final 1's own block order beats both motivated challengers. **The pre-screen is discredited as a
ranker**: it predicted regime-matched ≫ all-history ≫ window and the board read the exact
reverse — the low-rate-band gain was era proximity and sibling-step matching, not transferable
regime knowledge. One more offline proxy wrong about a temporal question; recorded for the report.
pseudo_pre43's within-block Spearman vs Final 1 is 0.936: its splice could never read; dropped.
kNN and the zero-build splices lose their motivation with the pre-screen; held unless the
calibration says the campaign continues and the pseudo-negatives donor also loses.

## The calibration step (2026-10-13)

- **Slot 1: `fam_bag15_post43floorrev`** (built, gated): Final 1's block order reversed under the
  floor. Δ vs 0.89203 = −w·(2·A22−1), w ≈ 0.06–0.10 → **A22 = 0.5 − Δ/(2w)**, Final 1's absolute
  within-block AUC. Worked examples: Δ = −0.022 → A22 ≈ 0.61–0.68; Δ = −0.045 → A22 ≈ 0.72–0.87;
  Δ = −0.06 → A22 ≈ 0.80–0.95. Decision rule: A22 ≥ ~0.85 (Δ ≤ ≈ −0.055) → remaining headroom
  ≲ 0.01 and both challengers already lost: **close the campaign for good**, finals stand, report
  takes the time. A22 ≤ ~0.75 → real headroom remains; continue with mechanism-diverse donors.
- ~~Slot 2: floorsplice of `B15_nbfeat_pseudo_p43neg`~~ **built 2026-10-12 and dropped**: CV =
  the parent's exactly (wiring check passed), but the within-block Spearman is **0.926** vs
  Final 1 — the pseudo-negatives reshuffle the full file (0.937) yet barely reorder the block,
  so the splice can never clear the noise floor. Pattern now measured three times (pseudo_pre43
  0.936, p43neg 0.926, every scored file 0.84–0.96): **every variant of the current pipeline
  orders the block essentially like Final 1**; only models from outside the pipeline (sub_06
  0.844, lowrate 0.687) differ, and both ordered it worse.
- **Slot 2 (contingent on the calibration): the `B15_nbfeat_pseudo_p43neg` full file** vs
  0.96069 — it passes the whole-file gate (0.960) and is the one untested mechanism left, but
  weakly motivated (its sibling pseudo_pre43 read −0.0052). Upload only if floorrev says the
  campaign continues; if the campaign closes, hold it and return to report work with one
  exploratory slot a day as before.
- Slot 3: held for the follow-up (a winner's promotion, or nothing).

## Round 2 (user-approved 2026-10-13): two kind-changes on the neighbour-feature axis, gate-only

The axis was closed at step 9 on amount, statistic, entry point and reach (five unreadable gates;
one tie-band gain in seven attempts since the +0.0022 adoption). The user re-opened it for the two
variants that change the *kind* of information — the only move that ever worked on this axis:

1. **`B15_nbfeat_scorew`** (`two_stage(nb_feat_scorew=True)`): score-weighted neighbour feature
   means, sum(s_j·x_j)/sum(s_j) with s_j = the neighbour's model-A score — the suspicious-neighbour
   profile, coupling the two proven information sources into a ratio of sums the trees cannot form.
2. **`B15_nbfeat_top1`** (`two_stage(nb_feat_top1=True)`): the raw features of the single
   highest-scoring in- and out-neighbour — a selection, not an aggregate (per-feature max was
   unreadable; the argmax-neighbour's vector was never tried).

Both are one change each on the 0.96069 reference (nb_feats=10), 20 extra model-B columns appended
after the existing block (column-order discipline). **No upload slot is spent unless a gate passes**
(Spearman < 0.98 vs every scored file). A pass takes slot 2 of 2026-10-13 behind the floorrev
calibration, displacing the weakly-motivated contingent `p43neg` full file; two gate failures close
the axis for good at zero slot cost. Prior: low (the axis's record); the readable-gain bar vs
0.96069 stays ±0.003.

**Gated 2026-10-13: both fail — `scorew` 0.993, `top1` 0.992 vs `fam_B15_nbfeat`**, inside the same
0.988–0.994 band as the five step-9 variants. Model B reaches the same test ranking whatever shape
the one-hop neighbour information takes (plain means, score-weighted means, the argmax-neighbour's
vector, max, delta, 2-hop, into-A). **The neighbour-feature axis is closed for good on the kind of
information as well**; no slot spent. The 2026-10-13 schedule stands: slot 1 = floorrev calibration,
slot 2 contingent on it.

## Round 3 (user-approved 2026-10-13): the label leak in random OOF scoring

**Mechanism.** Under random 5-fold OOF, the fold model that scores neighbour *j* was trained on row
*i* (label included) 80% of the time. Connected transactions have similar features, so *i*'s six
neighbour numbers carry a small dose of *i*'s own label; test scorers never saw a test label. The
board has already priced this direction: label information in training neighbour numbers hurts
monotonically (0 / 50 / 100% labels → 0.9562 / 0.9542 / 0.9462).

**Measured (14-step window, sub_06 model-A recipe, 3 seeds, read-only 2026-10-13):**

| OOF protocol | own-score AUC | `nb_in_mean` AUC | `nb_out_mean` AUC |
|---|---|---|---|
| random 5-fold (current) | 0.9981 | 0.9173 | 0.9020 |
| grouped by connected component of the labelled window subgraph | 0.9975 | 0.9133 | 0.8886 |

The random split inflates the neighbour signal by +0.004 (in) / +0.013 (out). The labelled window
subgraph has 3,177 components, the largest 1,136 rows (< one fold of ~2,340), so grouped folds are
feasible. Grouping keeps what made random OOF beat time-ordered OOF (+0.0023: the scorer sees the
row's own period) and removes only the neighbour leak.

**Candidates** (`two_stage(a_oof="group")`: `StratifiedGroupKFold`, groups = connected components):
1. `B15_nbfeat_goof` — one change on the 0.96069 reference (nb_feats=10). The reading file.
2. `B15_nbfeat_all_goof` — the same change on the Final 1 model (nb_feats=159); a finals candidate
   if (1) reads positive, and a more distinct Final 2 than `Aoof_random_bag10` (0.964) if it gates.

**Gate and reading.** Upload only if Spearman < 0.98 vs every scored file. Prior on readability:
uncertain — most model-B-side changes landed at 0.988–0.994, but this one changes how much model B
trusts the neighbour numbers rather than adding columns. Reading vs 0.96069: ≥ +0.003 signal, within
±0.003 tie (grouped OOF is the leak-free protocol and wins ties on principle), ≤ −0.003 rejected.
A gate failure closes the OOF-protocol axis at zero slot cost.

**Gated 2026-10-13: both fail.** CV improves on every fold, most at the far horizon, as the leak
hypothesis predicts (validation rows are scored like test rows, leak-free):

| file | CV 3-fold | hard 22–28 | horizon 22–35 | closest scored file (Spearman) |
|---|---|---|---|---|
| `fam_B15_nbfeat` (random OOF, parent) | 0.9689 | 0.9139 | 0.8215 | – |
| `fam_B15_nbfeat_goof` | 0.9739 | 0.9293 | 0.8443 | `fam_B15_nbfeat` 0.994 |
| `fam_B15_nbfeat_all` (random OOF, parent) | 0.9627 | 0.8963 | 0.8018 | – |
| `fam_B15_nbfeat_all_goof` | 0.9656 | 0.9056 | 0.8113 | `fam_B15_nbfeat_all_bag15` / `bag30` 0.995 |

The leak is real and removing it is the principled protocol, but it barely moves the test ranking
(0.994–0.995 like the parents): on the board it would return the parent's number plus noise. **No
slot spent; the OOF-protocol axis is closed** (time / random / 10-fold / fold-average / grouped).
Finals are unchanged. Open choice for the user: grouped OOF is the leak-free protocol and wins ties
on principle, so `final_model*.ipynb` could adopt it for the report — but the notebook must then
reproduce an *uploaded* file, which would cost a slot on a file the board cannot distinguish.

## Calibration read (2026-10-13): `fam_bag15_post43floorrev` = 0.83841

Δ vs the floor (0.89203) = −0.0536 = −w·(2·A22 − 1). Since A22 ≤ 1, w ≥ 0.0536; the public-row estimate
(27–49 post-43 positives) puts w ≈ 0.054–0.072, so **A22 ≈ 0.87–0.96**: Final 1 already orders the post-43
block well, and a perfect reordering is worth at most w·(1 − A22) ≈ 0.002–0.009. With both motivated donors
reading worse than Final 1's own order, **the post-43 ordering campaign closes with evidence** (exit condition met).

## Round 4 (2026-10-14): every CV-positive gate-failed change stacked into one model

User goal (2026-10-14): 7–8 more days of score improvement; near the top, small principled gains are worth
uploading even when unreadable. Each of these was CV-positive on the top-10 parent and individually unreadable
(0.988–0.994): grouped OOF, top-10 means into model A, max / delta / argmax-neighbour profile of the top-10,
step-mean model-A score. Stacked (`two_stage(..., nb_extra_k=10, nb_extra_stats=("max",), ...)`; the extras use the
top-10 only, the means keep all `nb_feats`):

| file | CV 3-fold | hard 22–28 | horizon 22–35 | closest scored file (Spearman) |
|---|---|---|---|---|
| `fam_B15_nbfeat` (parent, 0.96069) | 0.9689 | 0.9139 | 0.8215 | – |
| `fam_B15_nbfeat_goof` | 0.9739 | 0.9293 | 0.8443 | `fam_B15_nbfeat` 0.994 |
| **`fam_B15_nbfeat_combo`** | **0.9746** | **0.9311** | **0.8491** | `fam_B15_nbfeat` **0.982** |
| `fam_B15_nbfeat_all` (Final 1 model) | 0.9627 | 0.8963 | 0.8018 | – |
| `fam_B15_nbfeat_all_goof` (0.96209) | 0.9656 | 0.9056 | 0.8113 | 0.995 |
| **`fam_B15_nbfeat_all_combo`** | **0.9671** | **0.9097** | **0.8150** | `fam_B15_nbfeat_all_goof_bag15` 0.992 |

The stacked changes move the test ranking further than any single one did (0.982 vs 0.988–0.994): they add up.
Uploads: `fam_B15_nbfeat_combo` vs 0.96069 (the most distinct; ≥ +0.003 is a weak signal) and
`fam_B15_nbfeat_all_combo` vs the 5-seed parents (0.96177 / 0.96323 / goof 0.96209). Seeds 5–14 of the all-159
combo are building (`B15_nbfeat_all_combo_seedsBC`) for a 15-seed bag to compete with Final 1. Finals are chosen
by CV + principle with the board as a loss detector (< −0.003 rejects), never by the highest public score among ties.

**Read 2026-10-14: `fam_B15_nbfeat_all_combo` = 0.96362**, the highest public score (+0.0011 vs the 5-seed parents'
mean 0.9625, +0.0015 vs `all_goof` 0.96209) — inside seed noise as a single reading. **Seed check of the CV
(like for like, CV seeds 0–1 vs 5–6):**

| CV seeds | Final 1 model | + grouped OOF | + grouped OOF + extras (combo) |
|---|---|---|---|
| 0–1 | 0.9627 / 0.8963 / 0.8018 | 0.9656 / 0.9056 / 0.8113 | 0.9671 / 0.9097 / 0.8150 |
| 5–6 | 0.9651 / 0.9033 / 0.8130 | 0.9695 / 0.9168 / 0.8285 | 0.9643 / 0.9017 / 0.8052 |

CV seed noise is ≈ ±0.003 (3-fold) and ±0.01 (horizon). **Grouped OOF beats the parent on both seed pairs**
(+0.003/+0.004 3-fold, +0.010/+0.016 horizon): robust. **The extras do not**: +0.0015 on seeds 0–1, −0.0052 on
seeds 5–6 (horizon +0.004 / −0.023). The CV case for the extras was seed luck; the board's +0.0015 is one noisy
reading. Decision rule for Final 1: the 15-seed combo bag vs `fam_B15_nbfeat_all_goof_bag15` (0.96310) — combo bag
≥ 0.9645 (≈ +0.0015 on 15-seed bags, where seed noise is smaller) → combo; otherwise tie → grouped OOF alone (fewer
columns, CV-robust).

**Read 2026-10-14: `fam_B15_nbfeat_combo` = 0.95948, −0.0012 vs `fam_B15_nbfeat` (0.96069).** Two board readings of the
stacked extras: +0.0011–0.0015 (all-159) and −0.0012 (top-10) — they average ≈ 0, which matches the seed-checked
CV (extras +0.0015 / −0.0052). **The extras carry no gain; grouped OOF is the only robust part of round 4.** The
15-seed combo bag is still built and uploaded under the decision rule above (≥ 0.9645 → combo; otherwise goof bag15).

## Round 5 (2026-10-14): leave-one-step-out OOF (`a_oof="step"`)

Each window step is scored by a model on the other 13 window steps: like the test scorer it has never seen the
row's period and has ~the same data; no neighbour leak (edges never cross steps). It separates the two things
random OOF changed vs time-ordered OOF (same-period scorer; more data). `fam_B15_nbfeat_all_soof` (5 seeds):
CV seeds 0–1 **0.9685 / 0.9144 / 0.8261**, the best of the all-159 family (parent 0.9627 / 0.8963 / 0.8018, goof
0.9656 / 0.9056 / 0.8113, combo 0.9671 / 0.9097 / 0.8150); gate 0.994 vs the goof bag15. Seeds 5–14 building for the
seed check (the combo's seeds 0–1 lead vanished on seeds 5–6) and a 15-seed bag.

**Read 2026-10-14: `fam_B15_nbfeat_all_combo_bag15` = 0.96316** — an exact tie with the goof bag15 (0.96310) and the old
Final 1 (0.96313), below the pre-set 0.9645 bar. The 5-seed 0.96362 was seed noise. **Final 1 → `fam_B15_nbfeat_all_goof_bag15`**
(simpler; grouped OOF is the one CV-robust change). Extras closed.

**Seed check 2026-10-14: leave-one-step-out holds.** CV seeds 5–6 = 0.9685 / 0.9142 / 0.8251 (seeds 0–1 0.9685 / 0.9144 /
0.8261: the most seed-stable model of the family). Beats random OOF on both seed pairs (+0.003–0.006 3-fold, +0.012–0.024
horizon); ≈ grouped OOF (ahead on 0–1, slightly behind on 5–6). Both leak-free protocols are robust gains over random OOF.
`fam_B15_nbfeat_all_soof_bag15` built (seeds 0–14): 0.996 vs the goof bag15, 0.994 vs the old Final 1, 0.968 vs Final 2.
The protocol average (goof bag15 + soof bag15)/2 ranks 0.999 like the goof bag15 (not written; finals option, idea 4).

## Round 6 (2026-10-15): one model-B change each on Final 1 (`fam_B15_nbfeat_all_goof_bag15`, 0.96310)

Final 1 = the grouped-OOF bag15 (four OOF protocols tie on the board at 0.9629–0.9632; grouped OOF fits 5 scorers per
seed vs 14 for leave-one-step-out, so the hand-in notebook is faster). CV seeds 0–1, parent `all_goof` 0.9656 / 0.9056 / 0.8113:

| file | change | CV 3-fold / hard / horizon | Spearman vs Final 1 | verdict |
|---|---|---|---|---|
| `fam_B15_nbfeat_all_goof_col10` | model B colsample 1.0 | 0.9631 / 0.8976 / 0.8047 | 0.992 | worse CV, unreadable: dropped |
| `fam_B15_nbfeat_all_goof_dart` | model B DART (drop 0.1, 300 rounds) | 0.9567 / 0.8812 / 0.7944 | < 0.93 (nearest: plain two-stage 0.932) | clear CV loss; readable but resembles older weaker files: held |
| **`fam_B15_nbfeat_all_goof_blocks`** | model B = mean of fits on all / local+nb / aggregated+nb columns | **0.9657 / 0.9069 / 0.8166** | **0.920** | **passes the gate with neutral-to-positive CV: next upload** |

Reading rule for `blocks` vs Final 1's 5-seed sibling `all_goof` (0.96209) and the 5-seed family mean (≈ 0.9628): it is
readable (0.920), so ≥ +0.003 is a signal → build its 15-seed bag as the Final 1 candidate; ≤ −0.003 → closed; between →
tie (Final 1 stays). Its distance from Final 1 also makes a blended or alternative Final 2 possible if it reads well.

**Read 2026-10-15: `fam_B15_nbfeat_all_goof_blocks` = 0.96382**, the highest public score: +0.0017 vs its parent
`all_goof` (0.96209), +0.0010 vs the five 5-seed files of the Final 1 model (mean ≈ 0.9628). Tie band by the
pre-set ±0.003 rule, but readable: it ranks 0.920 like Final 1 and scores at least as well — two equally good,
differently wrong models, the precondition for a blend gain that every earlier 0.99-alike blend lacked. Next:
1. `B15_nbfeat_all_goof_blocks_seedsBC` (seeds 5–14, building) → `fam_B15_nbfeat_all_goof_blocks_bag15`, upload vs Final 1 (0.96310).
2. `(fam_B15_nbfeat_all_goof_bag15 + blocks_bag15) / 2`, upload.
3. Finals: blend highest → Final 1 = blend, Final 2 = the less blend-alike of blocks_bag15 / goof_bag15; else Final 1 =
   blocks_bag15 if ≥ 0.963, Final 2 = goof_bag15. `fam_B15_Aoof_random_bag10` leaves the finals either way.
Blend preview (with the 5-seed blocks file): the probability average ranks 0.995 like blocks and 0.951 like Final 1
(blocks' probabilities are more spread out and dominate); the **rank average** (pooled ranks, so the cross-period
order is kept) ranks 0.980 like each parent. The blend is built as a rank average.

**Built 2026-10-15:** seeds 5–14 of the block model; CV seeds 5–6 = 0.9656 / 0.9066 / 0.8140 vs `all_goof` seeds 5–6
0.9695 / 0.9168 / 0.8285 (−0.004 / −0.010 / −0.015; seeds 0–1 were +0.0001 / +0.0013 / +0.0053). The block model alone is
level-to-slightly-worse on CV; its public +0.001–0.0017 is noise-sized. The blend case rests on distinctness, not on it.
- `fam_B15_nbfeat_all_goof_blocks_bag15` (15 seeds): 0.921 vs Final 1, 0.999 vs the 5-seed blocks file (0.96382).
- `fam_blend_goof_blocks_bag15` = rank average of Final 1 and blocks_bag15: 0.980 vs each parent (passes the gate).
Both uploaded next. Reading: blend ≥ max(parents) + 0.002 → Final 1 = blend; blocks_bag15 < 0.9631 − 0.003 → the block
model is closed and the blend's gain (if any) is the diversity effect alone.

**Read 2026-10-16: `fam_blend_goof_blocks_bag15` = 0.96539**, +0.0023 vs Final 1 (0.96310), +0.0016 vs the 5-seed blocks
file (0.96382): the highest public score, and the first blend of two equally good models that rank differently (0.921).
Every earlier blend mixed 0.99-alike files. Learner diversity measured on old files: XGBoost model B ranked 0.944 like its
LightGBM parent at −0.002 public (the same "different but nearly as good" profile as blocks); CatBoost 0.888 at −0.010,
ExtraTrees 0.710 at −0.038, DART (current base) 0.872 with a clear CV loss.

**Read 2026-10-16: `fam_B15_nbfeat_all_goof_blocks_bag15` = 0.96385** (+0.0008 vs Final 1: tie). The blend (0.96539) is
+0.0015 above the better parent and +0.0023 above Final 1: above both, just under the pre-set +0.002 promotion bar.

## Round 7 (user-approved 2026-10-16): XGBoost as model B, a third blend member

Finals recommended 2026-10-16: Final 1 = `fam_blend_goof_blocks_bag15` (0.96539), Final 2 = `fam_B15_nbfeat_all_goof_bag15`
(0.96310). Built: `B15_nbfeat_all_goof_xgb` (5 seeds) and `_seedsBC` (seeds 5–14): XGBoost depth 4, 300 rounds, lr 0.05,
60% columns, on Final 1's 483 inputs (`two_stage(b_learner="xgb")`, unchanged branch). Expected blend gain 0 to +0.0015
(less distinct than blocks at 0.944 on the old base, −0.002 alone; diminishing returns as a third member).
**Pre-set inclusion rule:** add to the blend only if it ranks < 0.95 like both Final 1 and blocks_bag15 AND its CV on both
seed pairs is within 0.003 of `all_goof` (seeds 0–1 0.9656 / 0.9056 / 0.8113; seeds 5–6 0.9695 / 0.9168 / 0.8285).
Then upload the three-way equal-weight rank average vs 0.96539: ≥ +0.002 → Final 1; otherwise the two-way blend stays.

**Round 7 result (2026-10-16):** XGBoost model B ranks 0.944 like Final 1 and 0.910 like blocks_bag15 (distinctness half
passes), but CV is behind `all_goof` on both seed pairs: seeds 0–1 0.9648 / 0.9026 / 0.8062 (−0.001 / −0.003 / −0.005),
seeds 5–6 0.9646 / 0.9019 / 0.8050 (−0.005 / −0.015 / −0.024). **Fails the pre-set accuracy half: not added to the blend.**
A three-way blend would also rank 0.994 like the two-way blend (unreadable). `fam_B15_nbfeat_all_goof_xgb_bag15` built and
held, not uploaded. Caveat for the report: the block model also trailed on seeds 5–6 yet helped the blend, so a member's own
CV is a weak predictor of its blend value — the CV check of the blend itself is the right test (next, pending approval).

## Round 8 (user-approved 2026-10-16): CV of the blends themselves (`src/blend_cv.py`)

Members goof / blocks / xgb / soof run once per fold (3 × 7-step + horizon) and seed (0, 1, 5, 6); predictions cached in
`artifacts/blend_cv_preds.pkl`; blends = equal-weight pooled-rank averages, as submitted. Wiring check: the single-member
rows reproduce the family CVs exactly. Mean of both seed pairs (3-fold / hard / horizon), full table `artifacts/blend_cv.csv`:

| blend | CV |
|---|---|
| soof | **0.9685 / 0.9143 / 0.8256** (best on both seed pairs) |
| goof + soof | 0.9681 / 0.9130 / 0.8224 |
| blocks + soof | 0.9679 / 0.9115 / 0.8185 |
| goof | 0.9675 / 0.9112 / 0.8199 |
| goof + blocks (**Final 1, 0.96539 public**) | 0.9674 / 0.9097 / 0.8155 |
| blocks | 0.9657 / 0.9068 / 0.8153 |
| xgb | 0.9647 / 0.9022 / 0.8056 |

Readings: (1) **CV does not confirm the blend's board gain** (+0.0023 public; level-to-below goof alone on CV) — either the
board gain is partly noise (it is +0.0015 over blocks_bag15, near the floor) or the historical folds cannot see it, as
they missed the window choice; neither instrument can settle it. (2) xgb lowers every blend: exclusion confirmed. (3) soof
is the most consistent CV winner (best alone; improves every blend), board tie (0.96290). **Finals revised: Final 1 =
`fam_blend_goof_blocks_bag15` (board-best), Final 2 = `fam_B15_nbfeat_all_soof_bag15` (CV-best)** — the two instruments
disagree, so one final from each covers the private board best. Slot 3 of 2026-10-16: xgb alone (5 seeds, readable at
0.944) as a test of whether CV judges distinct members correctly: ≥ ~0.962 → CV undervalues them, the diversity direction
stays open on the board; ≤ ~0.959 → CV is right, the blend's board gain is probably noise, and the remaining days go to the
notebooks and the report.

**Read 2026-10-16: `fam_B15_nbfeat_all_goof_xgb` = 0.96189** — −0.0010 vs the 5-seed family mean (≈ 0.9629), +0.0001 vs
`fam_B15_nbfeat_all` (0.96177): a tie inside seed noise, while CV had it −0.003 / −0.009 / −0.014. Between the pre-set
thresholds (0.959 / 0.962), at the upper one: CV overstates the deficit of distinct members, as it did for blocks. The
diversity direction stays open but cannot be read on the public board through blends (any three-way blend ranks 0.99+ like
the current one); members can only be read alone, as here.

**2026-10-16, user-approved:** built `fam_blend3_goof_blocks_xgb_bag15` (equal-weight pooled-rank average of the goof,
blocks and xgb 15-seed bags; 0.994 vs the 2-way blend: unreadable, uploaded to be selectable; the finals choice between the
2-way and 3-way blend is a judgement between the board (xgb ties alone) and CV (xgb lowers every blend), made at the end).
`final_model*.ipynb` now carry `CONFIG["members"]` (model-B variants on shared model A and inputs: "lgbm", "blocks", "xgb")
and a pooled-rank `blend`; default `("lgbm", "blocks")` = Final 1. The validation section reports every member and the blend;
the reproducibility check compares each member bag and the blend with its uploaded file.

## Round 9 (user-approved 2026-10-17): two new blend members, each read alone first

Only a member uploaded alone is readable (any blend of ≥ 3 ranks 0.99+ like the current one). Built:
- `B15_nbfeat_all_goof_localA` (`two_stage(a_cols="local")`): model A on the local block only, so the six neighbour
  numbers come from a different scorer; model B unchanged (all columns).
- `B15_nbfeat_all_goof_rsub` (`two_stage(b_rsub=5)`): model B = mean of 5 fits, each on a random half of the 159 raw
  columns + all neighbour columns (subsets drawn from the seed): the block idea with random groups.
**Pre-set rule:** a member joins the final blend if it ranks < 0.95 like each existing member (goof / blocks / xgb bag15s)
and scores ≥ 0.962 alone; a joiner gets its 15-seed bag and a blend file the next day. Slots of 2026-10-17: 1 = the
3-way blend (selectable), 2 = localA alone, 3 = rsub alone (if it is distinct; otherwise the slot is kept).

**Round 9 result (2026-10-17): neither member qualifies.**

| member | CV seeds 0–1 (vs goof 0.9656 / 0.9056 / 0.8113) | vs goof bag15 | vs blocks bag15 | vs xgb bag15 |
|---|---|---|---|---|
| `localA` | 0.9684 / 0.9146 / 0.8278 (+0.003 / +0.009 / +0.017) | 0.991 | 0.914 | 0.940 |
| `rsub` | 0.9663 / 0.9063 / 0.8063 (+0.001 / +0.001 / −0.005) | 0.977 | 0.922 | 0.937 |

Both are distinct from blocks and xgb but too close to goof (rule: < 0.95 vs every member). Model A's input set again barely
moves the final ranking (0.991), as every model-A change before; random column halves land between goof and blocks.
`localA` has the strongest seeds 0–1 CV of any single model besides soof — a candidate replacement for goof inside the
blend only if a seed 5–6 check holds (board-unreadable either way). No upload for either on 2026-10-17.

**Read 2026-10-17: `fam_blend3_goof_blocks_xgb_bag15` = 0.96532**, −0.00007 vs the 2-way blend: an exact tie, as predicted.
Finals stand: Final 1 = `fam_blend_goof_blocks_bag15` (0.96539; simpler of the tied pair), Final 2 =
`fam_B15_nbfeat_all_soof_bag15` (CV-best; the 3-way blend is 0.994 like Final 1 and would hedge nothing).

## Round 10 (user-approved 2026-10-18): drift-aware feature removal on the goof model

Both lists live in `families.py` (`DRIFT_ADV`, `DRIFT_UNSTABLE`); the columns leave the raw block and, through
`Xtr.columns`, the neighbour-mean block. `feat_2` example: window 90th percentile exceeded by 10% of window rows but
28% / 32% of test rows (steps 36–42 / 43–49). Unstable list = pooled window AUC ≥ 0.05 from 0.5 with the per-step
direction opposite in ≥ 3 of 14 window steps (selected on window labels: CV optimistic).

| file | cols | CV seeds 0–1 (goof 0.9656 / 0.9056 / 0.8113) | vs goof bag15 | vs blocks bag15 | vs xgb bag15 | vs blend |
|---|---|---|---|---|---|---|
| `dropadv` (5 adversarial > 0.85) | 154 | **0.9714 / 0.9204 / 0.8434** (+0.006 / +0.015 / +0.032) | 0.970 | 0.920 | 0.936 | 0.964 |
| `dropunstable` (23 direction-unstable) | 136 | 0.9678 / 0.9124 / 0.8182 (+0.002 / +0.007 / +0.007) | 0.976 | 0.902 | 0.918 | 0.958 |

`dropadv` is the largest CV gain since the two-stage, growing with the horizon (the drift signature), and readable.
Reading rule vs the goof 5-seed family (≈ 0.9625–0.9629): ≥ +0.003 → seeds 5–14 for a 15-seed bag and a blend with
goof replaced; tie → blend member candidate (0.92–0.94 like blocks/xgb); ≤ −0.003 → `feat_2`'s signal outweighs its drift.

**Read 2026-10-18: `dropadv` = 0.95957** — −0.0025 vs goof 5-seed (0.96209), −0.003 vs the 5-seed family: tie-to-loss,
against the largest CV gain since the two-stage. Per-period levels: post-43 mean score +38% vs +9% pre-43 (pre/post ratio
11.1 → 8.8), within-period orders 0.975 / 0.961 like goof. **Correction (same day):** that probability-mean ratio is a
low-end calibration artefact; in ranks (what AUC sees) dropadv puts *fewer* post-43 rows in the top 5% (0.096 vs goof
0.113), like levelamp25 (0.093, −0.0015): not a flattening. Cause not identified. Drift handling closed on removal as on normalisation. `dropunstable` held (weaker,
label-selected rationale; its drift columns overlap only via feat_112/113).

## Round 11 (user-approved 2026-10-18): model-B hyperparameters on goof, as blend-member candidates

| file | CV seeds 0–1 (goof 0.9656 / 0.9056 / 0.8113) | vs goof15 | vs blocks15 | vs xgb15 | vs blend | post-43 share of top 5% (goof 0.113) |
|---|---|---|---|---|---|---|
| `xt` (`extra_trees=True`) | 0.9652 / 0.9048 / 0.8124 (tie) | 0.955 | 0.913 | 0.942 | 0.953 | 0.111 |
| `reg` (min 100 rows/leaf, 30% cols, λ 5) | 0.9638 / 0.8993 / 0.8007 (−0.002 / −0.006 / −0.011) | 0.978 | 0.903 | 0.949 | 0.959 | 0.115 |

`xt` ties goof on CV with a different ranking (the most distinct same-strength member after blocks); it misses the round-9
join rule only on goof (0.955 vs < 0.95). `reg` is CV-lower on every fold and 0.978 like goof: held. Upload `xt` alone vs
goof 5-seed (0.96209): ≥ ~0.962 → seeds 5–14 and a blend file (xt replacing or joining goof); ≤ −0.003 → closed.

**Read 2026-10-18: `xt` = 0.96678** — +0.0047 vs goof 5-seed (0.96209), 3× seed noise, +0.0014 vs the Final 1 blend: the
highest public score, from a single 5-seed model whose CV tied its parent (CV blind again, this time in our favour).
Mechanism reading: random split thresholds stop model B fitting exact cut-points of the training period; the board
rewards robustness that the historical folds cannot see.

## Round 12 (2026-10-18): build on extra_trees

Queued (`artifacts/logs/round12_xt.log`): `blocks_xt` (the blend partner on the xt recipe), `xt_seedsBC` (seeds 5–14 →
`xt_bag15`, the Final 1 candidate), `xt_r600` (600 rounds). Next, one change each after these read: the blend
`xt_bag15 + blocks_xt_bag15`; coarser thresholds (`max_bin` 63, a deterministic cousin of extra_trees); extra_trees in
model A as well; model-B leaves 31 under extra_trees (random splits may shift the depth optimum).
Finals rule: once `xt_bag15` is built and uploaded it replaces goof as Final 1's base; both notebooks get the
`extra_trees` switch and are re-executed before any finals change.

**Round 12 built (2026-10-18):**

| file | CV (xt seeds 0–1 0.9652 / 0.9048 / 0.8124) | vs xt | vs goof15 | vs blocks15 | post-43 top-5% share |
|---|---|---|---|---|---|
| `xt_seedsBC` (seeds 5–6 CV) | 0.9661 / 0.9073 / 0.8115 (seed-consistent) | 0.995 | – | – | – |
| `xt_bag15` (5 × seeds 0–4 + 10 × seeds 5–14) / 15 | – | 0.998 | 0.959 | 0.918 | – |
| `xt_r600` | 0.9668 / 0.9093 / 0.8134 | 0.974 | 0.937 | 0.881 | 0.106 |
| `blocks_xt` | 0.9628 / 0.8985 / 0.8029 (below blocks 0.9657 / 0.9069 / 0.8166) | 0.945 | 0.912 | 0.970 | 0.132 |

Blends (pooled ranks): xt15 + blocks15 0.976 vs xt / 0.989 vs the blend; xt15 + blocks_xt 0.984 vs xt — finals-only
(unreadable). Queued: `xt_bin63` (max_bin 63), `xt_L31` (31 leaves). Upload order: `xt_r600` (readable, moves further
along the axis that paid: 0.937 vs goof15 where xt is 0.955), then `blocks_xt` (readable 0.945 vs xt), `xt_bag15`
(selectable; Final 1 base once the notebooks reproduce it).

**Read 2026-10-18: `xt_r600` = 0.96363** (−0.0032 vs xt: rejected; 300 rounds stays). `xt_bin63`: 0.993 like xt, CV
0.9674 / 0.9106 / 0.8096 — unreadable, dropped. Caution: r600 ranks 0.974 like xt yet reads 0.003 lower, and is only
+0.0015 above goof 5-seed, so part of xt's +0.0047 may be seed luck. **Next upload = `xt_seedsBC` (seeds 5–14 only, no
seed shared with xt) as a replicate**, as `fam_B15_nbfeat_all_seedsB` was: ≥ ~0.965 → xt is real, `xt_bag15` becomes
Final 1's base and the notebooks get extra_trees; ≤ ~0.963 → xt was a tie with a lucky draw, finals stay.
`xt_L31`: CV 0.9644 / 0.9020 / 0.8024 (below xt on every fold), 0.981 like xt — fails the gate, dropped. The extra_trees
tuning neighbourhood (rounds, bins, leaves) is mapped: only the base xt setting (15 leaves, 300 rounds) is a candidate.

## Round 14 target (set by the user 2026-10-18): beat first place, public 0.97033

| reference | public | gap to 0.97033 |
|---|---|---|
| `xt` (5 seeds, one reading) | 0.96678 | +0.0036 |
| Final 1 blend (goof + blocks) | 0.96539 | +0.0049 |
| `xt_r600` (0.974 like xt) | 0.96363 | +0.0067 |

**Size of the gap in our own units.** Every real gain so far: two-stage +0.004, random OOF +0.0023, neighbour feature
means +0.0022, blend +0.0023, `xt` +0.0047 (unconfirmed). The gap is one `xt`-sized gain, or two typical ones. Seed noise
is ≈ 0.0015 per 5-seed file, so a single reading of +0.0036 is never conclusive: the target counts as reached when a
seed-bagged file (≥ 10 seeds) reads > 0.97033.

**Feasibility (judgement, not measurement).** If the replicate confirms `xt` near 0.966, one more gain of the blend's
size (+0.002–0.003) plus the bag lands at ≈ 0.968–0.969, still short; beating 0.97033 needs about two more real gains in
about seven days, a rate we have matched only once (2026-10-05 → 10-08). If the replicate puts `xt` near 0.964, the gap is
≈ 0.006, about three gains: unlikely. Estimate: roughly one chance in four to one in three if `xt` holds, well under one
in five if it does not. The leader's 0.97033 is itself one public reading on 30% of the test set and may contain luck;
the private board (70%) decides, so the route must not rely on tuning to the public 30%.

**Route, in order of expected gain:**
1. `xt_seedsBC` replicate (slot 1, 2026-10-19): fixes the base level.
2. The ensemble on the xt recipe: `blocks_xt` alone, then `xt_bag15 + blocks_xt_bag15` rank blend — the one structure that
   has gained on the board (+0.0023 from two equally strong, different members).
3. More randomised members of equal strength (the diversity the board pays for): `xt` with a different `colsample_bytree`
   (0.4 / 0.8), and `extra_trees` with random-subspace model B (`b_rsub`), each read alone, joined only if ≥ 0.962 alone and
   < 0.95 like each member.
4. If `xt` holds: notebooks get the `extra_trees` switch; finals = the best bagged blend + the strongest different file.

**Read 2026-10-19: `xt_seedsBC` = 0.96736** (seeds 5–14, no seed shared with xt): **xt confirmed.** Two independent seed
sets read 0.96678 / 0.96736 (mean 0.9671, +0.005 vs the goof 5-seed family). Rank 6; gap to first place 0.97033 = +0.0030.
Slots 2–3 of 2026-10-19: `xt_bag15` (selectable Final 1 base) and `blocks_xt` (blend-member reading: joins if ≥ 0.962).
**Read 2026-10-19: `blocks_xt` = 0.96630** (+0.0025 vs blocks 5-seed; 0.945 like xt): qualifies as a blend member. Blend
estimate (members' mean + the earlier blend's ≈ +0.002) ≈ 0.968–0.969. Built at once from existing CSVs: the pooled-rank
blend of `xt_bag15` and `blocks_xt` (5 seeds) as an early look; the like-for-like version needs `blocks_xt` seeds 5–14.

**Round 14 members built (2026-10-19): neither qualifies.**

| file | CV (xt 0.9652 / 0.9048 / 0.8124) | vs xt15 | vs blocks_xt | vs blocks15 | post-43 top-5% |
|---|---|---|---|---|---|
| `xt_rsub` (5 random column halves, extra_trees) | 0.9639 / 0.9002 / 0.8002 | 0.985 | 0.951 | 0.919 | 0.108 |
| `xt_col04` (40% columns per tree) | 0.9639 / 0.9002 / 0.8005 | 0.994 | 0.946 | 0.914 | 0.112 |

Both fail the gate against xt (0.984 / 0.993) and the member rule (< 0.95 like each member). Under extra_trees, extra
column randomisation lands back on xt's ranking: the randomisation axis is saturated at xt itself. Without extra_trees
`rsub` was 0.977 like goof; with it, 0.985 like xt. Column-sampling settings closed (1.0, 0.8 not built, 0.6 = xt, 0.4).
The blend partner for xt stays `blocks_xt` (different column *blocks*, not random columns); its seeds 5–14 are building.
**Built 2026-10-19:** `blocks_xt_seedsBC` (seeds 5–6 CV 0.9640 / 0.9018 / 0.8040, seed-consistent), `blocks_xt_bag15`
(5 × seeds 0–4 + 10 × seeds 5–14) / 15, and the like-for-like blend
`fam_blend_B15_nbfeat_all_goof_xt_bag15_B15_nbfeat_all_goof_blocks_xt_bag15` (pooled ranks; 0.987 like xt_seedsBC,
0.986 like blocks_xt): the Final 1 candidate, unreadable vs its members by construction.

## Round 15 (2026-10-19): after the xt + blocks_xt blend (0.96828)

**Read:** `fam_blend_..._xt_bag15_..._blocks_xt_bag15` = **0.96828**, the highest public score: +0.0016 over its members'
mean (0.9671 / 0.9663), the same blend effect as before (+0.0019). Two-member diversity pays twice now; a third member has
not (goof + blocks + xgb tied the 2-way blend, its xgb member 0.005 weaker than the others). Gap to first: +0.0021.

**1. Make Final 1 reproducible (no slot; required by CLAUDE.md rule 8).** `final_model.ipynb`, `final_model_kaggle.ipynb`,
`submission_notebook.ipynb`: model B gets `extra_trees` per member; `CONFIG["members"]` default = (xt, blocks_xt);
re-execute and check the blend and each member bag against the uploaded files.

**2. Finals.** Final 1 = this blend (selected on Kaggle). Final 2 should be strong *and* different, as a hedge against the
`extra_trees` gain being a public-30% effect. Candidates already scored: the old blend goof + blocks (0.96539, no
extra_trees, 0.971 like Final 1) or soof_bag15 (0.96290, 0.950). Proposed: **the old blend** (stronger; still the most
different strong file).

**3. Third member (gate first, slot only if it qualifies).** Stricter rule than round 9, because a weaker third member tied:
≥ 0.9655 alone (within ≈ 0.0015 of the members) and < 0.95 like both members. Candidates in order:
- `soof_xt`: leave-one-step-out model A + extra_trees model B. soof15 is the CV-best model, tied goof on the board and
  ranks 0.959 / 0.916 like the two members: similar strength, partly different ranking.
- `nbonly_xt`: model B on the neighbour columns only (6 scores + 318 means), extra_trees: the one column view the blocks
  member does not have; strength unknown (local-only lost −0.012 alone before).
- `xtA`: extra_trees in model A as well (the neighbour scores both members see change); gate check only.
Not pursued: XGBoost member (0.945 / 0.915 distinct but ≈ 0.005 weaker; its 3-way blend tied), more column randomisation
(saturated, round 14), blend weights other than equal (unreadable, fits the public 30%).

**4. Report.** Drift-evidence cell in `submission_notebook.ipynb` §1.3 (adversarial AUCs of kept features, `feat_2` shift);
§2–3 describe extra_trees (hypothesis: robustness to cut-points under drift) and the two-member blend.

**Feasibility of 0.97033:** needs ≈ +0.002, about one more blend-sized gain. Plausible only through a third member as strong
as the first two; the earlier 3-way blend says the odds are modest (judgement: about one in four).

**Round 15 third-member search (2026-10-19): none qualifies; search closed.**

| file | CV (xt 0.9652 / 0.9048 / 0.8124) | vs xt15 | vs blocks_xt15 | vs Final 1 | post-43 top-5% |
|---|---|---|---|---|---|
| `soof_xt` | 0.9634 / 0.8992 / 0.8024 | 0.996 | 0.950 | 0.985 | 0.115 |
| `nbonly_xt` | 0.8737 / 0.7996 / 0.7062 (fails sanity) | 0.392 | 0.490 | 0.445 | 0.307 |
| `xtA` | 0.9671 / 0.9104 / 0.8208 | 0.993 | 0.943 | 0.981 | 0.111 |

Under extra_trees, the model-A side (OOF protocol, extra_trees in A) converges onto xt's ranking: soof ranked 0.959 like
xt15 without extra_trees in B and 0.996 with it. The one view that differs strongly, neighbour columns only, is far weaker
(CV −0.09) and loses the period level (post-43 share of the top 5% 0.307 vs 0.11): the row's own features are essential.
Only column-*block* structure (blocks_xt) gives a different ranking at equal strength. `xtA` has the best CV of the xt
family (+0.002 / +0.006 / +0.008) but is unreadable (0.993); held as a possible like-for-like swap for xt inside Final 1,
which the board could not read either. Finals stand.

**Round 16 (2026-10-19/20).** `inout_xt` (blocks by money direction): CV 0.9699 / 0.9181 / 0.8290, 0.987 like xt — not
readable, not a member; held. **`xt_w21` read 0.96755** (CV +0.008 / +0.023 / +0.033, 0.982 like xt): +0.0004 vs the xt mean,
a tie; 14 steps stays. Under exact thresholds 21 steps cost −0.004; under extra_trees it costs nothing and gains nothing:
consistent with the cut-point hypothesis, but no route to 0.9700. Both routes of round 16 closed.

**Round 17 built (2026-10-20): longer windows under extra_trees as a third member.**

| file | CV (identical for every window ≥ 21: the hard and horizon folds train on steps ≤ 21) | vs xt15 | vs blocks_xt15 | vs Final 1 | post-43 top-5% |
|---|---|---|---|---|---|
| `xt_w28` | 0.9728 / 0.9278 / 0.8454 | 0.957 | 0.923 | 0.952 | 0.104 |
| `xt_all` (35 steps) | 0.9728 / 0.9278 / 0.8454 | **0.927** | **0.887** | 0.918 | 0.102 |

Both pass the upload gate. `xt_all` meets the member rule's distinctness half (< 0.95 like both members): the first strong
candidate that does. Upload `xt_all` alone: ≥ 0.9655 → joins (seeds 5–14 → bag15 → 3-member blend); < 0.9655 → not a
member. `xt_w28` (0.957 like xt, just outside the member rule) maps the window curve under extra_trees (14: 0.9671,
21: 0.9676, 28: ?, 35: ?) and is the fallback if `xt_all` is weak.
**Read 2026-10-20: `xt_all` = 0.96486** (−0.0022 vs the xt mean): below the member bar. Window curve under extra_trees:
14 → 0.9671, 21 → 0.9676, 35 → 0.9649 (without extra_trees: 0.9484 / 0.9462 / 0.9437 on the single-stage). The
third-member search is closed on columns, model A, randomisation and training data. `xt_w28` not uploaded (no decision
value: it would sit between 21 and 35 and ranks 0.957 like xt).

**Round 18 built (2026-10-20): `xt_w21_seedsBC`** (seeds 5–14; CV seeds 5–6 0.9726 / 0.9267 / 0.8386 vs seeds 0–1
0.9728 / 0.9278 / 0.8454: seed-consistent) and `xt_w21_bag15` (5 × seeds 0–4 + 10 × seeds 5–14) / 15, 0.984 like xt_bag15.
Upload `xt_w21_seedsBC` as the replicate of 0.96755 (0.996 like it: unreadable by design, a noise reading): ≥ ~0.968 → two
seed sets favour 21 steps, build `blocks_xt_w21` (15 seeds) and the 21-step Final 1 variant; ≤ ~0.967 → 14 = 21, Final 1 stands.
**Read 2026-10-20: `xt_w21_seedsBC` = 0.96676.** 21 steps: 0.96755 / 0.96676 (mean 0.9672); 14 steps: 0.96678 / 0.96736
(mean 0.9671). Equal. The window is closed under extra_trees; Final 1 (0.96828) stands. With the third-member and window
searches closed, no readable route to 0.9700 remains in the two-stage + blend architecture.
