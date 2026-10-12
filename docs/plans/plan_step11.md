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
