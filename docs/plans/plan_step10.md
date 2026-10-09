# Step 10 — Plan: variance, finals and the report

Date: 2026-10-09. Readings of the morning: **`fam_B15_nbfeat_all_bag10` = 0.96263** (the 10-seed bag,
between its parents 0.96177 and 0.96323: variance reduction behaves as expected; Final 1 is now
selectable) and **`fam_B15_nbfeat_gnn` = 0.95570** (−0.005 vs the 0.96069 reference: out-of-fold
GraphSAGE embeddings as model-B columns hurt; the trees-plus-embeddings recipe is closed on the
board as the in-sample version was on CV). One slot left today.

## Honest position

Every axis of the pipeline has now been read to its end on the public board, and the last reading of
a genuinely new information source was a clear loss:

| axis | last reading | state |
|---|---|---|
| neighbour scores (6 numbers) | +0.004; more statistics, 2-hop, labels, second pass: unreadable or worse | closed |
| neighbour features (means) | +0.002 (top-10); all 159, max, delta, 2-hop, into model A: within noise or unreadable | closed |
| GNN embeddings into model B | −0.005 | closed |
| window, weighting, depth, learner, scorer, stacking, pseudo-labels, score corrections | all read 2026-09-29 → 2026-10-08 | closed |
| seed noise of the final model | 0.0015 between two 5-seed bags | the remaining lever |

Two facts set what is worth doing now. The grade is the private 70%, where sampling noise is about
two thirds of the public board's. And the model's own seed variance (0.0015 on a 5-seed bag) is
comparable to the public differences we have been reading. Reducing that variance helps on the
private board with certainty, in expectation by a few tenths of a point; no probe left on the list
offers more than that with any confidence.

## Step 10 actions

### 1. Final 1 = a 15-seed bag of the all-159 model (today's third slot)
`B15_nbfeat_all_seedsC` (seeds 10–14) is building; `fam_B15_nbfeat_all_bag15.csv` = the probability
average of the three 5-seed bags. Upload it today so it is selectable. Reading: a sanity check only
(expected 0.9622–0.9630); the point is variance on the private board.

### 2. Final 2 = a 10-seed bag of the previous architecture (first slot of 2026-10-10)
`B15_Aoof_random_seedsB` (seeds 5–9) is queued behind it; `fam_B15_Aoof_random_bag10.csv` = the
average with the scored seeds 0–4 file (0.95849). It ranks about 0.96 like Final 1, which is the hedge
the policy asks for, with half the seed variance. Upload it so it is selectable; expected 0.958–0.960.

### 3. Select the finals on Kaggle
`fam_B15_nbfeat_all_bag15` + `fam_B15_Aoof_random_bag10`. Revisit only if a later reading beats
0.96263 by more than 0.003 (the noise-aware threshold) with a file that ranks < 0.98 like it.

### 4. Notebook standard
`final_model*.ipynb` reproduce the 10-seed bag exactly (4.4e-16). When the 15-seed bag is the finals
file, set `seeds=tuple(range(15))` and re-execute (about 2 h); the column-order list in `CONFIG` stays.

### 5. The report (deadline 2026-10-25)
From here the effort goes to the write-up: `docs/report_notes.md` plus the Claude Doc hold every
table and figure. The through-line is the one in CLAUDE.md: let the model lean on what is recent and
on what its neighbours look like, through predictions rather than engineered features; historical CV
is a guide for features, not for temporal choices; the board is noisy and most changes are
unreadable; the one principled lever at the end is variance.

### 6. What would re-open the search
One slot a day stays available for an idea that changes the information set and passes the gate.
Nothing on the current list qualifies: bridging reaches too few rows, test-row self-training
reinforces the post-43 blind spot, unlabelled-neighbour feature means break parity, and every
within-pipeline variant is closed. A qualifying idea would have to bring information from outside
the 159 features and the labelled-only graph; none is known.

## Reading rules (revised 2026-10-08)
Versus the file a change is made on: ≥ +0.003 gain, ≤ −0.003 loss, between is a tie (the simpler
model wins). The ±0.002 band under-stated the final model's seed noise (0.0015).
