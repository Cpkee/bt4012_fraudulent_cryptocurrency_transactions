# Step 9 — Plan: the neighbour-feature axis, second round

Date: 2026-10-08. Reference for one-change probes: **`fam_B15_nbfeat.csv` = 0.96069** (two-stage +
means of the top-10 raw features over labelled in/out neighbours as 20 model-B columns). Highest
public score: `fam_B15_nbfeat_all.csv` = 0.96177 (means of all 159 features), +0.0011 = tie band, so
the simpler file stays the reference and the all-159 file leads the finals. 6th on the board.

## What 2026-10-08 established

| file | change on the reference | Spearman vs ref | public | reading |
|---|---|---|---|---|
| B15_nbfeat | top-10 feature means (20 columns) | 0.973 vs the old best | **0.96069** | +0.0022, gain |
| B15_nbfeat20 | top-20 | 0.989 | – | unreadable |
| B15_nbfeat_max | mean + max of the top-10 | 0.994 | – | unreadable |
| B15_nbfeat_delta | + row value minus each mean | 0.992 | – | unreadable |
| B15_nbfeat_all | all 159 feature means (318 columns) | 0.979 | 0.96177 | +0.0011, tie |

So: the neighbours' *features* carry information the six score numbers do not (the step-1 CV
rejection was for a single-stage model; inside the two-stage the board says +0.002). The *amount* of
feature information is closed: the top ten carry nearly all of it, and extra statistics or
differences of the same columns change nothing. What has not changed yet is **where** the
neighbour features enter.

## Candidates (one change each on `fam_B15_nbfeat`)

### 1. Neighbour feature means into model A as well — `B15_nbfeat_A` (building)
Model A has never seen any graph information: it scores every transaction from its own 159
features, and those scores become the six neighbour numbers. The feature means need no model, so
they can be computed first and given to model A too (train: labelled window rows as the neighbour
set, as for model B; test: the test file). Model B is unchanged (159 + 6 + 20).
- Why it might work: better scores in → better neighbour numbers out; the six numbers are 30% of
  model B's gain and have only ever been produced by a graph-blind scorer.
- Why it might not: the window has 11,705 rows; a 31-leaf model A with 20 more columns may fit
  the window's neighbourhoods rather than the test period's. CV is the sanity check.
- Readability: every score changes → every neighbour number changes; expected 0.94–0.98.
- Reading rule vs 0.96069: ≥ +0.002 gain, ≤ −0.002 loss. Third slot of 2026-10-08 if it passes the
  gate; otherwise first slot of 2026-10-09.

### 2. Only if 1 reads a gain: combine with the all-159 means — `B15_nbfeat_all_A`
The two readable gains of the axis stacked (all 159 means to both models). One upload.

### 3. Only if 1 is unreadable or loses: two-hop feature means — `B15_nbfeat_hop2`
Means of the neighbours' neighbour means (features, not scores). The score version was unreadable
(0.992); the feature version is the last untried direction on this axis. Low prior; build and gate,
upload only if it passes.

### Not on the list
- Model-B depth / column fraction under the larger input (setting-level axes, closed on the board).
  The 318-column file reached 0.96177 with the unchanged recipe; a re-tune would most likely rank
  ≥ 0.98 like it (B_reg 0.980, B_mono 0.988).
- Any further statistic or subset of the neighbour features (closed above).
- Bridging / test-row self-training (argued against by the 2026-10-07 diagnostics).

## Finals track (private 70%; two hand-picked files)

Policy: the highest public file plus the strongest file that ranks < 0.97 like it.
- Final 1: `fam_B15_nbfeat_all` (0.96177).
- Final 2: `fam_B15_Aoof_random` (0.95849, ranks 0.960 like Final 1) by the policy; the alternative is
  `fam_B15_nbfeat` (0.96069, ranks 0.979 like Final 1), which gives up some hedge for 0.002 of
  public score. Decide at the end; if candidate 1 or 2 wins it becomes Final 1.

## Notebook standard

`final_model*.ipynb` reproduce `fam_B15_nbfeat` exactly (2026-10-08). The all-159 variant is a
`CONFIG` change (`nb_features` = the 159 cleaned columns); candidate 1 is a code change (the
feature means also enter model A) and goes into the notebook only if adopted.

## Exit condition

If candidates 1 and 3 both fail to read a gain, the neighbour axis is closed on both the amount
and the entry point. Remaining effort then goes to the finals and the report, with one slot a day
kept for an idea that changes the information set and passes the gate.

## Status (2026-10-08, evening)

Candidate 1 `B15_nbfeat_A` ranks **0.991** like the reference (CV 3-fold 0.9708, horizon 0.8288, the
family's best CV): unreadable, not uploaded. Feeding the feature means to model A changes its
scores too little to move the final ranking; model B already extracts what those columns carry.
The entry point is closed. Candidate 3 (`B15_nbfeat_hop2`, two-hop feature means) is being built
and gated as the plan says; the third slot of 2026-10-08 is held unless it passes.

Candidate 3 `B15_nbfeat_hop2` ranks **0.988** like the reference (CV 3-fold 0.9692): unreadable, not
uploaded. **Exit condition reached.** The neighbour-feature axis is closed on amount (top-10 ≈ all
159), statistic (max, delta), entry point (model A) and reach (two hops). Six files built on
2026-10-08, two read (one gain, one tie), four unreadable. Third slot of the day held.

Standing from here: finals = `fam_B15_nbfeat_all` + `fam_B15_Aoof_random` (alternative Final 2
`fam_B15_nbfeat`); `final_model*.ipynb` reproduce the reference; report work resumes; at most one
slot a day on an idea that changes the information set and passes the gate.

## Round 2 (approved by the user 2026-10-08, after the research below)

Decisions: (1) the third slot of 2026-10-08 goes to a **seed replicate of the leader**
(`B15_nbfeat_all_seedsB`, seeds 5–9): it fails the distinctness gate by design and is uploaded as a
noise reading, to tell whether 0.96177 vs 0.96069 is seed luck (the finals pair depends on it);
(2) two closed setting axes get **one re-read each under the larger model-B input**: per-step
balanced row weights (`B15_nbfeat_stepbal`, a period-invariance principle not tried before) and
23 leaves (`B23_nbfeat`); (3) one new information source: **strictly out-of-fold GNN embeddings as
extra model-B columns** (the literature's "trees + node embeddings" recipe; our in-sample version
leaked labels in step 3); transductive self-training and unlabelled-neighbour feature means stay
held; (4) finals = the policy pair `fam_B15_nbfeat_all` + `fam_B15_Aoof_random`.

Research grounding (methods only, never facts about this data):
- Trees on tabular features beat GNNs on the public dataset with this structure, and adding node
  embeddings to the trees was the one graph contribution that helped (Weber et al. 2019,
  arXiv 1908.02591); BGNN (Ivanov & Prokhorenkova, ICLR 2021) makes the same point by feeding
  GBDT predictions into a GNN. Our two-stage is the collective-classification / stacked-learning
  idea (neighbour predictions as features; Fast & Jensen 2008; Kou & Cohen 2007).
- Temporal shift in tabular data (Cai et al., ICML 2025): recent-only training and random
  within-period splits beat time-ordered protocols — exactly what the board told us (window 14,
  random OOF); per-period reweighting / group-DRO is the robustness family behind the
  step-balanced weights.
- The post-43 collapse is reported for every model family in the literature on the public
  dataset; nothing published fixes it. We treat it as unfixable and hedge on the private board.
