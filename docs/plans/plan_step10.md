# Step 10 — Lecture 8 (Social Network Analysis) mapped onto the pipeline

Date: 2026-10-09. Reference for one-change probes: **`fam_B15_nbfeat.csv` = 0.96069**; highest public
score `fam_B15_nbfeat_all_seedsB` 0.96323 (seed noise 0.0015 on this model). Queue today: slot 1
`fam_B15_nbfeat_gnn`, slot 2 `fam_B15_nbfeat_all_bag10`, **slot 3 = the step-10 candidate if it passes
the gate** (user decision 2026-10-09). Purpose stated by the user: personal learning first, board second.

## 1. Mapping: what the lecture teaches vs what this pipeline already does

| Lecture 8 technique | Where it sits in this project | Status |
|---|---|---|
| Representation: adjacency list, degree, in/out-degree, source / sink | `graph_features.topology_features` (in/out/total degree, in/out ratio, unlabelled-neighbour counts, component size, DAG depth/height). Step-4 probe ranked 0.994 like the reference | closed (unreadable) |
| Plain PageRank (uniform restart) | same probe, normalised within the component | closed |
| Tie strength / weighted edges | the edge list has two columns (`txId1`, `txId2`) and no weight; neighbourhood overlap is the only computable tie weight | not pursued |
| **Homophily test** (observed cross-edge rate r vs 2xy) | never measured before this step; it is the precondition the two-stage relies on | **measured below** |
| Betweenness, closeness, eigenvector centrality | never built; per-step graphs are 1,600–7,900 nodes, exact computation is cheap | open, low expectation (same family as topology) |
| Reciprocity | zero by construction: edges never cross steps and each step's graph is a DAG | not applicable |
| Density, clustering coefficient, diameter, small world, preferential attachment | descriptive; never built | report material only |
| **Fraud PageRank** (restart vector on known fraud = collective inference) | the two-stage is its one-hop cousin: mean/max/count of model-A scores over labelled in/out neighbours. Labels in training neighbour numbers hurt monotonically (0.9462), so the "seed" must be model-A scores on both sides. Multi-hop variants lost every time (2-hop means unreadable; output smoothing −0.012) | **measured below** (leave-self-out) |
| **Community detection** (modularity / Louvain, spectral, Girvan–Newman, CPM) | never built in any form | **candidate built: `B15_nbfeat_comm`** |
| Spectral / MDS node coordinates | the same idea as the out-of-fold GraphSAGE embedding file (`fam_B15_nbfeat_gnn`, slot 1 today) | wait for that reading |

The architecture already is the lecture's central claim ("being close to fraud is evidence of fraud",
operationalised as neighbour predictions into a second model). What the lecture adds that the pipeline
never used: (a) the homophily test as the justification, (b) multi-hop propagation with a restart,
(c) the community instead of the one-hop neighbourhood as the unit over which "what my neighbours look
like" is averaged. Only (c) changes which rows carry neighbour information.

## 2. Diagnostics (2026-10-09, read-only, scratchpad `sna_diag.py`, `ppr_self.py`; single seed)

### 2a. Homophily, per step, labelled–labelled train edges (steps 1–35)

r = share of labelled edges joining a licit to an illicit node; 2xy = the share expected if labels were
scattered at random; ratio = r / 2xy (< 1 = homophilic).

| step | labelled rows | fraud rate | labelled edges | r | 2xy | ratio |
|---|---|---|---|---|---|---|
| 1 | 2,147 | 0.008 | 1,924 | 0.007 | 0.016 | 0.46 |
| 9 | 778 | 0.319 | 484 | 0.242 | 0.434 | 0.56 |
| 13 | 809 | 0.360 | 564 | 0.245 | 0.461 | 0.53 |
| 22 | 1,763 | 0.090 | 1,537 | 0.037 | 0.163 | 0.23 |
| 29 | 1,174 | 0.280 | 911 | 0.060 | 0.403 | 0.15 |
| 33 | 441 | 0.052 | 429 | 0.077 | 0.099 | 0.78 |
| 35 | 1,341 | 0.136 | 1,002 | 0.063 | 0.235 | 0.27 |
| window 22–35 pooled | 11,705 | – | 9,317 | 0.063 | 0.249 (mean) | 0.25 |

**All 35 steps are homophilic** (ratio 0.11–0.84; median ≈ 0.27). The lecture's precondition holds on
this data, which is the data-grounded justification for the two-stage. Homophily cannot be measured on the
test steps (no labels), so nothing here says whether it weakens after step 43.

### 2b. Communities on the full per-step graph (Louvain, networkx, seed 0; unlabelled nodes as structure)

Every step's edge graph is one connected component (giant share 1.00), so components carry no
information; Louvain splits each step into 37–103 communities (modularity 0.84–0.93; median labelled
community size 65–235 nodes).

| quantity | train window 22–35 | test 36–49 |
|---|---|---|
| rows with a labelled 1-hop neighbour (current six numbers defined) | 0.780 | 0.746 |
| rows with another labelled row in their community | 0.995 | 0.997 |
| rows that gain a first labelled "neighbour" via the community | **0.216** | **0.251** |
| median / q90 other labelled members in the community | 31 / 118 | 37 / 102 |

Label-signal check in the window (labels of the *other* rows, leave-self-out; information only, never a
model input): 1-hop label mean AUC 0.8495 (defined 78%); community label mean AUC **0.9235** (defined
99.5%); on the 21.6% of rows with no 1-hop neighbour (fraud rate 0.282, vs 0.116 with one), the community
mean alone has AUC 0.913. Spearman between the two means 0.54: the community carries different
information, not a smoothed copy.

### 2c. Model-A-score columns (single seed, random 5-fold OOF in the window)

| column | defined | AUC alone (window) | Spearman vs 1-hop mean | vs own pA |
|---|---|---|---|---|
| nb_in_mean / nb_out_mean (current) | 0.53 / 0.54 | 0.920 / 0.901 | – | – |
| **comm_mean** (others in the community) | 0.995 | **0.924** | 0.61 | 0.59 |
| comm_max | 0.995 | 0.851 | 0.47 | 0.49 |
| comm_n, comm_size | 1.00 | 0.48, 0.42 | – | – |
| personalised PageRank, restart ∝ pA, undirected, *with* the row's own score in the restart | 1.00 | 0.980 | 0.63 | 0.60 |
| the same, **leave-self-out** (linear correction e_i(1−α)G_ii) | 1.00 | 0.896 | 0.61 | 0.48 |

The fraud-PageRank column looks strong only because the restart vector contains the row's own model-A
score: a stacked version of pA, and stacked scores lost on the board (logreg −0.004, kNN −0.003). With
the self term removed it is 0.896, a little above the 1-hop mean (0.877 on the same construction) and
below the community mean. The community mean is the better-defined, simpler quantity; PageRank is not
built.

Test period: comm_mean is defined for 99.7% of test rows on both sides of step 43; its median falls from
0.0125 (36–42) to 0.0031 (43–49), the same collapse the six numbers show.

## 3. Candidate (one change on `B15_nbfeat`): `B15_nbfeat_comm`

`two_stage(14, b_params=B15_PARAMS, b_estimators=300, a_oof="random", nb_feats=10, nb_comm=True)`:
three more model-B columns, `comm_mean`, `comm_max`, `comm_n` = mean, max and count of model-A scores
over the *other* scored rows of the row's community (blank where there are none), built exactly like the
six numbers: scored set = labelled window rows (random OOF) in training, test rows in the test file; the
community partition (`communities()`, cached at `artifacts/communities_louvain_s0.csv`) uses no label and no
feature, so train and test are treated alike. Column verified against a brute-force computation.

Readability expectation: the columns are defined for 22% (train) / 25% (test) of rows that today carry
no neighbour value, and the community mean is only 0.61 rank-correlated with the 1-hop mean, so the
ranking should move more than the 2-hop / bridge variants did (expected Spearman 0.93–0.97 vs the best).
Reading rule vs 0.96069: ≥ +0.003 gain (seed noise on this model is 0.0015), ≤ −0.003 loss, between = tie.

### Gate result (2026-10-09)

| file | CV 3-fold | hard | horizon | PR | Spearman vs best | verdict |
|---|---|---|---|---|---|---|
| fam_B15_nbfeat_comm.csv | **0.9769** | 0.9361 | **0.8511** | 0.9137 | 0.963 (`fam_B15_nbfeat`) | passes the gate: slot 3 of 2026-10-09 |

**Passes at 0.963**, the most distinct one-change file since the recent-window probes, and the first
neighbour-information change whose CV is *above* the reference (3-fold 0.9689 → 0.9769, horizon
0.8215 → 0.8511): every earlier neighbour gain (two-stage, random OOF, nbfeat) came with a lower CV.
Reading rule vs 0.96069: ≥ +0.003 gain, ≤ −0.003 loss.

Follow-ups only if it reads positive, one change each: community feature means (the top-10 raw features
over the community, the nbfeat idea on the community unit); community statistics at resolution 0.5 / 2;
the same columns on the all-159 base for the finals bag.
