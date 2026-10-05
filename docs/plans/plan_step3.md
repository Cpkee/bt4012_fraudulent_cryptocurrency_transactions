# Step 3 — GNN under a strictly inductive protocol, plus the one untried tabular lever

Scope: option **C** from the brainstorm (GNN, honestly compared) and the LightGBM hyperparameter
search that step 2's noise analysis pointed at. Collective classification (step 4) and
self-supervised pretraining (step 5) stay out unless the GNN surprises.

Starting point (`results.md`): sub_04, bagged LightGBM on 159 features (150 raw + 9 within-step
ranks). Rolling CV 0.9822 / hard fold 0.9567 / horizon fold 0.9135; public LB 0.942, rank 19/20.
Public-LB noise: seed spread 0.016 on the horizon fold, 30% subsample std 0.006.

## Status (2026-09-27)

- §0 housekeeping: first commit made by the user (f11f461); noise paragraph written into
  `report_notes.md` §3; torch / PyG not yet installed (GNN not started).
- §1 hyperparameter search: **done and adopted**. Random search (40 configs) found the optimum on
  the small-leaves edge; a leaves × rounds grid located the plateau at **7 leaves × 600 rounds,
  lr 0.05, no row subsampling** (`models.LGBM_TUNED`). Five seeds: horizon 0.9147 → 0.9420, hard
  0.9568 → 0.9675, 3-fold 0.9821 → 0.9854. Notebook §16, `artifacts/lgbm_search*.csv`.
- sub_07 written (tuned, v2b, bagged ×5), ledger row under "Step 3"; public LB pending upload.
- §2 GNN: **done** (`gnn.py`, `artifacts/cv_gnn*.csv`, report_notes §2.3). Strict inductive
  protocol with a no-overlap assertion. MLP (no edges) > GraphSAGE on every fold; rewired edges
  beat real edges at the horizon; full-graph context helps a little but stays below the MLP; the
  neural family is ~0.03 below the tuned LightGBM. Hybrids (embeddings → LightGBM, rank blend)
  run as campaign probes `gnn_embed` / `gnn_blend` (plan_step4.md).
- torch 2.6 (CPU) and torch_geometric 2.6 installed and pinned in requirements.txt.

---

## Why this step

- The brief lists graph neural networks among the models to explore (section 2.3). The report
  currently has no GNN.
- Step 1's edge-shuffle ablation and the 2026 re-evaluation (arXiv 2604.19514) both say the
  topology carries no forward-transferable signal. A GNN run under the same leakage-free folds
  either confirms that with a stronger method or overturns it. Both are results.
- LightGBM's tree parameters were never tuned, and a seed spread of 0.016 on the horizon fold
  says the model is higher-variance than it needs to be. This is the last cheap, plausible lever.

## 0. Housekeeping first (short)

1. **First git commit.** Every file is still untracked and several Claude sessions have touched
   the repo. Commit the code, notebook, plans, notes, figures, `results.md` and `submissions/`;
   keep `artifacts/` and `.env` ignored. Needs the user's go-ahead; do it before any step-3 edit.
2. **Noise paragraph** in `report_notes.md` section 3: seed spread, subsample std, correlation
   between submissions, and why sub_03's public score was a lucky seed. Numbers already computed.
3. Environment: `torch` (CPU) and `torch_geometric` pinned for Python 3.9 in `requirements.txt`.
   Apple Silicon, CPU only; the graphs are small enough (≤142k nodes, 163k edges) for full-batch.

## 1. LightGBM hyperparameter search (cheap, run first, in the background)

- Random search, ~40 configurations over: `num_leaves` {15, 31, 63, 127}, `min_child_samples`
  {10, 20, 40, 80, 160}, `colsample_bytree` {0.3, 0.5, 0.6, 0.8}, `subsample` {0.6, 0.8, 1.0},
  `reg_lambda` {0, 1, 5, 20}, `reg_alpha` {0, 1, 5}, `learning_rate` {0.02, 0.03, 0.05} with
  rounds scaled so `rounds × lr ≈ 9` (the current 300 × 0.03).
- Features fixed at v2b. Four folds, two seeds each. Rank by horizon-fold AUC with the 3-fold
  mean as a guardrail (must not drop more than 0.001).
- Top 3 confirmed with five seeds. Adopt only if the horizon fold improves by more than 0.002.
- Also record per-configuration seed spread: a configuration with the same mean and half the
  spread is worth taking even without a mean gain, because the private LB is one draw.

## 2. GNN, strictly inductive

### Protocol (the point of the exercise)

- Folds and metric identical to steps 1–2 (`FOLDS_WITH_HORIZON`, pooled and per-step AUC, gap).
- For a fold with training steps ≤ T: the training graph contains **only** nodes with step ≤ T
  (all of them, labelled and unlabelled, since message passing over unlabelled nodes is
  legitimate); loss on labelled nodes only. The validation graph contains only the validation
  steps' nodes. Because edges never cross steps, no validation node is ever adjacent to a
  training node, so the model cannot see validation structure during training.
- Test graph: the 15,329 labelled test nodes and the 12,724 edges among them. Dangling edges to
  the 46,668 unlabelled test-period nodes are dropped (no features exist for them).
- **Parity variant**: because the test graph has only labelled nodes, also train on the
  labelled-only train graph (31,235 nodes) so train and test neighbourhoods have the same
  sparsity. Both variants are reported.

### Features and preprocessing

- Node features = the v2b matrix (150 raw + 9 within-step ranks). Raw columns are z-scored with
  statistics from the fold's training nodes only; rank columns are already in [0, 1].
- No time step, no period identifiers, no graph statistics as input features.

### Models (all in `gnn.py`, PyTorch Geometric)

| model | purpose |
|---|---|
| MLP, same width/depth, no edges | the control that isolates message passing; a GNN that does not beat it is using nothing from the graph |
| GraphSAGE, 2 layers, hidden 128, mean aggregation, dropout 0.3, linear root/skip connection | the main GNN (the skip is the Alarab et al. 2020 improvement on Elliptic) |
| GraphSAGE on randomly rewired edges (within step, DAG-preserving, `graph_features.shuffle_edges`) | the ablation from step 1, repeated for the GNN |
| GAT, 2 layers, 4 heads (optional, only if GraphSAGE beats the MLP) | attention variant |

Training: Adam, lr 1e-3, weight decay 5e-4, unweighted BCE (AUC is rank-based; a weighted
variant is one extra row), full-batch, fixed epoch count chosen on CV from {100, 200, 300} the
same way the LightGBM round count was chosen. No early stopping on a drifting tail. Three seeds.
Estimated cost: ~2–3 minutes per fit on CPU, ~50 fits, ~2 hours in the background.

### Hybrids (only if the GNN is at least competitive with the MLP)

- GNN penultimate embeddings (128-d) concatenated to the v2b features → LightGBM, same folds.
  This is the +0.008 trick from Weber et al. 2019.
- Rank blend LightGBM + GNN probabilities at weights 0.1 / 0.2 / 0.3.

### Decision rule

Adopt a GNN-derived component only if it improves the horizon fold by more than 0.002 without
lowering the 3-fold mean. Otherwise the GNN is reported as a negative result with three numbers
that make the negative informative: GNN vs MLP (does message passing help at all), real vs
rewired edges (is it the topology), full vs labelled-only graph (does the parity gap matter).

## 3. Submissions (3 per day)

| file | content | purpose |
|---|---|---|
| sub_07 | tuned LightGBM, v2b, bagged ×5 | prices the tuning (vs sub_04) |
| sub_08 | best GNN alone (GraphSAGE or hybrid) | one leaderboard point for the GNN row of the report |
| sub_09 | best overall (tuned LightGBM + GNN component if adopted, else sub_07 with 10 seeds) | final candidate |

Log all three in the step-3 table of `results.md` with the horizon-fold column.

## 4. Report exhibits this step produces

- One table: MLP vs GraphSAGE (full graph, labelled-only graph, rewired edges) vs LightGBM on
  the same four folds. This is the section 2.3 "why not the GNN" evidence.
- Per-step AUC on the horizon fold, GNN vs LightGBM.
- Tuning table: top 5 configurations with mean and seed spread.
- Leaderboard ledger with the three new rows.

## Code changes

| file | change |
|---|---|
| `requirements.txt` | `torch<2.7`, `torch_geometric` |
| `gnn.py` (new) | fold-wise graph builder (period subgraphs, z-scoring on training nodes), `MLP`, `SAGE`, `GAT` modules with skip connection, `train_fold()` returning probabilities and embeddings, `evaluate_gnn()` mirroring `validation.evaluate` on the same folds |
| `models.py` | `lgbm_search()` random search helper with seed-spread reporting |
| `main.ipynb` | section 16: hyperparameter search; 17: GNN protocol, models, ablations; 18: hybrids, submissions |
| `report_notes.md` | section 3 noise paragraph now; GNN and tuning sections after the run |

## Verification checklist

- [ ] Git commit made; `git status` clean except ignored paths
- [ ] Noise paragraph in `report_notes.md` section 3
- [ ] Tuning table with seed spread; adopted or rejected by the stated rule
- [ ] GNN protocol assertion in code: no validation node id appears in any training-graph edge
- [ ] Four-way GNN table (MLP / SAGE full / SAGE labelled-only / SAGE rewired) on four folds, three seeds
- [ ] Hybrid rows if run; decision recorded
- [ ] sub_07–09 written, sanity-checked, logged; public LB filled in after upload
- [ ] `plan_step3.md` status block written

## What this step leaves out

Collective classification, self-supervised pretraining, temporal GNNs such as EvolveGCN. After
this step, modelling stops and the report is written; the deadline is 25 October 2026.
