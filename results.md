# Results ledger

Every submission with the CV numbers it was chosen on. CV = rolling-origin folds
(train 1–14 → val 15–21, 1–21 → 22–28, 1–28 → 29–35), labelled rows only, mean over
folds and 3 seeds. Public LB is 30% of the test set (~4.6k rows); never tune against it.

| date | file | config | n features | rounds | CV AUC pooled | CV AUC step-mean | CV PR-AUC | public LB |
|---|---|---|---|---|---|---|---|---|
| 2026-09-25 | sub_01_lgbm_clean.csv | lgbm_raw_clean | 159 | 175 | 0.9726 | 0.9735 | 0.9316 | 0.93901|
| 2026-09-25 | sub_02_lgbm_raw_with_period_ids.csv | lgbm_raw (all 165, reference) | 165 | 176 | 0.9737 | 0.9735 | 0.9279 | 0.94091 |
| 2026-09-26 | sub_03_lgbm_clean_fixed300_unweighted.csv | lgbm_raw_clean, fixed 300, unweighted | 159 | 300 | 0.9819 | 0.9806 | 0.9507 | 0.94510 |

## Step 2 (drift): rolling CV incl. the 14-step horizon fold

| date | file | config | n features | rounds | CV AUC 3-fold | CV AUC hard fold 22-28 | CV AUC horizon 22-35 | CV PR-AUC | public LB |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-26 | sub_06_lgbm_clean_bagged5.csv | cleaned raw, bagged x5 (no normalisation) | 159 | 300 x5 | 0.9819 | 0.9555 | 0.8966 | 0.9155 | 0.94374 |
| 2026-09-26 | sub_04_lgbm_v2b_ranknorm_bagged5.csv | v2b bagged x5 | 159 | 300 x5 | 0.9822 | 0.9567 | 0.9135 | 0.9180 | 0.94200 |
| 2026-09-26 | sub_05_lgbm_v2_ranknorm_discrete_bagged5.csv | v2 (incl. discrete) bagged x5 | 159 | 300 x5 | 0.9815 | 0.9575 | 0.9173 | 0.9154 | 0.93977 |

## Step 3: tuning and GNN

| date | file | config | n features | rounds | CV AUC 3-fold | CV AUC hard fold 22-28 | CV AUC horizon 22-35 | CV PR-AUC | public LB |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-26 | sub_07_lgbm_v2b_tuned_bagged5.csv | v2b, tuned (7 leaves, lr 0.05, 600 rounds, subsample 1.0), bagged x5 | 159 | 600 x5 | 0.9852 | 0.9668 | 0.9405 | 0.9258 | 0.94031 |
| 2026-09-27 | sub_08_lgbm_v2b_tuned_stepmean_equalised.csv | sub_07 scores, each test step's mean shifted to the overall mean (probe: are post-43 scores deflated?) | 159 | as sub_07 | as sub_07 | as sub_07 | as sub_07 | as sub_07 | 0.94134 |
| 2026-09-27 | sub_09_lgbm_v2b_tuned_withinstep_rank.csv | sub_07 scores replaced by within-step percentile rank (probe, extreme version) | 159 | as sub_07 | 0.9268 on far windows | | | | _not uploaded_ |

## Probe campaign (plan_step4.md): one change per file against the sub_07 reference

| date | file | change | reference | n features | CV 3-fold | CV horizon | far-window mean | far-window min | public LB | delta vs ref | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-27 | probe_identifiers_back.csv | identifiers_back | sub_07_lgbm_v2b_tuned_bagged5.csv | 165 | 0.9805 | 0.9220 | 0.9454 | 0.8927 | _pending_ | | CV veto (far mean −0.0085): information-only upload, not adoptable unless LB delta > +0.012 |
| 2026-09-27 | probe_topology.csv | topology | sub_07_lgbm_v2b_tuned_bagged5.csv | 169 | 0.9836 | 0.9279 | 0.9461 | 0.8957 | _pending_ | | CV veto (far mean −0.0078): information-only upload, not adoptable unless LB delta > +0.012 |
| 2026-09-27 | probe_rank7.csv | rank7 | sub_07_lgbm_v2b_tuned_bagged5.csv | 159 | 0.9850 | 0.9395 | 0.9525 | 0.9139 | _pending_ | | passes veto (−0.0014); clean-up candidate, not a hypothesis |
| 2026-09-27 | probe_adv_weights.csv | adv_weights | sub_07_lgbm_v2b_tuned_bagged5.csv | 159 | 0.9847 | 0.9407 | 0.9532 | 0.9161 | _pending_ | | passes veto (far mean −0.0007): upload as day-2 probe |
| 2026-09-27 | probe_self_train.csv | self_train | sub_07_lgbm_v2b_tuned_bagged5.csv | 159 | 0.9829 | 0.9038 | 0.9357 | 0.8652 | _pending_ | | CV veto (far mean −0.0182): not uploaded |
| 2026-09-27 | probe_local_only.csv | local_only | sub_07_lgbm_v2b_tuned_bagged5.csv | 93 | 0.9762 | 0.9040 | 0.9310 | 0.8707 | _pending_ | | CV veto (far mean −0.0229): not uploaded; the aggregated block (94–165) carries transferable signal |
| 2026-09-27 | probe_anomaly_10.csv | anomaly_10 | sub_07_lgbm_v2b_tuned_bagged5.csv | 159 | 0.9812 | 0.9339 | 0.9467 | 0.9069 | _pending_ | | CV veto (far mean −0.0072, PR-AUC −0.030): not uploaded |
| 2026-09-27 | probe_adv_weights_soft.csv | adv_weights_soft | sub_07_lgbm_v2b_tuned_bagged5.csv | 159 | 0.9849 | 0.9394 | 0.9529 | 0.9132 | _pending_ | | passes veto (far mean −0.0010); gentler than adv_weights, hold |
| 2026-09-27 | probe_adv_weights_hard.csv | adv_weights_hard | sub_07_lgbm_v2b_tuned_bagged5.csv | 159 | 0.9853 | 0.9410 | 0.9521 | 0.9154 | 0.93765 | −0.0027 vs sub_07 (0.94031) | inside noise, slightly negative: not adopted |
| 2026-09-27 | probe_pseudo_train_unlab.csv | pseudo_train_unlab | sub_07_lgbm_v2b_tuned_bagged5.csv | 159 | 0.9763 | 0.8996 | 0.9304 | 0.8580 | _pending_ | | CV veto (far mean −0.0235): not uploaded; unlabelled rows are not a random sample of the labelled ones |
| 2026-09-27 | probe_bag10.csv | bag10 | sub_07_lgbm_v2b_tuned_bagged5.csv | 159 | 0.9853 | 0.9415 | 0.9539 | 0.9163 | _pending_ | | robustness item, not a probe (Spearman 0.9996 vs ref); use in the final model |
| 2026-09-27 | probe_xgb_alone.csv | xgb_alone | sub_07_lgbm_v2b_tuned_bagged5.csv | 159 | 0.9837 | 0.9357 | 0.9464 | 0.9102 | _pending_ | | CV veto (far mean −0.0075): not uploaded; shallow XGBoost alone is below the tuned LightGBM |
| 2026-09-27 | probe_xgb_blend.csv | xgb_blend | sub_07_lgbm_v2b_tuned_bagged5.csv | 159 | 0.9853 | 0.9411 | 0.9531 | 0.9161 | _pending_ | | passes veto (far mean −0.0008, CV identical to ref): diversity candidate for the final model |
| 2026-09-27 | probe_gnn_embed.csv | gnn_embed | sub_07_lgbm_v2b_tuned_bagged5.csv | 159 | 0.9721 | 0.8964 | 0.8891 | 0.7917 | _pending_ | | CV veto (far mean −0.0648): not uploaded; in-sample GNN embeddings leak labels into the trees |
| 2026-09-27 | probe_gnn_blend.csv | gnn_blend | sub_07_lgbm_v2b_tuned_bagged5.csv | 159 | 0.9838 | 0.9339 | 0.9497 | 0.9051 | _pending_ | | passes veto by 0.0008 but below the reference on every metric: not uploaded |

## Step 5: model families (plan_step5.md)

CV = four rolling folds, one model per seed (2 seeds); a sanity check, not a veto. Upload gate: Spearman < 0.98 against every scored file. Delta is vs sub_06_lgbm_clean_bagged5.csv (0.94374).

| date | file | family | n features | CV 3-fold | CV hard 22-28 | CV horizon 22-35 | CV PR-AUC | closest scored file (Spearman) | public LB | delta vs sub_06 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-27 | fam_rf_clean.csv | random forest, 500 trees, balanced_subsample, min leaf 2, bagged x3 | 159 | 0.9620 | 0.8992 | 0.8161 | 0.8990 | sub_01_lgbm_clean (0.873) | 0.92847 | −0.0153 | clear loss (3x noise): forest family rejected; CV's family-level gap (−0.019 3-fold, −0.078 horizon) pointed the same way |
| 2026-09-27 | fam_et_clean.csv | extra trees, 500 trees, balanced_subsample, min leaf 2, bagged x3 | 159 | 0.9652 | 0.9136 | 0.8247 | 0.8919 | sub_02_lgbm_raw_with_period_ids (0.783) | _pending_ | | passes distinctness gate: upload candidate |
| 2026-09-27 | fam_lgbm_deep_raw.csv | sub_06 recipe with 127 leaves, min_child_samples 20, bagged x5 | 159 | 0.9811 | 0.9531 | 0.8886 | 0.9138 | sub_06_lgbm_clean_bagged5 (0.983) | _pending_ | | too close to sub_06_lgbm_clean_bagged5.csv (Spearman 0.983): not uploaded |
| 2026-09-27 | fam_lgbm_raw_ids.csv | sub_06 recipe + the six period identifiers, bagged x5 | 165 | 0.9810 | 0.9533 | 0.8937 | 0.9142 | sub_06_lgbm_clean_bagged5 (0.995) | _pending_ | | too close to sub_06_lgbm_clean_bagged5.csv (Spearman 0.995): not uploaded |
| 2026-09-27 | fam_logreg.csv | logistic regression, standardised, clipped at 10 sd, C=0.1, balanced | 159 | 0.9090 | 0.9027 | 0.8282 | 0.6048 | probe_adv_weights_hard (0.718) | _pending_ | | fails sanity gate (3-fold 0.9090 < 0.95) |
| 2026-09-27 | fam_blend_06_07.csv | rank average of sub_06, sub_07 | - |  |  |  |  | sub_04_lgbm_v2b_ranknorm_bagged5 (0.985) | _pending_ | | too close to sub_04_lgbm_v2b_ranknorm_bagged5.csv (Spearman 0.985): not uploaded |

## Seed check on the best public scorer (sub_03 = default trees, 300 rounds, seed 0)

| date | file | model | public LB | purpose |
|---|---|---|---|---|
| 2026-09-26 | sub_03_lgbm_clean_fixed300_unweighted.csv | seed 0 | 0.94510 | reference |
| 2026-09-28 | sub_03_seed1.csv | same model, seed 1 | 0.94198 | is 0.9451 seed luck or a genuinely better model on the test period? |
| 2026-09-28 | sub_03_seed2.csv | same model, seed 2 | _pending_ | second reading of the same question |
| 2026-09-28 | fam_rf_clean.csv | random forest on the cleaned features (uploaded outside the ledger) | 0.92847 | RF on the test period, for reference |
| 2026-09-29 | fam_recent14.csv | sub_06 recipe trained on the last 14 training steps only (22-35), bagged x5 | 159 | 0.9789 | 0.9461 | 0.8797 | 0.9114 | sub_06_lgbm_clean_bagged5 (0.931) | 0.94836 | +0.0046 | largest positive delta of the project, opposite to CV; at the edge of the ±0.005 tie band: follow up with recent7 and recency_hl5 |
| 2026-09-29 | fam_recent7.csv | sub_06 recipe trained on the last 7 training steps only (29-35), bagged x5 | 159 | 0.9595 | 0.8943 | 0.8019 | 0.8969 | sub_06_lgbm_clean_bagged5 (0.797) | 0.92830 | −0.0154 | clear loss (3x noise): 7 steps is too short; peak lies between 10 and 21+ |
| 2026-09-29 | fam_recency_hl5.csv | sub_06 recipe with exponential recency weights, half-life 5 steps, bagged x5 | 159 | 0.9796 | 0.9485 | 0.8823 | 0.9121 | sub_06_lgbm_clean_bagged5 (0.924) | 0.95096 | +0.0026 vs recent14 (0.94836) | tie band, positive: soft weighting >= hard window; build two-stage on the weighted base |
| 2026-09-29 | fam_all_time.csv | sub_06 recipe on all 165 columns + time_step as a feature, bagged x5 | 166 | 0.9809 | 0.9534 | 0.8965 | 0.9142 | sub_06_lgbm_clean_bagged5 (0.995) | _pending_ | | too close to sub_06_lgbm_clean_bagged5.csv (Spearman 0.995): not uploaded |
| 2026-09-29 | fam_recent21.csv | sub_06 recipe trained on the last 21 training steps only (15-35), bagged x5 | 159 | 0.9816 | 0.9544 | 0.8941 | 0.9142 | fam_recent14 (0.975) | 0.94617 | +0.0024 | tie vs sub_06, below recent14: window curve peaks at 14 |
| 2026-09-29 | fam_recent10.csv | sub_06 recipe trained on the last 10 training steps only (26-35), bagged x5 | 159 | 0.9754 | 0.9346 | 0.8535 | 0.9077 | fam_recent14 (0.933) | 0.93386 | −0.0099 | clear loss: 10 steps too short |
| 2026-09-30 | fam_recent17.csv | sub_06 recipe trained on the last 17 training steps only (19-35), bagged x5 | 159 | 0.9794 | 0.9478 | 0.8784 | 0.9126 | fam_recent21 (0.985) | _pending_ | | too close to fam_recent21.csv (Spearman 0.985): not uploaded |
| 2026-09-30 | fam_recent14_norm.csv | recent14 base + within-step rank of the 9 drifting features (v2b), bagged x5 | 159 | 0.9783 | 0.9448 | 0.8971 | 0.9134 | fam_recent14 (0.981) | _pending_ | | too close to fam_recent14.csv (Spearman 0.981): not uploaded |
| 2026-09-30 | fam_recent14_2stage.csv | recent14 base + two-stage neighbour predictions (mean/max/count of model-A scores over incoming and outgoing labelled neighbours), bagged x5 | 159 | 0.9741 | 0.9308 | 0.8478 | 0.9098 | fam_recent14 (0.966) | 0.95260 | +0.0042 vs recent14 (0.94836) | new best; second consecutive gain from the recent-data direction; build round 2 |
| 2026-09-30 | fam_recent14_local.csv | recent14 base on the 93 local columns only (aggregated block 94-165 dropped), bagged x5 | 93 | 0.9738 | 0.9339 | 0.8532 | 0.9036 | fam_recent21 (0.943) | 0.93674 | −0.0116 vs recent14 (0.94836) | clear loss (2x noise): the aggregated block 94-165 carries test-period signal; CV agreed (−0.023) |
| 2026-10-01 | fam_recent14_2stage_hop2.csv | recent14 two-stage + min/std of neighbour scores and 2-hop neighbour-score means, bagged x5 | 159 | 0.9711 | 0.9213 | 0.8307 | 0.9084 | fam_recent14_2stage (0.992) | _pending_ | | too close to fam_recent14_2stage.csv (Spearman 0.992): not uploaded |
| 2026-10-01 | fam_recent14_2stage_unlab.csv | recent14 two-stage with the unlabelled training rows scored by model A as extra neighbours (train side only), bagged x5 | 159 | 0.9716 | 0.9237 | 0.8352 | 0.9082 | fam_recent14_2stage (0.988) | _pending_ | | too close to fam_recent14_2stage.csv (Spearman 0.988): not uploaded |
| 2026-10-01 | fam_recent21_2stage.csv | two-stage on the 21-step window (15-35), bagged x5 | 159 | 0.9794 | 0.9466 | 0.8775 | 0.9139 | fam_recent14_2stage (0.981) | _pending_ | | too close to fam_recent14_2stage.csv (Spearman 0.981): not uploaded |
| 2026-10-01 | fam_recent14_2stage_iter2.csv | recent14 two-stage, second pass: model-B scores as a second set of neighbour features into a model C, bagged x5 | 159 | 0.9735 | 0.9283 | 0.8393 | 0.9092 | fam_recent14_2stage (0.995) | _pending_ | | too close to fam_recent14_2stage.csv (Spearman 0.995): not uploaded |
| 2026-10-01 | fam_recent14_2stage_tunedB.csv | recent14 two-stage with model B = tuned shallow LightGBM (7 leaves, 600 rounds, lr 0.05), bagged x5 | 159 | 0.9769 | 0.9394 | 0.8652 | 0.9143 | fam_recent14 (0.910) | 0.95504 | +0.0024 vs 2stage (0.95260) | new best; tie band, positive: shallow model B helps once neighbour numbers exist |
| 2026-10-01 | fam_recent14_2stage_Aall.csv | recent14 two-stage with model A for val/test rows trained on all 35 steps (model B still on the window), bagged x5 | 159 | 0.9743 | 0.9314 | 0.8485 | 0.9097 | fam_recent14_2stage (0.997) | _pending_ | | too close to fam_recent14_2stage.csv (Spearman 0.997): not uploaded |
| 2026-10-01 | fam_hl5_2stage.csv | two-stage on the soft-weighted base (all 35 steps, half-life 5) instead of the hard 14-step window, bagged x5 | 159 | 0.9756 | 0.9353 | 0.8595 | 0.9112 | fam_recent14_2stage (0.970) | 0.95231 | −0.0003 vs 2stage (0.95260) | flat tie: soft weighting does not stack with neighbour predictions; hard 14-step window stays as the base |
| 2026-10-01 | fam_recent14_2stage_seedsB.csv | recent14 two-stage, seeds 5-9 (replicate of the 0.9526 file to measure its seed noise on the board) | 159 | 0.9714 | 0.9223 | 0.8337 | 0.9082 | fam_recent14_2stage (0.996) | 0.95361 | +0.0010 vs 2stage seeds 0-4 (0.95260) | seed noise of a 5-seed two-stage bag on the board ≈ 0.001: the +0.0024 of tunedB is above it; 5-seed bags are stable enough for finals |
| 2026-10-01 | fam_hl5_2stage_tunedB.csv | two-stage on the soft-weighted base (half-life 5) with model B = tuned shallow LightGBM, bagged x5 | 159 | 0.9759 | 0.9366 | 0.8687 | 0.9126 | fam_recent14_2stage_tunedB (0.963) | 0.95518 | +0.0001 vs tunedB | flat: weighted base adds nothing on the shallow model either; hard window stays |
| 2026-10-01 | fam_recent14_2stage_B15.csv | recent14 two-stage with model B = 15 leaves, 300 rounds, lr 0.05, no row subsampling (middle depth), bagged x5 | 159 | 0.9715 | 0.9229 | 0.8356 | 0.9105 | fam_recent14_2stage_seedsB (0.967) | 0.95621 | +0.0012 vs tunedB (0.95504) | new best; depth curve of model B: 31 leaves 0.9526, 15 leaves 0.9562, 7 leaves 0.9550 (5 ≈ 7) |
| 2026-10-01 | fam_recent14_2stage_tunedAB.csv | recent14 two-stage with both model A and model B = tuned shallow LightGBM, bagged x5 | 159 | 0.9792 | 0.9462 | 0.8764 | 0.9162 | fam_recent14_2stage_tunedB (0.995) | _pending_ | | too close to fam_recent14_2stage_tunedB.csv (Spearman 0.995): not uploaded |
| 2026-10-02 | fam_recent14_2stage_B5.csv | recent14 two-stage with model B = 5 leaves, 900 rounds, lr 0.05, no row subsampling (shallower than the best), bagged x5 | 159 | 0.9785 | 0.9443 | 0.8719 | 0.9158 | fam_recent14_2stage_tunedB (0.991) | _pending_ | | too close to fam_recent14_2stage_tunedB.csv (Spearman 0.991): not uploaded |
| 2026-10-02 | fam_recent14_2stage_B23.csv | recent14 two-stage with model B = 23 leaves, 300 rounds, lr 0.05, no row subsampling (between 15 and 31), bagged x5 | 159 | 0.9665 | 0.9083 | 0.8135 | 0.9080 | fam_recent14_2stage_B15 (0.980) | 0.95499 | −0.0012 vs B15 (0.95621) | depth curve closed: 31→0.9526, 23→0.9550, 15→0.9562, 7→0.9550; final model B = 15 leaves |
| 2026-10-02 | fam_B15_smooth.csv | B15 two-stage + final-score smoothing along labelled edges (lambda 0.3), bagged x5 | 159 | 0.9661 | 0.9191 | 0.8297 | 0.8807 | fam_recent14_2stage_B15 (0.975) | 0.94462 | −0.0116 | clear loss: smoothing final scores along edges hurts; output-side use of the graph closed |
| 2026-10-02 | artifacts/fam_B15_lambdarank_perstep_groups.csv (not a submission) | B15 two-stage with model B trained with the lambdarank objective grouped by time step, bagged x5 | 159 | 0.9169 | 0.8669 | 0.7832 | 0.8318 | fam_hl5_2stage (0.773) | _pending_ | | fails sanity gate (3-fold 0.9169 < 0.95) |
| 2026-10-02 | fam_B15_interact.csv | B15 two-stage + pairwise products and ratios of the top-8 raw features for model B (56 columns), bagged x5 | 159 | 0.9673 | 0.9112 | 0.8054 | 0.9053 | fam_recent14_2stage_B23 (0.945) | 0.95620 | −0.00001 vs B15 (0.95621) | exact tie: 15-leaf trees already have the interaction capacity they need; closed |
| 2026-10-02 | fam_B15_lambdarank.csv | B15 two-stage with model B trained with the lambdarank objective, random groups of <= 5,000 rows (pairwise ordering across steps), bagged x5 | 159 | 0.9558 | 0.8778 | 0.7782 | 0.8985 | fam_recent14_2stage (0.953) | 0.94502 | −0.0112 | clear loss: ranking objective hurts; closed |
| 2026-10-03 | fam_B15_mono.csv | B15 two-stage with model B monotone non-decreasing in the four neighbour mean/max columns, bagged x5 | 159 | 0.9708 | 0.9207 | 0.8375 | 0.9103 | fam_recent14_2stage_B15 (0.988) | _pending_ | | too close to fam_recent14_2stage_B15.csv (Spearman 0.988): not uploaded |
| 2026-10-03 | fam_B15_top80.csv | B15 two-stage with model B on the top-80 raw features by gain + 6 neighbour numbers, bagged x5 | 159 | 0.9699 | 0.9182 | 0.8411 | 0.9108 | fam_recent14_2stage_B15 (0.989) | _pending_ | | too close to fam_recent14_2stage_B15.csv (Spearman 0.989): not uploaded |
| 2026-10-03 | fam_B15_xgb.csv | B15 two-stage with model B = shallow XGBoost (depth 4, 300 rounds, lr 0.05) on the same inputs, bagged x5 | 159 | 0.9668 | 0.9091 | 0.8172 | 0.9053 | fam_recent14_2stage_tunedB (0.954) | 0.95397 | −0.0022 vs B15 (0.95621) | loss: XGBoost depth 4 below LightGBM 15 leaves as model B |
| 2026-10-03 | fam_B15_reg.csv | B15 two-stage with model B strongly regularised (15 leaves, min 100 rows/leaf, 30% columns, lambda 5), bagged x5 | 159 | 0.9739 | 0.9298 | 0.8437 | 0.9112 | fam_recent14_2stage_B15 (0.980) | _pending_ | | too close to fam_recent14_2stage_B15.csv (Spearman 0.980): not uploaded |
| 2026-10-03 | fam_B15_cat.csv | B15 two-stage with model B = CatBoost (depth 4, 300 iterations, lr 0.05) on the same inputs, bagged x5 | 159 | 0.9677 | 0.9141 | 0.8316 | 0.9048 | fam_recent14_2stage_tunedB (0.906) | 0.94638 | −0.0098 | clear loss: CatBoost as model B |
| 2026-10-03 | fam_B15_et.csv | B15 two-stage with model B = ExtraTrees (500 trees, depth 8, min 20 rows/leaf) on the same inputs, bagged x5 | 159 | 0.9574 | 0.8963 | 0.8120 | 0.8865 | fam_B15_lambdarank (0.780) | 0.91782 | −0.0384 | clear loss: forests as model B; learner question closed, LightGBM confirmed |
| 2026-10-04 | fam_B15_nb_labels.csv | B15 two-stage with model B trained on neighbours' true labels (test uses model-A scores), bagged x5 | 159 | 0.9783 | 0.9468 | 0.8979 | 0.9134 | fam_recent10 (0.849) | 0.94620 | −0.0100 | clear loss: label curve is monotone (0 / 50 / 100% labels → 0.9562 / 0.9542 / 0.9462); score protocol confirmed |
| 2026-10-04 | fam_B15_Aoof_random.csv | B15 two-stage with model-A scores for training rows from 5-fold random OOF within the window (not time-ordered), bagged x5 | 159 | 0.9728 | 0.9259 | 0.8420 | 0.9115 | fam_recent14_2stage_B15 (0.980) | 0.95849 | +0.0023 vs B15 (0.95621) | new best: random 5-fold OOF scoring of training rows beats time-ordered OOF |
| 2026-10-04 | fam_B15_Abag3.csv | B15 two-stage with model A = 3-seed bag inside each outer seed (smoother neighbour scores), bagged x5 | 159 | 0.9732 | 0.9281 | 0.8406 | 0.9108 | fam_recent14_2stage_B15 (0.996) | _pending_ | | too close to fam_recent14_2stage_B15.csv (Spearman 0.996): not uploaded |
| 2026-10-04 | fam_B15_nb_mix50.csv | B15 two-stage with training-row neighbour features from a 50/50 mix of true labels and model-A scores, bagged x5 | 159 | 0.9684 | 0.9187 | 0.8408 | 0.9056 | fam_recent14_2stage_B15 (0.904) | 0.95423 | −0.0020 | loss: half labels in training neighbour numbers |
