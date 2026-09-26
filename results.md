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
| 2026-09-26 | sub_06_lgbm_clean_bagged5.csv | cleaned raw, bagged x5 (no normalisation) | 159 | 300 x5 | 0.9819 | 0.9555 | 0.8966 | 0.9155 | 0.93977 |
| 2026-09-26 | sub_04_lgbm_v2b_ranknorm_bagged5.csv | v2b bagged x5 | 159 | 300 x5 | 0.9822 | 0.9567 | 0.9135 | 0.9180 | 0.94200 |
| 2026-09-26 | sub_05_lgbm_v2_ranknorm_discrete_bagged5.csv | v2 (incl. discrete) bagged x5 | 159 | 300 x5 | 0.9815 | 0.9575 | 0.9173 | 0.9154 | 0.94374 |

## Step 3: tuning and GNN

| date | file | config | n features | rounds | CV AUC 3-fold | CV AUC hard fold 22-28 | CV AUC horizon 22-35 | CV PR-AUC | public LB |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-26 | sub_07_lgbm_v2b_tuned_bagged5.csv | v2b, tuned (7 leaves, lr 0.05, 600 rounds, subsample 1.0), bagged x5 | 159 | 600 x5 | 0.9852 | 0.9668 | 0.9405 | 0.9258 | _pending_ |
