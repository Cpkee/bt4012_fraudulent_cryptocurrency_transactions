# BT4012 — fraudulent cryptocurrency transactions

Kaggle competition `bt-4012-competition-2026`: score each Bitcoin transaction by its probability of being illicit;
the metric is ROC AUC on the pooled test set (steps 36–49).

**Final model (public 0.97066):** a two-stage LightGBM trained on the labelled rows of the last 14 training steps.
Model A scores every transaction from its own 159 features. Model B re-scores it from those features, its labelled
neighbours' model-A scores (6 columns) and their feature means (318 columns), with random split thresholds
(`extra_trees`) and illicit rows up-weighted. Two model-B variants (all columns, and a block ensemble) at three class
weightings, 15 seeds each, are rank-averaged.

| selection | file | public ROC AUC |
|---|---|---|
| Final 1 | `fam_F1_wavg.csv` (the six members above) | 0.97066 |
| Final 2 | `fam_F1_unbal.csv` (the two variants at the balanced class weight only) | 0.97020 |

## Quick start

Requirements: Python 3.9 (the pins in `requirements.txt` are the last releases that support it). On macOS, LightGBM
needs OpenMP: `brew install libomp`.

```bash
git clone https://github.com/Cpkee/bt4012_fraudulent_cryptocurrency_transactions.git
cd bt4012_fraudulent_cryptocurrency_transactions
python3.9 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 1. Get the data

The notebooks look for the four competition files (`train.csv`, `test.csv`, `txs_edgelist.csv`,
`sample_submission.csv`) in this order: anywhere under `/kaggle/input`, then `~/.cache/bt4012/bt-4012-competition-2026/`,
then `./data/`. Either:

- **Manual:** download them from the competition's Data tab on Kaggle and put them in `./data/`, or
- **API:** accept the competition rules on Kaggle, put your API token in `.env` (keep the `KGAT_` prefix), and run:

```bash
echo 'KAGGLE_KEY=KGAT_...' > .env
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from dotenv import load_dotenv; load_dotenv(); \
from kaggle_data import download_competition; print(download_competition('bt-4012-competition-2026'))"
```

This downloads into `~/.cache/bt4012/bt-4012-competition-2026/` and is a no-op once the files are there.

### 2. Run the hand-in notebook

`e1130084.ipynb` is self-contained (it imports no project modules) and goes from the raw files to `submission.csv`.
Open it in Jupyter or VS Code and run all cells, or from the shell:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute e1130084.ipynb \
    --output e1130084.executed.ipynb --ExecutePreprocessor.timeout=-1
```

- Output: `submission.csv` in the repository root (`/kaggle/working/submission.csv` on Kaggle).
- Runtime: about 16 minutes on a laptop CPU for Final 1.
- Switches in the first code cell:

| setting | default | effect |
|---|---|---|
| `CONFIG["class_weights"]` | three weightings | `{"balanced": dict(is_unbalance=True)}` reproduces **Final 2** (2 members, about 10 minutes) |
| `CONFIG["extra_windows"]` | `()` | `(21,)` adds the six members refitted on 21 steps (`fam_F1_wavg_win`, scored 0.97034, not selected; about 30 minutes) |
| `FAST` | `False` | `True`: 2 seeds for the submission and 1 for validation, a quick check run (not the uploaded file) |
| `USE_GPU` | `False` | `True`: LightGBM GPU learner where available (Kaggle); results then differ slightly from the uploaded file |

**Reproducibility check.** At the end of section 3 the notebook compares its output with the uploaded files if they
are present (in `submissions/` locally, or attached as a dataset on Kaggle). It prints
`reproduced 7 of 7 uploaded files exactly` for Final 1 (3 of 3 for Final 2) and stops with an error if any file
differs by more than 1e-9. In a fresh clone `submissions/` is empty (gitignored), so each file is reported as
"not found here, not checked" and the run still completes.

### 3. On Kaggle

Upload `e1130084.ipynb` (or `final_model_kaggle.ipynb`) as a Kaggle notebook, attach the competition data, optionally
set `USE_GPU = True`, and run all cells. The data is found anywhere under `/kaggle/input`.

## Other ways to run

| what | how |
|---|---|
| Variant notebook | `final_model.ipynb`: the same pipeline with more `CONFIG` switches (`members` picks the model-B variants and class weightings, `oof` the out-of-fold protocol, `nb_features` the neighbour columns). The default reproduces Final 1. `final_model_kaggle.ipynb` is the same with a GPU switch and Kaggle paths |
| One experiment | `.venv/bin/python src/families.py <name>` builds a named variant from the `FAMILIES` dict in `src/families.py`: CV, distinctness gate against every scored file, `submissions/fam_<name>.csv` and a ledger row in `docs/results.md`. Example: `src/families.py ref` reproduces the base single-stage recipe. A two-stage build takes 5–30 minutes |
| Rank blend of existing files | `.venv/bin/python src/families.py blend <prefix1> <prefix2> ...`: each prefix must match exactly one CSV in `submissions/` (e.g. `sub_06`); writes `submissions/fam_blend_<...>.csv` |
| Exploration notebook | `notebooks/main.ipynb` (steps 1–3: EDA, adversarial validation, graph features, GNNs); about 2 hours, re-runs every early experiment |

In zsh, pass several variant names literally or as `${=NAMES}`: a plain `$NAMES` is not split into separate arguments.
GNN code (`src/gnn*.py`) needs `torch` and `torch_geometric` (in `requirements.txt`); the final notebooks do not.

## Layout

```
e1130084.ipynb             hand-in notebook: raw files -> submission.csv, laid out as report sections 2.1-2.3
final_model.ipynb          variant notebook with CONFIG switches (default = Final 1)
final_model_kaggle.ipynb   the same for a Kaggle notebook (GPU switch, /kaggle paths)
submission.csv             output of the last notebook run
src/
  families.py              experiment harness: variants (FAMILIES), two-stage model, distinctness gate, ledger rows
  validation.py            rolling-origin folds, horizon fold, per-step AUC
  models.py drift.py graph_features.py probes.py blend_cv.py post43_probes.py post43_prescreen.py
  gnn.py gnn_cache.py gnn_cache_window.py   GraphSAGE / MLP models and their cache builders
  kaggle_data.py           competition download with a KGAT_ token into ~/.cache/bt4012
  paths.py                 repository and data locations used by every module
notebooks/main.ipynb       exploration and early experiments (steps 1-3)
docs/
  results.md               ledger: every file with its CV and public score
  report_notes.md          methodology and results write-up
  instructions.md          competition brief
  plans/                   step plans 1-12
  figures/                 model diagram and its script
submissions/  artifacts/   generated CSVs, CV tables and caches (gitignored)
```
