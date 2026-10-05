# BT4012 — fraudulent cryptocurrency transactions

Kaggle competition `bt-4012-competition-2026`: score each Bitcoin transaction by its probability
of being illicit (ROC AUC). Current best: two-stage LightGBM with neighbour predictions,
public 0.95849 (`submissions/fam_B15_Aoof_random.csv`).

## Layout

```
final_model.ipynb          submission notebook: raw files -> preprocessing -> model -> submission.csv
final_model_kaggle.ipynb   same, for a Kaggle notebook (GPU switch, /kaggle paths)
submission.csv             output of the last final-model run
src/                       experiment code
  families.py              experiment harness: variants, distinctness gate, ledger rows
  validation.py models.py drift.py graph_features.py gnn.py gnn_cache.py probes.py
  kaggle_data.py           data download into ~/.cache/bt4012
  paths.py                 repo / data locations used by every module
notebooks/main.ipynb       exploratory and experiment notebook (steps 1-3)
docs/
  results.md               ledger: every submission with CV and public score
  report_notes.md          methodology and results write-up
  instructions.md          competition brief
  plans/                   step plans 1-6
  figures/                 model diagram and its script
submissions/  artifacts/   generated CSVs and CV tables (gitignored)
```

## Setup and running

```bash
python3.9 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo 'KAGGLE_KEY=KGAT_...' > .env          # Kaggle API token, keep the KGAT_ prefix
```

- Data: `src/kaggle_data.py` downloads it to `~/.cache/bt4012` (the first cells of
  `notebooks/main.ipynb` call it). The final notebooks also look in `/kaggle/input` and `data/`.
- Final model: open `final_model.ipynb` and run all cells.
- Experiments: `.venv/bin/python src/families.py <variant>` (e.g. `ref` reproduces the base recipe).
- Working rules for this repo are in `CLAUDE.md`.
