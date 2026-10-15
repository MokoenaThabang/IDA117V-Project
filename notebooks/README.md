# Notebooks

Template-ordered Jupyter notebooks for Q1 (numbers only) and Q2 (full). They do **not** train Logistic Regression, Random Forest, MLP, or the embedding net.

| File | What it does |
| --- | --- |
| `q1.ipynb` | Locked Q1 counts: class mix, 7-day cut check (duration columns are banned), DummyClassifier val macro-F1 |
| `q2.ipynb` | Template 2.1 → 2.2 → 2.3 → 2.4: dictionary/audit, five train-only figures (including the temporal chart), split-before-fit, leakage-safe `prep.py` Pipeline |

## How to run

1. `python -m pip install -r requirements.txt`
2. Put frozen Drive splits in `data/processed/` (`train.parquet`, `val.parquet`, `test.parquet`). See [`data/README.md`](../data/README.md).
3. From the **repo root** (so `src/prep.py` and `data/processed/` resolve):

```bash
jupyter notebook notebooks/q1.ipynb
jupyter notebook notebooks/q2.ipynb
```

Or open the `.ipynb` files in VS Code / Cursor and run all cells.

Q2 writes figures to `figures/` in this repo. If `/cursor/stores/self/media/q2/` exists (this VM), it writes there too, using the same filenames as the Q2 report (`fig01_target_distribution.png` … `fig05_temporal_train.png`).

Seed **42**. Fit transformers on **train** only. Do not tune on test.
