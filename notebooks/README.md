# Notebooks

Template-ordered Jupyter notebooks. Run from the **repo root**.

| File | What it does |
| --- | --- |
| `q1.ipynb` | Locked Q1 counts: class mix, 7-day cut check, DummyClassifier val macro-F1 |
| `q2.ipynb` | Template 2.1 → 2.2 → 2.3 → 2.4: dictionary/audit, five train-only figures, split-before-fit, `prep.py` Pipeline |
| `q3.ipynb` | Template 3.1 → 3.5: Dummy + LR + Random Forest + MLP + tabular embedding net on the frozen split. Loads `experiments/q3_val_metrics.csv` if present, else `python -m src.train_q3`. **No test look, no champion, no Streamlit.** |

## How to run

1. `python -m pip install -r requirements-research.txt`
2. Put frozen Drive splits in `data/processed/` (`train.parquet`, `val.parquet`, `test.parquet`). See [`data/README.md`](../data/README.md).
3. From the **repo root** (so `src/prep.py` and `data/processed/` resolve):

```bash
jupyter notebook notebooks/q1.ipynb
jupyter notebook notebooks/q2.ipynb
jupyter notebook notebooks/q3.ipynb
```

Or open the `.ipynb` files in VS Code / Cursor and run all cells.

Q2 writes figures to `figures/`. Q3 writes `figures/fig_q3_*.png` and `experiments/q3_*.csv`. Seed **42**. Fit transformers on **train** only. Do not tune on test.
