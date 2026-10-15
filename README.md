# IDA117V — Cape Town C3 resolution classifier

Predict **Resolved Fast / Resolved Slow / Unresolved** for a City of Cape Town SAP C3 **repair/fault** request **at logging**.

Lecturer **approved** this municipal Open Data source. Written/code due **20 October 2026**. Presentation slot **not yet announced** (do not treat the template’s 5–17 Oct window as the slot). Team names: add later; do not invent.

- Dataset: https://odp-cctegis.opendata.arcgis.com/datasets/90711027d33940b5a06e96ad8a8f7ede_0/explore
- Splits (preferred): https://drive.google.com/drive/folders/1ca8M2Gjq-sbe7I8wrt_WUyBm8523PwCq
- Seed: **42** (`src/prep.py`)
- Fast/Slow cut: **7 days** (empirical ~75th percentile of resolved durations)
- Main split: stratified random **70/15/15** on `target`. Time split = robustness only.
- UI later: **Streamlit** (a notebook cell is not the UI)

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Execution order

1. `python -m pip install -r requirements.txt`
2. Obtain modelling parquets — see [`data/README.md`](data/README.md) (Drive first; portal paging is fallback). Put `train.parquet`, `val.parquet`, `test.parquet` in `data/processed/`.
3. `python -m src.eda` — five figures from **train only** → `figures/`
4. `python -m src.baseline` — `DummyClassifier` most-frequent **and** stratified. Fits `prep.py` on **train only**. Logs val metrics; one locked test pass. Writes `models/dummy_most_frequent.joblib` and `experiments/experiment_log.csv`.
5. Jupyter (template order): [`notebooks/q1.ipynb`](notebooks/q1.ipynb), [`notebooks/q2.ipynb`](notebooks/q2.ipynb), then [`notebooks/q3.ipynb`](notebooks/q3.ipynb) (Dummy + LR + Random Forest + MLP + embedding net). How to run: [`notebooks/README.md`](notebooks/README.md). Or `python -m src.train_q3`.
6. Later: tune **only** the validation winner (Q5). **Do not tune on test.** Streamlit is Q7, not Q3.

## Leakage-safe pipeline

`src/prep.py` is the source of truth. Features: complaint type, department, suburb_group, ward, sub-council, created date. Banned: completed date, changed-on, notification IDs/text, resolution_days, raw suburb, work centre.

## Counts (Drive files, 15 Oct 2026)

| Split | n | Fast | Slow | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,120,469 | 1,429,124 | 455,121 | 236,224 |
| Val | 454,386 | 306,241 | 97,526 | 50,619 |
| Test (locked) | 454,387 | 306,241 | 97,526 | 50,620 |
| **Total** | **3,029,242** | | | |

Raw team extract: 5,199,661 rows (8 Oct 2026).

## Layout

- `notebooks/` — Q1–Q3 Jupyter notebooks
- `src/` — `prep.py`, EDA, Dummy, Q3 training
- `data/` — obtain-instructions, licence, sample (raw/processed parquets are gitignored)
- `experiments/` — dummy + Q3 val logs
- `figures/` — train-only EDA and Q3 val plots

## Docs

- [Concept notes](docs/concept-notes.md)
- [Dataset card](docs/dataset-card.md)
- [Data dictionary](docs/data-dictionary.md)
- [Licence text](data/licence.txt)
- UI dropdowns: `data/app_choices.json`

This PR trains Dummy + LR + Random Forest + MLP + the embedding net on the frozen split. It does **not** tune a champion, score a new test pass for those four, or ship Streamlit.
