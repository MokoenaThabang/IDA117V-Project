# Cape Town service-request resolution classifier

IDA117V project. At the moment a City of Cape Town municipal service request is logged, estimate which resolution outcome is likely:

- Resolved Fast
- Resolved Slow
- Unresolved

The intended users are municipal service departments that need to see requests which may need earlier attention. Phase 1 prepares the repository, records the dataset source, and audits the extract if it is present. It does not train a model.

## Dataset

The project plan names **SAP C3 Notifications** on the **City of Cape Town Open Data Portal**, covering service requests from **2023 until 18 September 2026**.

The dataset page for this project is:

https://odp-cctegis.opendata.arcgis.com/datasets/90711027d33940b5a06e96ad8a8f7ede_0/explore

The published title is **Service Requests 2023 until 06 October 2026 (Deprecated)**. The City marks the layer as frozen. Created-on dates in the feature service run from 1 January 2023 through 6 October 2026, which is later than the plan's end date. Field names, the licence, and the local-copy note are in [`reports/data_source.md`](reports/data_source.md). A gitignored parquet copy can be built with `python -m src.fetch_c3` when `data/raw/` is empty. Notebook 01 loads that file, or a CSV or Excel file placed in `data/raw/`, and writes `data/processed/service_requests_labelled.parquet`. Do not commit the raw extract. The repository MIT licence covers the code only, not the City dataset.

## Setup (Windows)

From the project folder (the folder that contains this README):

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`py -m venv .venv` is the usual Windows command. If `py` is not available, use `python -m venv .venv` instead.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned neural-network framework is **PyTorch** (CPU build). `requirements.txt` points at the PyTorch CPU index so a Linux install does not download CUDA libraries. No model is trained by the install.

Optional path and threshold overrides live in [`.env.example`](.env.example). Copy it to `.env` only if you need them. Leave the day thresholds blank until a later notebook chooses them from training data.

## How to open the notebooks

JupyterLab is the interface installed by `requirements.txt`. Start it from the project folder:

```bat
jupyter lab
```

Then open `notebooks/01_data_acquisition_and_audit.ipynb`. After that notebook has written the labelled file, open `notebooks/02_exploratory_data_analysis.ipynb`.

The classic `jupyter notebook` command is a separate package and is not pinned here. JupyterLab opens and runs the same `.ipynb` files. Use JupyterLab for this project.

Run each notebook from top to bottom with the project folder as the working directory. Notebooks do not rely on variables left in memory by another notebook.

Check that the source modules import and the folders resolve:

```bat
python -m unittest tests.test_smoke
```

If no raw file is present, this check is still expected to pass. Asking the loader to read data will print manual download steps and exit with status 1:

```bat
python -m src.data_loader
```

## Folder guide

| Path | Purpose |
| --- | --- |
| `data/raw/` | Original download only. Gitignored. Not modified by code. |
| `data/interim/` | Working extracts after cleaning. Gitignored. |
| `data/processed/` | Model-ready tables written by later notebooks. Gitignored. |
| `notebooks/` | Numbered analysis notebooks, in the order below. |
| `src/` | Reusable loading, label, metric, and figure code. |
| `models/` | Saved pipelines from a later phase. Fitted files are gitignored. |
| `app/` | Prediction interface in a later phase. It will load a saved pipeline and will not retrain. |
| `results/figures/` | Saved plots. |
| `results/tables/` | Saved metric and audit tables. |
| `reports/` | Dataset source notes and later written findings. |
| `tests/` | Small checks that the package imports and paths resolve. |

`src/config.py` holds paths, the random seed (`42`), the three class names, and the optional day thresholds.

## Notebook order

Run them in this order once each file exists. Notebooks 1 and 2 are in the repository. Notebook 2 reads the labelled file from notebook 1 and does not train a model. Its written summary is [`reports/phase3_eda_findings.md`](reports/phase3_eda_findings.md).

| Step | Notebook | What it will cover | Status |
| --- | --- | --- | --- |
| 1 | `notebooks/01_data_acquisition_and_audit.ipynb` | Source, raw file, column audit | Created |
| 2 | `notebooks/02_exploratory_data_analysis.ipynb` | EDA and required figures | Created |
| 3 | `notebooks/03_preprocessing_and_baselines.ipynb` | Cleaning, features, dummy baseline | Not created |
| 4 | `notebooks/04_traditional_machine_learning.ipynb` | Logistic regression and a tree model | Not created |
| 5 | `notebooks/05_neural_networks.ipynb` | Neural-network models in PyTorch | Not created |
| 6 | `notebooks/06_evaluation_and_optimisation.ipynb` | Comparison, overfitting, champion tuning | Not created |
| 7 | `notebooks/07_ablation_and_statistical_testing.ipynb` | Ablations and McNemar's test | Not created |
| 8 | `notebooks/08_final_model_export.ipynb` | Save the pipeline used by the app | Not created |

## Label rules

Notebook 01 builds the class from **Created On Date** and **Completed Date**. The provisional cut is seven calendar days. **Completed Date** and **Changed on** are excluded from model features. Open requests younger than the observation window are left unlabelled. See `src/preprocessing.py`.

## Reproducibility

- Python 3.12 was used to pin the versions in `requirements.txt`.
- `RANDOM_SEED` is 42.
- Raw data is not rewritten.
- Preprocessing that learns from data (imputation, encoding, scaling, feature selection, resampling) must be fit on training data only. That fitting is not part of Phase 1.
