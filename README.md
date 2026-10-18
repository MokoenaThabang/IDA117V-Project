# Cape Town service-request resolution classifier

IDA117V project. At the moment a City of Cape Town municipal service request is logged, estimate which resolution outcome is likely:

- Resolved Fast
- Resolved Slow
- Unresolved

The intended users are municipal service departments that need to see requests which may need earlier attention. The estimate is a review aid. It must not be used to deny or reduce service.

The draft report is [`reports/final_report.md`](reports/final_report.md). It is assembled from the saved notebook outputs. This repository does not declare the project complete. Lecturer approval of the embedding network has been recorded and that model is not the champion. No stakeholder interview has been recorded. The readiness notes are in [`reports/submission_readiness.md`](reports/submission_readiness.md).

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
| `models/` | Saved pipelines. `models/frozen_resolution_pipeline.joblib` is the file the app loads. Other fitted files stay gitignored. |
| `app/` | Streamlit interface. It loads the saved pipeline and does not retrain. |
| `results/figures/` | Saved plots. |
| `results/tables/` | Saved metric and audit tables. |
| `reports/` | Dataset source notes and later written findings. |
| `tests/` | Small checks that the package imports and paths resolve. |

`src/config.py` holds paths, the random seed (`42`), the three class names, and the optional day thresholds.

## Notebook order

Run them in this order, from the project folder, after the raw extract is in `data/raw/`. Each notebook finds the project root by looking for `src/config.py` in the working directory or a parent. Do not hardcode a machine path. Notebook 02 reads the labelled file from notebook 01 and does not train a model. Its written summary is [`reports/phase3_eda_findings.md`](reports/phase3_eda_findings.md). Notebook 03 fits a most-frequent dummy and a logistic regression on the training period only. Its written summary is [`reports/phase4_baselines.md`](reports/phase4_baselines.md). Notebooks 04 and 05 fit the traditional and neural models on those same partitions. Their written summary is [`reports/phase5_models.md`](reports/phase5_models.md). Notebook 05 chooses a provisional champion from validation macro-F1 and does not score the locked test. Notebooks 06 and 07 tune that champion on chronological folds inside the training period, score the locked test once, and then run the ablations and McNemar's test. Their written summary is [`reports/phase6_evaluation.md`](reports/phase6_evaluation.md). Notebook 08 checks the saved pipeline and does not fit a model.

Notebook 06 records a historical comparator before the locked test is opened. That recorded field is the multilayer perceptron, because the selector used for the frozen decision excluded the embedding network. Question 6.2 uses a separate comparison. The strongest approved model other than the champion, by Phase 5 validation macro-F1, is the embedding network. `python -m src.locked_test_comparison` loads the saved checkpoint, checks the 2025 validation macro-F1, scores the existing locked test once, and writes `results/tables/13_question62_record.json`. It does not change the champion. The earlier perceptron table remains `results/tables/07_mcnemar_test.csv`. `python -m src.model_comparison` still checks that historical record.

Notebooks 01 to 07 already contain executed outputs. They were not run again from an empty environment while the Phase 8 draft was written. Notebook 08 was executed.

| Step | Notebook | What it will cover | Status |
| --- | --- | --- | --- |
| 1 | `notebooks/01_data_acquisition_and_audit.ipynb` | Source, raw file, column audit | Created |
| 2 | `notebooks/02_exploratory_data_analysis.ipynb` | EDA and required figures | Created |
| 3 | `notebooks/03_preprocessing_and_baselines.ipynb` | Leakage-safe features, dummy and logistic baselines | Created |
| 4 | `notebooks/04_traditional_machine_learning.ipynb` | Logistic regression and histogram gradient boosting | Created |
| 5 | `notebooks/05_neural_networks.ipynb` | Multilayer perceptron and the approved embedding network. The embedding network does not win on validation macro-F1. Its locked-test score is the Question 6.2 comparison only. | Created |
| 6 | `notebooks/06_evaluation_and_optimisation.ipynb` | Champion tuning and one locked-test evaluation | Created |
| 7 | `notebooks/07_ablation_and_statistical_testing.ipynb` | Ablations and McNemar's test | Created |
| 8 | `notebooks/08_final_model_export.ipynb` | Check the saved pipeline. It does not refit. | Created |

## Label rules

Notebook 01 builds the class from **Created On Date** and **Completed Date**. The provisional cut is seven calendar days. **Completed Date** and **Changed on** are excluded from model features. Open requests younger than the observation window are left unlabelled. See `src/preprocessing.py`.

## Reproducibility

- Python 3.12 was used to pin the versions in `requirements.txt`, including SciPy 1.18.1 because `src/statistics.py` imports it.
- `RANDOM_SEED` is 42.
- Raw data is not rewritten.
- Preprocessing that learns from data (imputation, encoding, scaling, rare-category grouping) is fit on the training period only. Validation and test rows are not resampled.
- Class imbalance in the Phase 5 models is handled with balanced class weights. SMOTE is not used.
- The Phase 5 provisional champion is chosen from validation macro-F1. The locked test is not scored in that phase.
- Phase 6 tunes only that champion. The search is chronological cross-validation on the training period and optimises macro-F1. The frozen pipeline is scored once on the locked test and is not changed afterwards.
- Work Center stays out of the delivered model. A later sensitivity fit compared the frozen booster with and without work centre on the 2025 validation period only. It did not replace the champion.

## Prediction interface

The app loads `models/frozen_resolution_pipeline.joblib`. It does not fit a model and it does not rebuild the encoder. The file is a refit of the frozen Phase 6 settings on the training period only. Those settings are not changed after the locked test.

From the project folder, with the virtual environment activated:

```bash
streamlit run app/app.py
```

On Windows, use the same command after `.venv\Scripts\activate`.

The form asks for complaint type, suburb, ward, subcouncil, and the request creation date. Work centre is not requested. The creation date is used only to derive the weekday, the month, and the weekend flag. Dropdown values are categories kept from the training period. A value outside that list is grouped with rare or unseen levels. The demonstration sample is a set of frequent training categories, not a logged request and not an unresolved case.

`requirements.txt` already pins Streamlit and joblib. Install it with the setup commands above before launching the app.

To write the pipeline file again from the labelled extract and the frozen decision in `results/tables/06_frozen_decision.json`:

```bash
python -m app.export_frozen_pipeline --force
```

That command needs `data/processed/service_requests_labelled.parquet`. The app itself does not run it. The export checks that the refit matches the locked-test predictions already saved by notebook 06, and it stops without replacing the file if they differ.

## Deliverables

| File | What it holds |
| --- | --- |
| [`reports/final_report.md`](reports/final_report.md) | Report source. The PDF is `reports/final_report.pdf`, written by `python scripts/export_report_pdf.py`. |
| [`reports/experiment_log.md`](reports/experiment_log.md) | Experiments that were run, and experiments that were not. |
| [`results/tables/08_model_metrics.csv`](results/tables/08_model_metrics.csv) | Model metrics copied from the phase tables. |
| [`results/tables/07_paired_predictions.parquet`](results/tables/07_paired_predictions.parquet) | Historical paired locked-test predictions for the perceptron comparison. No notification ids. |
| [`results/tables/13_question62_record.json`](results/tables/13_question62_record.json) | Question 6.2 McNemar record: frozen champion against the embedding network. |
| [`reports/demonstration_checklist.md`](reports/demonstration_checklist.md) | Demonstration cases run on the saved pipeline. |
| [`reports/presentation_outline.md`](reports/presentation_outline.md) | 11-minute presentation outline. |
| [`reports/contribution_statement.md`](reports/contribution_statement.md) | Contribution statement. Work rows are filled from the repository. Signature fields are blank. |
| [`reports/team_contribution_template.md`](reports/team_contribution_template.md) | Earlier blank role table. Use the contribution statement for the signed page. |

Plots and confusion matrices are in `results/figures/`. The City extract stays in `data/raw/` and `data/processed/`, which are gitignored. Do not commit those files or a file that contains notification ids, coordinates, or personal contact details. The published layer does not include a name, phone number, email address, or street address.
