# Reproducibility and submission audit

Checked on 18 October 2026. This audit does not say the assignment is complete. No stakeholder interview has been conducted. The brief does not require one. At the time of this audit the embedding network had not been given a locked-test score, and it was not selected as champion. Later the same day, Question 6.2 scored the saved checkpoint on the existing locked test. That comparison is `results/tables/13_question62_record.json`. It did not change the champion. The findings below describe the audit as it stood before that score.

## Embedding experiment

`python -m src.embedding_experiment` retrains the approved network. It deletes the locked test split before encoding. When `models/embedding_network.pt` already exists, the command scores that checkpoint and writes the new fit under `evidence/reproducibility/embedding_retrain/`. It does not replace the checkpoint unless `--overwrite-checkpoint` is passed. That flag was not used.

The saved checkpoint and the retrain both matched `results/tables/10_embedding_metrics.csv` and `results/tables/10_embedding_history.csv`. The largest absolute gap was about 1e-16, which is floating-point noise. Validation macro-F1 remained 0.480687. The champion stayed histogram gradient boosting. The record is `results/tables/12_embedding_reproduction.json`.

The corrected learning curve is `results/figures/10_embedding_learning_curves.png`. Its title is "Approved embedding network". `results/figures/05_embedding_learning_curves.png` was redrawn from the same saved history, so it no longer carries the pending title. The final report uses the corrected figure.

## Logistic regression

The reported baseline is unchanged: `make_logistic_baseline` in `src/preprocessing.py`, saga, C = 1, L2, balanced class weights, an intercept, max_iter 200, tolerance 0.001, seed 42. Its validation macro-F1 is 0.460268. It stops at iteration 200 with a convergence warning.

The training matrix has 2,616,595 rows and 1,718 columns. Stored one-hot values are 1, with zeros implicit. The weekend column has training mean about 0 and standard deviation 1, so the numeric feature is already standardised. Each of the six categorical blocks sums to 1 on every row, so the intercept and the full one-hot coding are linearly dependent. Column norms range from about 11.3 to 1,618, a ratio of about 143, because frequent levels are much more common than rare levels. Saga is the sparse multinomial solver. A dense solver would materialise a matrix of about 16.7 GiB. L2 regularisation is present. Warnings were not filtered out.

Two separate variants used the same iteration cap and were not adopted:

| Variant | Iterations | Warning | Converged | Validation macro-F1 |
| --- | ---: | --- | --- | ---: |
| No intercept | 200 | Yes | No | 0.467314 |
| Variance-scaled one-hot block | 200 | Yes | No | 0.465106 |

The earlier max_iter 1000 fit also warned and scored 0.453. It was not adopted. None of these fits read the locked test. The files are `results/tables/12_logistic_conditioning.json`, `results/tables/12_logistic_variants.csv`, and `results/tables/11_logistic_convergence.json`.

## Old learning-curve figure

`results/figures/05_embedding_learning_curves.png` was redrawn from `results/tables/10_embedding_history.csv` with the title "Approved embedding network". Notebook 05 still writes that filename on a future run. The figure displayed in the final report is `results/figures/10_embedding_learning_curves.png`.

## Notebooks 01 to 07

The copies were executed with `python scripts/execute_notebooks_isolated.py`. Outputs went to `evidence/reproducibility/notebooks/isolated/`. The original notebooks, `data/processed/service_requests_labelled.parquet`, `results/tables/06_test_predictions.parquet`, and `models/embedding_network.pt` were not replaced. Executed copies of notebooks 01 to 06 are in `evidence/reproducibility/notebooks/executed/`. Notebook 07 was not written there because execution stopped before the notebook could be saved. The log is `evidence/reproducibility/notebooks/execution_log.txt`.

| Notebook | Result |
| --- | --- |
| 01 Data acquisition and audit | Finished. Audit tables matched the saved tables. |
| 02 Exploratory data analysis | Finished. Saved tables matched. |
| 03 Preprocessing and baselines | Finished. Scores matched. Logistic fit time changed from 244.6 to 224.3 seconds. The convergence warning and iteration count 200 were the same. |
| 04 Traditional models | Finished. Validation macro-F1 matched: logistic regression 0.460268, histogram gradient boosting 0.481660. Fit times differed. |
| 05 Neural networks | The first attempt failed because the notebook imported `PENDING_EMBEDDING_NOTE`, which no longer exists. The import was changed to `APPROVED_EMBEDDING_NOTE`, and the embedding rows are now marked approved in the notebook source. The rerun finished. Every metric matched the saved `05_split_metrics.csv`. The approval label in that saved table was still `pending_lecturer_approval`; it was updated to `approved` without changing the scores. The same label was updated in `05_model_comparison.csv`, `05_experiment_config.json`, and `08_model_metrics.csv`. |
| 06 Evaluation and optimisation | Finished in the isolated directory. Search scores, candidate scores, and the locked-test summary matched apart from fit times. The isolated frozen decision names `embedding_network` as the comparator. The original `results/tables/06_frozen_decision.json` still names `multilayer_perceptron` and was not replaced. |
| 07 Ablation and McNemar | Stopped before the comparator was scored. The cell requires the frozen comparator to be the multilayer perceptron. The isolated notebook 06 had named the embedding network, because that model is now approved and its validation macro-F1 is 0.480687, above the perceptron at 0.476466 and below histogram gradient boosting at 0.481660. The embedding network was not scored on the locked test. The saved McNemar table was not replaced. |

Notebook 08 was not rerun. It does not fit a model.

## Stakeholder interview

`reports/stakeholder_interview_guide.md` still says the interview has not been conducted. The checklist is blank. No participant, date, answer, or finding was added.

## Submission evidence already present

These files were checked on disk and were not recreated:

- `results/figures/09_roc_per_class_locked_test.png`
- `results/figures/09_pr_per_class_locked_test.png`
- `results/tables/09_suburb_error_audit.csv`
- `results/tables/09_ward_error_audit.csv`
- `evidence/screenshots/01_blank_form.png`
- `evidence/screenshots/02_completed_sample.png`
- `evidence/screenshots/03_prediction.png`
- `evidence/screenshots/04_validation_error.png`
- `reports/final_report.md`
- `reports/final_report.pdf`

The report figures named in `reports/final_report.md` are present. The PDF was regenerated after this audit.

## Commands

From the repository root, with the labelled parquet in place:

```bash
python -m src.embedding_experiment
python -m src.logistic_convergence
python -m src.logistic_conditioning
python scripts/execute_notebooks_isolated.py
python scripts/export_report_pdf.py
python -m pytest tests -q
```

`python -m src.embedding_experiment` does not replace an existing checkpoint. The notebook command writes only under `evidence/reproducibility/notebooks/`. `python -m src.model_comparison` checks the saved champion and comparator predictions and writes `results/tables/07_comparison_record.json`. It does not refit either model.

## Comparator

A clean rerun of notebook 06 previously named the embedding network because that model is now approved and the old selector took the highest approved validation macro-F1 other than the champion. Notebook 07 then stopped, because it required the multilayer perceptron. The saved predictions were already aligned on `source_index` and were not the cause. The comparator rule now uses the same tie-break as the champion and excludes the embedding network, because that network is not scored on the locked test. Notebook 07 reads the frozen decision and stops if that name disagrees with the rule. The saved McNemar comparison remains the multilayer perceptron comparison.

## Unresolved

- No stakeholder interview.
- Logistic regression still does not converge at the reported settings or at the checked variants. The 200-iteration intercept model remains the reported baseline.
- The embedding network was not selected as champion. Its later locked-test score is the Question 6.2 comparison only.
