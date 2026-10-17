# Experiment log

This log lists the experiments that were actually run and the files that hold their settings and scores. It does not add a new model. Full-precision metrics are in `results/tables/08_model_metrics.csv`.

## Environment

- Python 3.12.3, as used to execute the notebooks.
- Direct pins are in `requirements.txt`, including pandas 3.0.6, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, torch 2.14.1+cpu, and streamlit 1.66.0.
- Random seed 42 for samplers, logistic regression, boosting, and the neural networks. The chronological split does not use the seed.
- No GPU was used.

## Data and split

- Source file, gitignored: `data/raw/c3_notifications.parquet`, 5,372,466 rows, downloaded from the feature layer on 16 October 2026.
- Analysis file, gitignored: `data/processed/service_requests_labelled.parquet`.
- Exact business duplicates removed: 779,663. Repeated notification ids that are not exact copies were kept.
- Labelled rows: 4,576,802. Unlabelled: 16,001.
- Train: 2023-01-01 to 2024-12-31, 2,616,595 rows.
- Validation: 2025-02-01 to 2025-12-31, 936,469 rows. January 2025 has no rows.
- Locked test: 2026-01-01 to 2026-04-30, 422,033 rows.
- Labelled rows after 30 April 2026 left out of every model split: 601,705.
- Counts: `results/tables/04_split_class_counts.csv` and `results/tables/04_split_exclusions.csv`.

## Executed experiments

| ID | What was fit | Where the decision was made | Score file | Locked test |
| --- | --- | --- | --- | --- |
| E1 | Most-frequent dummy | Training majority class | `results/tables/04_baseline_summary.csv` | Scored once in notebook 03 |
| E2 | Logistic regression, saga, C = 1, L2, balanced weights, max_iter 200, tol 0.001, seed 42 | Fixed before the fit. Did not converge. | `results/tables/04_baseline_summary.csv` and `results/tables/05_split_metrics.csv` | Scored once in notebook 03. Phase 5 repeated the validation score and did not rescore the test. |
| E3 | Histogram gradient boosting, learning rate 0.1, 31 leaves, min_samples_leaf 50, L2 1, max_iter 40, early stopping on validation macro-F1 | Phase 5 validation curve. Predictions use iteration 2. | `results/tables/05_split_metrics.csv`, `results/tables/05_boosting_curve.csv` | Not scored in Phase 5 |
| E4 | Multilayer perceptron, 1718-128-64-3, dropout 0.3, AdamW, batch 8192, patience 3 | Restored epoch 3 of 6 | `results/tables/05_mlp_history.csv` | Not scored in Phase 5 |
| E5 | Embedding network, embedding size 16, hidden size 64 | Restored epoch 4 of 7 | `results/tables/05_embedding_history.csv` | Not scored. Pending lecturer approval. Not used to choose the champion. |
| E6 | Champion search: untuned settings plus 8 `ParameterSampler` draws, seed 42, three expanding folds inside 2023–2024, score mean macro-F1 | Training folds only. Trial 5 had the highest mean, 0.544. | `results/tables/06_search_summary.csv`, `results/tables/06_search_folds.csv` | Not used |
| E7 | Untuned and tuned boosting refits, early stopping off, iteration cap = median training-fold iteration | 2025 validation macro-F1. Tuned 0.470, untuned 0.477. Frozen choice: untuned, 12 iterations. Saved before the test was opened. | `results/tables/06_candidate_metrics.csv`, `results/tables/06_frozen_decision.json` | Not used for the choice |
| E8 | Frozen boosting model, one locked-test score | No change after the score | `results/tables/06_locked_test_summary.csv`, `results/tables/06_test_predictions.parquet` | Scored once |
| E9 | Ablation: drop suburb, ward, and subcouncil. Same frozen hyperparameters. | Validation only | `results/tables/07_ablation_summary.csv` | Not scored |
| E10 | Ablation: class_weight left unset. Same features and hyperparameters. | Validation only | `results/tables/07_ablation_summary.csv` | Not scored |
| E11 | Multilayer perceptron retrained with the Phase 5 procedure, then scored once on the locked test | Comparator named from the Phase 5 validation table before the test scores were read. Retrained validation macro-F1 matched Phase 5. | `results/tables/07_comparator_retrained_validation.csv`, `results/tables/07_comparator_locked_test_summary.csv` | Scored once |
| E12 | McNemar comparison of E8 and E11 on the same 422,033 rows | Two-sided test of paired error rates, alpha 0.05 | `results/tables/07_mcnemar_test.csv`, `results/tables/07_paired_predictions.parquet` | Uses the two saved prediction columns |
| E13 | Export of the frozen pipeline and a check that it reproduces the E8 labels | No new fit inside notebook 08 | `models/frozen_resolution_pipeline.joblib`, `results/tables/08_export_check.json` | Comparison only. 422,033 of 422,033 labels matched. |

## Experiments that were not run

- The work-centre ablation did not fit a second model. Work Center was already excluded. The recorded change is zero by construction. See the `remove_work_centre` row in `results/tables/07_ablation_summary.csv`.
- The embedding network is not an approved experiment. A lecturer has not approved that architecture in this repository.
- No search was run for logistic regression, the multilayer perceptron, or the embedding network.
- SMOTE was not used.
- A dense random forest was not fit. The one-hot matrix is too large for a dense forest in this environment.
- Per-class ROC and precision-recall curves were not saved as figures. Macro ROC-AUC and macro average precision were computed where `predict_proba` was available.
- No suburb-level or ward-level comparison of model error was run.
- No stakeholder interview or City review was run.
- Notebooks 01 to 07 were not executed a second time from an empty environment during Phase 8. Their saved cell outputs are the source of the scores above. Notebook 08 was executed in Phase 8 and does not fit a model.

## Champion rule

Phase 5 named histogram gradient boosting because it had the highest validation macro-F1 among approved models (0.482). Phase 6 did not reopen that family choice. Inside the family, the tuned draw was kept only if its 2025 validation macro-F1 beat the untuned refit. It did not. The frozen pipeline is the untuned setting. The Phase 5 figure of 0.482 used iteration 2, chosen on the 2025 curve. The frozen refit uses 12 iterations, the median from the training folds, and its 2025 macro-F1 is 0.477.
