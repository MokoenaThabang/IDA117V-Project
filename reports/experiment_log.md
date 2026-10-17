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
| E5 | Embedding network, embedding size 16, hidden size 64. Lecturer approval was later recorded. The architecture was not changed, and no approval date was supplied. | Restored epoch 4 of 7. Validation macro-F1 0.481, below histogram gradient boosting at 0.482, so it does not become the champion. | `results/tables/05_embedding_history.csv`, `results/tables/05_split_metrics.csv` | Not scored. The champion rule uses validation macro-F1. |
| E6 | Champion search: untuned settings plus 8 `ParameterSampler` draws, seed 42, three expanding folds inside 2023–2024, score mean macro-F1 | Training folds only. Trial 5 had the highest mean, 0.544. | `results/tables/06_search_summary.csv`, `results/tables/06_search_folds.csv` | Not used |
| E7 | Untuned and tuned boosting refits, early stopping off, iteration cap = median training-fold iteration | 2025 validation macro-F1. Tuned 0.470, untuned 0.477. Frozen choice: untuned, 12 iterations. Saved before the test was opened. | `results/tables/06_candidate_metrics.csv`, `results/tables/06_frozen_decision.json` | Not used for the choice |
| E8 | Frozen boosting model, one locked-test score | No change after the score | `results/tables/06_locked_test_summary.csv`, `results/tables/06_test_predictions.parquet` | Scored once |
| E9 | Ablation: drop suburb, ward, and subcouncil. Same frozen hyperparameters. | Validation only | `results/tables/07_ablation_summary.csv` | Not scored |
| E10 | Ablation: class_weight left unset. Same features and hyperparameters. | Validation only | `results/tables/07_ablation_summary.csv` | Not scored |
| E11 | Multilayer perceptron retrained with the Phase 5 procedure, then scored once on the locked test | Comparator named from the Phase 5 validation table before the test scores were read. Retrained validation macro-F1 matched Phase 5. | `results/tables/07_comparator_retrained_validation.csv`, `results/tables/07_comparator_locked_test_summary.csv` | Scored once |
| E12 | McNemar comparison of E8 and E11 on the same 422,033 rows | Two-sided test of paired error rates, alpha 0.05 | `results/tables/07_mcnemar_test.csv`, `results/tables/07_paired_predictions.parquet` | Uses the two saved prediction columns |
| E13 | Export of the frozen pipeline and a check that it reproduces the E8 labels | No new fit inside notebook 08 | `models/frozen_resolution_pipeline.joblib`, `results/tables/08_export_check.json` | Comparison only. 422,033 of 422,033 labels matched. |
| E14 | Frozen hyperparameters with and without Work Center. The without-work-centre refit matched the saved champion validation macro-F1. | 2025 validation only. The champion file was not replaced. Work Center has no assignment time, so the higher with-work-centre score is not a logging-time result. | `results/tables/09_work_centre_comparison.csv`, `results/tables/09_work_centre_per_class.csv`, `results/tables/09_work_centre_config.json` | Not scored |
| E15 | No new fit. One-vs-rest ROC and precision-recall curves from the E8 probabilities. | No model choice. Macro ROC-AUC and macro average precision were checked against `06_locked_test_summary.csv` and left unchanged. | `results/tables/09_per_class_ranking_locked_test.csv`, `results/figures/09_roc_per_class_locked_test.png`, `results/figures/09_pr_per_class_locked_test.png` | Uses the saved probabilities |
| E16 | No new fit. Suburb and ward error tables from the E8 predictions joined back to the labelled file by the saved row index. | No model choice. Class metrics require a group of at least 200 rows and a class of at least 30 rows. | `results/tables/09_suburb_error_audit.csv`, `results/tables/09_ward_error_audit.csv`, `results/tables/09_geographic_audit_config.json` | Uses the saved predictions |
| E17 | Approved embedding network refit. Same architecture, seed, and training-only encoder as E5. Work centre excluded. | Validation macro-F1 matched E5 at 0.480687. Histogram gradient boosting remains higher at 0.481660, so the champion is unchanged. The refit is not a second vote. | `results/tables/10_embedding_metrics.csv`, `results/tables/10_embedding_per_class.csv`, `results/tables/10_embedding_champion_decision.json`, `models/embedding_network.pt` | Not scored |

## Experiments that were not run

- The Phase 6 `remove_work_centre` row still did not fit a second model. E14 is the fitted comparison. E14 does not make Work Center a logging-time feature.
- No search was run for logistic regression, the multilayer perceptron, or the embedding network. Approval recorded the existing embedding architecture. It did not add a search.
- SMOTE was not used.
- A dense random forest was not fit. The one-hot matrix is too large for a dense forest in this environment.
- No stakeholder interview or City review was run.
- Notebooks 01 to 07 were not executed a second time from an empty environment during Phase 8. Their saved cell outputs are the source of the scores above. Notebook 08 was executed in Phase 8 and does not fit a model.

## Champion rule

Phase 5 named histogram gradient boosting because it had the highest validation macro-F1 among approved models (0.482). Phase 6 did not reopen that family choice. Inside the family, the tuned draw was kept only if its 2025 validation macro-F1 beat the untuned refit. It did not. The frozen pipeline is the untuned setting. The Phase 5 figure of 0.482 used iteration 2, chosen on the 2025 curve. The frozen refit uses 12 iterations, the median from the training folds, and its 2025 macro-F1 is 0.477.
