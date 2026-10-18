# Submission readiness against the official brief

Checked on 18 October 2026. The brief is `reports/IDA117V_Project_Template.pdf` (IDA117V Project Template, Semester 2 2026, 100 marks). Question headings and the rubric do not use the same totals: question 3 is headed 18 but its parts sum to 22, and question 7 is headed 10 but its parts sum to 7. The rubric totals are the ones that sum to 100, and this note uses those.

This note does not say the submission is complete.

## What the brief asks for, and what is in the repository

| Brief item | Marks | Status | Evidence |
| --- | --- | --- | --- |
| 1.1 Problem, population, decision-maker, unit, inputs, classes, horizon | 4 | Partial | `reports/final_report.md` section 1 names Cape Town service requests, residents and businesses, and municipal departments. The unit is one request at logging. Classes are Resolved Fast, Resolved Slow, and Unresolved. The horizon is seven days. The cover still has no student names, student numbers, group number, lecturer, or submission date. |
| 1.2 Primary metric, service indicator, false-positive and false-negative cost | 2 | Partial | Macro-F1 is the primary metric. The report does not state a separate service-delivery success indicator, and it does not use the words false positive or false negative. |
| 1.3 Provenance, licence, coverage, limitations, POPIA, privacy, fairness, misuse | 4 | Partial | `reports/data_source.md` and report section 2 record the City URL, custom terms, coverage, and limitations. Coordinates and notification ids are withheld. POPIA is not named. |
| 2.1 Data dictionary and audit | 4 | Met | `results/tables/02_data_dictionary.csv`. Duplicates, missingness, invalid dates, and leakage fields are in the report. |
| 2.2 At least five figures | 6 | Met | `results/figures/03_01_target_distribution.png` through `results/figures/03_05_monthly_request_volume.png`, plus the duration chart `03_06`. |
| 2.3 Split before fitting; test reserved | 3 | Met | Chronological split. Encoders and the scaler are fit on the training period. The champion search does not read the locked test. |
| 2.4 Leakage-safe pipeline, including imbalance treatment | 5 | Met | `src/preprocessing.py` and the frozen pipeline. Class weights are the imbalance treatment. SMOTE is not used. |
| 3.1 Naïve baseline | 2 | Met | Most-frequent dummy. Validation macro-F1 0.273. |
| 3.2–3.3 Four substantive models: two traditional and two neural | 13 | Met | Logistic regression and histogram gradient boosting. Multilayer perceptron and the lecturer-approved embedding network. The naïve baseline is extra. |
| 3.4 Training detail and seeds | 5 | Partial | Architecture, loss, optimiser, epochs, batch size, regularisation, early stopping, and seed 42 are recorded. A compute-environment description is not in the report. |
| 3.5 Rationale for each family | 2 | Met | Report sections 6 and 7. |
| 4.1 Same held-out metrics, including PR-AUC | 4 | Partial | Validation accuracy, macro-F1, weighted-F1, recall, ROC-AUC, and PR-AUC are in `results/tables/05_split_metrics.csv` for all four substantive models. The section 8 table now includes the embedding network. Precision for every class is in `results/tables/05_per_class.csv`, not repeated as a classification report in the narrative. |
| 4.2 Classification report, confusion matrix, one common ROC figure | 5 | Partial | Validation confusion matrices exist for all four substantive models under `results/figures/05_confusion_*_validation.png`. Per-class precision, recall, F1, and support are in `results/tables/05_per_class.csv`. There is no single figure with every model's ROC curve. `results/figures/09_roc_per_class_locked_test.png` is the champion only, one-versus-rest by class, not a macro/micro overlay of all models. |
| 4.3 Comparison table, bar chart, variability, validation champion | 3 | Partial | `results/figures/05_validation_comparison.png` and the validation table. The champion is chosen on validation macro-F1. Random cross-validation means are not shown. The chronological design is the reason. Fold means exist for the champion search only. |
| 4.4 Overfitting and neural curves | 4 | Met | Train-versus-validation gaps are in the report. MLP and embedding loss and macro-F1 curves exist. The boosting validation curve exists. |
| 5.1–5.3 Optimise only the strongest validation model; one test score | 10 | Met | Eight random trials, seed 42, inside 2023–2024. The tuned refit lost on 2025 validation macro-F1 (0.470 against 0.477), so the locked pipeline is the untuned 12-iteration booster. Its one locked-test macro-F1 is 0.494. |
| 6.1 At least three one-change ablations | 5 | Met with a caveat | Geographic features removed, and class weights turned off, are real validation refits. The Phase 6 work-centre row did not fit a second model. A later validation fit did add work centre and did not replace the champion. That later fit is the third change. It is not leakage-safe, because the extract has no assignment time. |
| 6.2 McNemar against the strongest comparator on the same test cases | 5 | Not met as specified | The saved test compares the frozen booster with the multilayer perceptron: 233,776 both correct, 19,282 champion only, 21,854 comparator only, 147,121 both wrong. Statistic 160.6875. p-value 8.006e-37. Alpha 0.05. Hypotheses are stated, and the practical gap is separated from the p-value. The strongest other model on validation macro-F1 is the embedding network, at 0.480687, above the perceptron at 0.476466. The brief asks for the strongest comparator on identical test cases. The embedding network has no locked-test predictions, so that paired test has not been run. |
| 7.1–7.4 Interface | 7 | Met locally | `streamlit run app/app.py` loads `models/frozen_resolution_pipeline.joblib`. The form shows the class, probabilities, and a plain-language meaning. Invalid input is rejected. `reports/demonstration_checklist.md` has more than three scored cases and invalid cases. Screenshots are in `evidence/screenshots/`. There is no hosted demonstration link. The brief allows a local launch command instead. |
| 8.1 Recommendation, limits, fairness, when not to use the model | 2 | Met | Report section 16. The suburb and ward audit is not claimed as a disparity finding. |
| 8.2 Repository package | 3 | Partial | README, `requirements.txt`, seed 42, notebook order, source URL, data dictionary, saved pipeline, and interface code are present. The raw extract is not in git. There is no reference list for the libraries, and AI-assisted work is not labelled. |
| 8.3 Presentation and signed contribution statement | 2 | Not met | `reports/presentation_outline.md` is about 11 minutes. There is no slide file and no record that every member presented. `reports/team_contribution_template.md` has no names or signatures. The brief's presentation window is 5–17 October 2026. |

## Verified files that did not change in this check

- Frozen decision: `results/tables/06_frozen_decision.json`
- Aligned predictions: `results/tables/06_test_predictions.parquet`, `results/tables/07_paired_predictions.parquet`
- McNemar record: `results/tables/07_comparison_record.json`
- Corrected embedding curve: `results/figures/10_embedding_learning_curves.png`
- Champion ROC and precision-recall: `results/figures/09_roc_per_class_locked_test.png`, `results/figures/09_pr_per_class_locked_test.png`
- Suburb and ward audits: `results/tables/09_suburb_error_audit.csv`, `results/tables/09_ward_error_audit.csv`
- Tests: `python -m pytest tests -q` passed 80 tests on 18 October 2026, before this brief comparison. This comparison did not refit a model.

## Still required before the brief can be treated as met

1. A McNemar comparison of the frozen champion with the embedding network on the same 422,033 locked-test rows, if the marker reads "strongest comparator" as the next model on validation macro-F1. That requires generating embedding-network test predictions. Those predictions do not exist, and this check did not create them.
2. One ROC figure that shows the substantive models together, in the macro or micro one-versus-rest form the brief names.
3. An explicit service-delivery success indicator, the cost of false positives and false negatives, and a POPIA discussion in the report.
4. The cover fields: student names, student numbers, group number, lecturer, and submission date.
5. A signed contribution statement, and the presentation the brief asks every member to give.
6. A short reference list, and a label for AI-assisted work if institutional policy requires one.

Logistic regression still does not converge at the checked settings. The report does not claim that it does. No stakeholder interview was conducted. The brief does not contain an interview question; it does require a defined decision-maker and a recommendation to that stakeholder.
