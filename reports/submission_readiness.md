# Submission readiness against the official brief

Checked on 18 October 2026. The brief is `reports/IDA117V_Project_Template.pdf` (IDA117V Project Template, Semester 2 2026, 100 marks). Question headings and the rubric do not use the same totals: question 3 is headed 18 but its parts sum to 22, and question 7 is headed 10 but its parts sum to 7. The rubric totals are the ones that sum to 100, and this note uses those.

This note does not say the submission is complete.

## What the brief asks for, and what is in the repository

| Brief item | Marks | Status | Evidence |
| --- | --- | --- | --- |
| 1.1 Problem, population, decision-maker, unit, inputs, classes, horizon | 4 | Partial | `reports/final_report.md` section 1 names Cape Town service requests, residents and businesses, and a reviewer in the municipal service department. The unit is one request at logging. Classes are Resolved Fast, Resolved Slow, and Unresolved. The horizon is seven days. The cover fields are present and blank: student names, student numbers, group number, lecturer, and submission date were not supplied. |
| 1.2 Primary metric, service indicator, false-positive and false-negative cost | 2 | Met | Macro-F1 is the primary metric. The service indicator is whether a request that will not resolve within seven days is brought to a reviewer. Section 1 states the operational cost of a false positive and a false negative for the three classes, using the champion's locked-test Resolved Slow precision 0.210 and Unresolved recall 0.418. |
| 1.3 Provenance, licence, coverage, limitations, POPIA, privacy, fairness, misuse | 4 | Met as a discussion | Section 1.1 names POPIA and covers personal information, lawful processing, minimisation, access, retention, and security. It does not claim that compliance has been certified. Fairness and misuse are in section 16. Provenance remains in section 2 and `reports/data_source.md`. |
| 2.1 Data dictionary and audit | 4 | Met | `results/tables/02_data_dictionary.csv`. Duplicates, missingness, invalid dates, and leakage fields are in the report. |
| 2.2 At least five figures | 6 | Met | `results/figures/03_01_target_distribution.png` through `results/figures/03_05_monthly_request_volume.png`, plus the duration chart `03_06`. |
| 2.3 Split before fitting; test reserved | 3 | Met | Chronological split. Encoders and the scaler are fit on the training period. The champion search does not read the locked test. |
| 2.4 Leakage-safe pipeline, including imbalance treatment | 5 | Met | `src/preprocessing.py` and the frozen pipeline. Class weights are the imbalance treatment. SMOTE is not used. |
| 3.1 Naïve baseline | 2 | Met | Most-frequent dummy. Validation macro-F1 0.273. |
| 3.2–3.3 Four substantive models: two traditional and two neural | 13 | Met | Logistic regression and histogram gradient boosting. Multilayer perceptron and the lecturer-approved embedding network. The naïve baseline is extra. |
| 3.4 Training detail and seeds | 5 | Partial | Architecture, loss, optimiser, epochs, batch size, regularisation, early stopping, and seed 42 are recorded. A compute-environment description is not in the report. |
| 3.5 Rationale for each family | 2 | Met | Report sections 6 and 7. |
| 4.1 Same held-out metrics, including PR-AUC | 4 | Partial | Validation accuracy, macro-F1, weighted-F1, recall, ROC-AUC, and PR-AUC are in `results/tables/05_split_metrics.csv` for all four substantive models. The section 8 table now includes the embedding network. Precision for every class is in `results/tables/05_per_class.csv`, not repeated as a classification report in the narrative. |
| 4.2 Classification report, confusion matrix, one common ROC figure | 5 | Met | Validation confusion matrices exist for all four substantive models under `results/figures/05_confusion_*_validation.png`. Per-class precision, recall, F1, and support are in `results/tables/05_per_class.csv`. The common locked-test figure is `results/figures/13_roc_all_models_locked_test.png`. It has one macro one-versus-rest curve for the dummy, logistic regression, histogram gradient boosting, the multilayer perceptron, and the embedding network. Class AUCs, macro AUC, and micro AUC are in `results/tables/13_roc_all_models_locked_test.csv`. No curve was skipped. The champion-only class curves remain in `results/figures/09_roc_per_class_locked_test.png`. |
| 4.3 Comparison table, bar chart, variability, validation champion | 3 | Partial | `results/figures/05_validation_comparison.png` and the validation table. The champion is chosen on validation macro-F1. Random cross-validation means are not shown. The chronological design is the reason. Fold means exist for the champion search only. |
| 4.4 Overfitting and neural curves | 4 | Met | Train-versus-validation gaps are in the report. MLP and embedding loss and macro-F1 curves exist. The boosting validation curve exists. |
| 5.1–5.3 Optimise only the strongest validation model; one test score | 10 | Met | Eight random trials, seed 42, inside 2023–2024. The tuned refit lost on 2025 validation macro-F1 (0.470 against 0.477), so the locked pipeline is the untuned 12-iteration booster. Its one locked-test macro-F1 is 0.494. |
| 6.1 At least three one-change ablations | 5 | Met with a caveat | Geographic features removed, and class weights turned off, are real validation refits. The Phase 6 work-centre row did not fit a second model. A later validation fit did add work centre and did not replace the champion. That later fit is the third change. It is not leakage-safe, because the extract has no assignment time. |
| 6.2 McNemar against the strongest comparator on the same test cases | 5 | Met | The Question 6.2 comparator is the embedding network, validation macro-F1 0.480687, above the perceptron at 0.476466 and below histogram gradient boosting at 0.481660. On the same 422,033 locked-test rows: 229,320 both correct, 23,738 champion only, 18,625 embedding only, 150,350 both wrong. Statistic 616.8719. p-value 3.582e-136. Alpha 0.05. The record is `results/tables/13_question62_record.json`. The champion was not changed. The perceptron comparison, statistic 160.6875, remains in `results/tables/07_mcnemar_test.csv` as a historical result. |
| 7.1–7.4 Interface | 7 | Met locally | `streamlit run app/app.py` loads `models/frozen_resolution_pipeline.joblib`. The form shows the class, probabilities, and a plain-language meaning. Invalid input is rejected. `reports/demonstration_checklist.md` has more than three scored cases and invalid cases. Screenshots are in `evidence/screenshots/`. There is no hosted demonstration link. The brief allows a local launch command instead. |
| 8.1 Recommendation, limits, fairness, when not to use the model | 2 | Met | Report section 16. The suburb and ward audit is not claimed as a disparity finding. |
| 8.2 Repository package | 3 | Partial | README, `requirements.txt`, seed 42, notebook order, source URL, data dictionary, saved pipeline, and interface code are present. The raw extract is not in git. The report now has a reference list and an AI-assistance disclosure based on the git history. A module-specific signed AI form was not in the brief or the repository. |
| 8.3 Presentation and signed contribution statement | 2 | Not met | The brief asks for a 10–12 minute presentation, participation by every member, an interface demonstration, a defence, and a statement signed by all members, during 5–17 October 2026. `reports/presentation_outline.md` is the speaking plan. There is no slide file, attendance record, recording, or defence outcome. `reports/contribution_statement.md` names the only human git author and leaves the signature cells empty. |

## Verified files that did not change in this check

- Frozen decision: `results/tables/06_frozen_decision.json`
- Aligned predictions: `results/tables/06_test_predictions.parquet`, `results/tables/07_paired_predictions.parquet`
- McNemar record: `results/tables/07_comparison_record.json`
- Corrected embedding curve: `results/figures/10_embedding_learning_curves.png`
- Champion ROC and precision-recall: `results/figures/09_roc_per_class_locked_test.png`, `results/figures/09_pr_per_class_locked_test.png`
- Suburb and ward audits: `results/tables/09_suburb_error_audit.csv`, `results/tables/09_ward_error_audit.csv`
- Tests: `python -m pytest tests -q` passed 80 tests on 18 October 2026, before this brief comparison. This comparison did not refit a model.

## Still required before the brief can be treated as met

1. Student numbers, the group number, the lecturer, the submission date, and any group members who are not named in git. The cover records only Thabang Mokoena, from the 15 October 2026 commits.
2. Signatures on `reports/contribution_statement.md`. The brief requires them. The cells are empty because none were supplied. This file is not a signed declaration.
3. Evidence that the presentation was given and that a defence took place. The brief’s window is 5–17 October 2026. The outline exists. Attendance does not.
4. A compute-environment description is still thin, and random cross-validation means are not shown for every model. The chronological split is the design that was used.

Question 6.2, the common ROC figure, the false-positive and false-negative discussion, and the POPIA discussion are now in the report. Logistic regression still does not converge at the checked settings. The report does not claim that it does. No stakeholder interview was conducted. The brief does not contain an interview question.
