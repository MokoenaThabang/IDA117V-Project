# Cape Town service-request resolution classifier

Draft report for IDA117V. The numbers come from the executed notebooks and the tables in `results/tables/`. This draft does not say the project is complete. The last section lists what was not run and what still needs a decision.

A PDF was not produced in this environment. Pandoc is not installed. From a machine that has Pandoc and a PDF engine, export with:

```bash
pandoc reports/final_report.md -o reports/final_report.pdf --resource-path=reports
```

The figure paths are relative to this file.

## 1. Problem definition

At the moment a municipal service request is logged, the project estimates which of three resolution outcomes is likely: Resolved Fast, Resolved Slow, or Unresolved. The estimate uses only information that can be treated as known at logging.

The requests are City of Cape Town SAP C3 notifications. The people affected are the residents and businesses whose requests are in that system, and the municipal service departments that receive them. No stakeholder interview was conducted, so this report does not claim a department finding or a resident finding. The intended benefit, as stated for the project, is to show requests that may need earlier human attention.

The estimate is not a service decision. It must not be used to deny or reduce service to an area.

## 2. Dataset provenance, licence, and dataset card

The dataset card is the measured description of the extract. Provenance details that were read from the portal on 16 October 2026 are in `reports/data_source.md`.

| Item | Recorded value |
| --- | --- |
| Owner | City of Cape Town |
| Project name | SAP C3 Notifications |
| Published title | Service Requests 2023 until 06 October 2026 (Deprecated) |
| Page | https://odp-cctegis.opendata.arcgis.com/datasets/90711027d33940b5a06e96ad8a8f7ede_0/explore |
| Access | Public feature service |
| Licence | City of Cape Town Open Data terms of use. Custom. Not a Creative Commons licence. The City provides the data as is and says reliance is at the user's risk. |
| Plan period | 2023 until 18 September 2026 |
| Created-on dates in the file | 1 January 2023 through 6 October 2026 |
| Rows in the feature layer and the local raw file | 5,372,466 |
| Local raw file | `data/raw/c3_notifications.parquet`, gitignored, saved from the feature layer on 16 October 2026 because the Hub CSV was still generating |
| Publisher status | Deprecated and frozen as of 6 October 2026. Not connected to the live system. An open request may have been completed after the extract was taken. |
| Code licence | MIT for the repository code only. It does not cover the City data. |

Coverage is city service requests in the published layer, not a survey sample. The feature-service name still says "until 20 May 2026", which is older than the maximum created date. The item description still contains unfilled phrases for a retirement month and a contact address, and a historic-year list that does not match the 2023 start of `Created_On_Date`.

Limitations that follow from the source: the layer is frozen; completion and changed-on dates extend past the created-on window, including completions in the year 2000; coordinates are often missing or outside a local metre envelope and are not model inputs; the raw extract is not stored in git.

## 3. Data audit and exploratory figures

Notebook 01 audited the raw extract. Notebook 02 described the labelled file and did not train a model. The five figures required for the exploratory phase are Figures 1 to 5. Figure 6 is the additional duration chart saved by the same notebook.

After the audit, 779,663 exact duplicate business rows were removed. 4,592,803 rows remain in the analysis file. 4,576,802 are labelled. 309,758 rows still share a notification id with a non-identical row; those rows were kept. Complaint type, work centre, and Created On Date have no missing values in the analysis file. Subcouncil is missing on 4.38% of rows, ward on 1.78%, and suburb on 0.98%. Suburb is the placeholder `#` on 365,919 rows (7.97%). Subcouncil is `Not assigned` on 1,622,881 rows (35.34%). There are 44,905 distinct suburb values and 2,076 work centres.

### Figure 1. Target distribution

![Target distribution](../results/figures/03_01_target_distribution.png)

Of the labelled requests, 2,969,242 (64.9%) are Resolved Fast, 937,145 (20.5%) are Resolved Slow, and 670,415 (14.6%) are Unresolved. Another 16,001 rows have no label. Accuracy alone would reward a prediction of Resolved Fast. Macro-F1 is the primary metric because it gives the three classes equal weight.

### Figure 2. Missingness and data quality

![Missingness and data quality](../results/figures/03_02_missingness_and_quality.png)

A blank completion date is usually an open request, not a value to impute into a predictor. 670,415 blanks are open requests old enough to be labelled Unresolved. 15,981 blanks are still inside the seven-day window and stay unlabelled. 20 rows have a completion date before the creation date and are excluded. Stored placeholders are larger than several of the genuine gaps. The coordinate flag, which does not store the coordinates, marks 22.55% missing, 54.76% outside the local metre envelope, and 22.69% inside it. No map was drawn. The envelope is not an official coordinate reference system.

### Figure 3. Outcome by work centre

![Outcome by work centre](../results/figures/03_03_outcome_by_work_centre.png)

The chart shows the 12 centres with the most labelled requests. They cover 1,415,810 requests, 30.9% of labelled requests. Energy is the largest, with 294,487 labelled requests. The median centre has 45 labelled requests, and 1,216 centres have fewer than 100. Among larger centres, the outcome mix is not uniform: some electricity contractor centres are mostly Resolved Fast, while Revenue Protection East is 98.5% Unresolved among centres with at least 5,000 labelled requests. These differences are descriptive. The extract has no assignment time, so work centre was not used as a model input.

### Figure 4. Outcome by complaint type

![Outcome by complaint type](../results/figures/03_04_outcome_by_complaint_type.png)

The 12 most common labelled complaint types cover 2,014,815 requests, 44.0% of labelled requests. No Power (474,818 requests) is 90.0% Resolved Fast. SEW: Blocked/Overflow (471,118) is closer to the overall mix. Suspect Tampering is 15.4% Resolved Fast, 45.2% Resolved Slow, and 39.4% Unresolved. Names beginning `DO NOT USE` remain in the extract. Complaint type is a logging-time field and is a model input. Rare names are grouped later, on the training rows only.

### Figure 5. Monthly request volume

![Monthly request volume](../results/figures/03_05_monthly_request_volume.png)

A typical month that contains rows has about 104,427 requests. January 2025 has none. February 2025 has 33,496 and September 2025 has 500. From May 2026 the Unresolved share rises, and September 2026 has no Resolved Slow row. October 2026 is partly inside the observation window, so many of its requests have no label. The late rise in Unresolved is the edge of a frozen extract, which is why rows created after 30 April 2026 are left out of the model splits.

### Figure 6. Resolution duration

![Resolution duration](../results/figures/03_06_resolution_duration.png)

Of 3,906,387 completed requests, 28.9% were closed on the same calendar day. The source stores dates and not times, so a duration of zero can still have taken hours. 76.0% of completions fall within seven days. The median completed duration is 2 days and the 75th percentile is 7 days. The maximum is 1,135 days. 14,953 completions longer than 365 days were kept.

## 4. Target construction

The label is built from Created On Date and Completed Date. It is not a City service standard. The cut is seven calendar days, which matches the 75th percentile of completed durations in this file. The observation window is also seven days. The reference date for the audit is 6 October 2026, the latest created date in the file.

- A completion within seven calendar days is Resolved Fast.
- A longer completion is Resolved Slow, including durations above 365 days.
- A request that is still open, and whose age is at least seven days, is Unresolved.
- An open request younger than seven days is `insufficient_follow_up` and is not labelled. There are 15,981 such rows. They are not treated as failures.
- A completion after the reference date is treated as not yet known, and the same open-request rule is applied.
- Twenty rows have a completion before creation and are excluded.
- Missing or malformed created dates: none in this file.

Completed Date, Changed on, and the derived duration are used to build the label. They are not predictors.

## 5. Cleaning, features, split, and leakage

Cleaning kept the long durations and the repeated notification ids that are not exact copies. Placeholder text such as `#`, `GENERIC TOWN 0000`, and `Not assigned` was kept as its own category rather than imputed. Coordinates and notification type were withheld before modelling.

The split is chronological. It does not use the random seed.

| Split | Dates | Labelled rows | Resolved Fast | Resolved Slow | Unresolved |
| --- | --- | --- | --- | --- | --- |
| Train | 2023-01-01 to 2024-12-31 | 2,616,595 | 66.2% | 29.3% | 4.5% |
| Validation | 2025-02-01 to 2025-12-31 | 936,469 | 69.3% | 9.4% | 21.3% |
| Locked test | 2026-01-01 to 2026-04-30 | 422,033 | 68.0% | 15.5% | 16.4% |

The validation window starts on 1 January 2025. January 2025 has no rows. The mature end is 30 April 2026 because May 2026 is where the labelled mix breaks. That cut was set from the monthly pattern before the models were fit. It was not chosen to maximise a score. 601,705 labelled rows created after that date are in none of the three splits.

Included predictors: complaint type, suburb, ward, subcouncil, weekday name, month name, and a weekend indicator. Weekday and month are one-hot encoded for the linear and neural models. The weekend indicator is the only numeric predictor. Its scaler is fit on the training rows.

Excluded: Work Center, because the extract has no assignment time; the raw Created On Date; Completed Date; Changed on; resolution duration; the target; notification identifiers; and coordinates.

For one-hot models, a training category is kept when it has at least 200 training rows. Rarer categories and categories first seen later become `(rare)`. Missing categorical values become `(missing)`. The encoded training matrix has 2,616,595 rows and 1,718 columns, with 18,316,165 stored values. A dense float32 copy would be 16.7 GiB, so logistic regression and the multilayer perceptron use the sparse matrix.

Histogram gradient boosting uses native categorical codes. The booster accepts at most 255 categories, so the tree encoder keeps at most 253 training levels plus missing and rare. Suburb and complaint type each keep 253 levels. An unseen category uses the trained rare code when that code exists.

Class imbalance uses balanced weights from the training counts: Resolved Fast 0.504, Resolved Slow 1.137, Unresolved 7.372. Validation and test rows are not resampled. SMOTE is not used, because the predictors are categories and SMOTE would invent levels that were not logged.

## 6. Dummy baseline and traditional models

The most-frequent dummy predicts the majority training class, Resolved Fast. On the validation period its accuracy is 0.693 and its macro-F1 is 0.273. On the locked test its accuracy is 0.680 and its macro-F1 is 0.270. Recall is 1 for Resolved Fast and 0 for the other two classes. That is the floor a useful model has to clear on macro-F1.

Logistic regression uses saga, C = 1, L2, balanced class weights, max_iter 200, tolerance 0.001, and seed 42. Fit time was 247.9 seconds. The solver stopped at the iteration cap with a convergence warning. Validation scores are accuracy 0.578, macro-F1 0.460, and weighted-F1 0.630. Locked-test scores from notebook 03 are accuracy 0.577, macro-F1 0.467, and weighted-F1 0.610. Locked-test recall is 0.672 for Resolved Fast, 0.454 for Resolved Slow, and 0.303 for Unresolved. Resolved Slow precision is 0.202.

Histogram gradient boosting, in the Phase 5 configuration, uses log loss, learning rate 0.1, at most 31 leaves, minimum 50 samples per leaf, L2 regularisation 1, and balanced weights. Early stopping watches validation macro-F1. Fit time was 17.9 seconds. It trained 8 iterations and the predictions use iteration 2. Validation macro-F1 at that iteration is 0.482. Accuracy is 0.591, weighted-F1 is 0.640, Resolved Slow recall is 0.416, and Unresolved recall is 0.492. Macro ROC-AUC is 0.739 and macro average precision is 0.506.

A dense random forest on the one-hot matrix was not fit.

## 7. Neural networks

The multilayer perceptron reads the training-only one-hot pipeline. The architecture is 1,718 inputs, hidden widths 128 and 64, and 3 logits. Activation is ReLU. Dropout is 0.3. The optimiser is AdamW with learning rate 0.001 and weight decay 0.0001. Batch size is 8,192. The loss is class-weighted cross-entropy. The maximum is 8 epochs, with patience 3 on validation macro-F1. The network has 228,483 parameters. It ran 6 epochs, restored epoch 3, and stopped early. Fit time was 329.5 seconds.

Validation scores: accuracy 0.601, macro-F1 0.476, weighted-F1 0.647, Resolved Slow recall 0.452, Unresolved recall 0.394, macro ROC-AUC 0.739, macro average precision 0.508.

The learning curve is `results/figures/05_mlp_learning_curves.png`. Training loss fell from 0.638 at epoch 1 to 0.553 at epoch 6. Validation loss rose from 1.461 to 1.993. Validation macro-F1 peaked at 0.476 on epoch 3. The training curve is computed with dropout turned off so it can be compared with the validation curve.

![MLP learning curves](../results/figures/05_mlp_learning_curves.png)

The second network embeds each categorical field. Embedding dimension is 16, the hidden layer has 64 units, and dropout is 0.3. The optimiser, batch size, loss, and early-stopping rule match the perceptron. Unseen categories use index 0, and that embedding row stays at zero. It ran 7 epochs and restored epoch 4. Fit time was 236.2 seconds. Validation macro-F1 is 0.481. This architecture is pending lecturer approval. It is not an approved experiment, and it did not choose the champion. Its curve is `results/figures/05_embedding_learning_curves.png`. Training loss fell while validation loss rose, as it did for the perceptron.

## 8. Evaluation and model comparison

Macro-F1 is the primary metric. Accuracy, weighted-F1, and recall for Resolved Slow and Unresolved are reported with it. Where the model implements `predict_proba`, macro one-versus-rest ROC-AUC and macro average precision are reported. Those are the ROC and precision-recall summaries that were computed. Separate per-class ROC and precision-recall plots were not saved.

Phase 5 validation comparison, approved models only:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Slow recall | Unresolved recall | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Histogram gradient boosting | 0.591 | 0.482 | 0.640 | 0.416 | 0.492 | 0.739 | 0.506 |
| Multilayer perceptron | 0.601 | 0.476 | 0.647 | 0.452 | 0.394 | 0.739 | 0.508 |
| Logistic regression | 0.578 | 0.460 | 0.630 | 0.460 | 0.367 | 0.729 | 0.496 |
| Most-frequent dummy | 0.693 | 0.273 | 0.567 | 0.000 | 0.000 | not computed | not computed |

![Validation comparison](../results/figures/05_validation_comparison.png)

The dummy has the highest accuracy and the lowest macro-F1. Logistic regression has the highest approved Resolved Slow recall. Boosting has the highest approved macro-F1 and the highest approved Unresolved recall.

The locked test was scored for the dummy and logistic regression in notebook 03. It was not scored in Phase 5. It was scored once more, after the Phase 6 freeze, for the frozen booster and for the multilayer perceptron. Those later scores are in section 10. Confusion matrices for the Phase 5 validation fits are in `results/figures/05_confusion_*_validation.png`.

## 9. Overfitting diagnosis and controls

Training macro-F1 is higher than validation macro-F1 for every fitted model in Phase 5.

| Model | Training macro-F1 | Validation macro-F1 | Gap |
| --- | --- | --- | --- |
| Histogram gradient boosting | 0.573 | 0.482 | +0.091 |
| Multilayer perceptron | 0.633 | 0.476 | +0.156 |
| Logistic regression | 0.578 | 0.460 | +0.118 |
| Embedding network, pending approval | 0.586 | 0.481 | +0.105 |

The perceptron gap is the largest. Its validation loss rose while training loss fell, which is the overfitting pattern the early-stopping rule was there to limit. The restored epoch is the one with the highest validation macro-F1, not the last epoch.

The boosting validation curve peaks at iteration 2 (macro-F1 0.482) after a first iteration at 0.474. Iterations 3 to 8 stay near 0.474 to 0.477. The per-iteration training score in that file is the library's 10,000-row subset, so the training macro-F1 in the table above is the one computed on the full training period. The curve is `results/figures/05_boosting_validation_curve.png`.

Controls that were used: a chronological split, encoders and the scaler fit on the training period only, rare-category grouping, dropout and weight decay on the neural nets, L2 on the booster and the logistic model, early stopping on validation macro-F1, and a refusal to resample the validation or test rows. Logistic regression did not reach its tolerance, so its fit is stopped by the iteration cap rather than by a converged solution. The Phase 5 boosting iteration was selected with the 2025 validation scores. Phase 6 does not repeat that. Its frozen iteration cap comes from the training folds.

## 10. Hyperparameter optimisation and champion selection

The Phase 5 provisional champion is histogram gradient boosting, because its validation macro-F1 is the highest among approved models. The pending embedding network was excluded from that choice. The locked test was not an input.

Phase 6 tunes only that family. The search space is learning rate {0.05, 0.1, 0.2}, maximum leaves {15, 31, 63}, minimum samples per leaf {50, 200, 500}, and L2 {0.1, 1.0, 10.0}. Eight draws come from `ParameterSampler` with seed 42. Trial 0 is the untuned Phase 5 setting. Each trial is scored on three expanding folds inside the training period:

- train through 30 June 2023, score the rest of 2023
- train through 31 December 2023, score the first half of 2024
- train through 30 June 2024, score the rest of 2024

The trial score is the mean macro-F1. Early stopping inside a fold uses that fold's later slice. Trial 5, learning rate 0.2, 15 leaves, minimum 50 samples, L2 0.1, has the highest mean, 0.544, and a median selected iteration of 4. The untuned setting has mean 0.540 and median iteration 12.

Both settings were then refit on all training rows with early stopping off and the iteration cap set to that median. On the 2025 validation period the untuned refit scores macro-F1 0.477 and the tuned refit scores 0.470. The pre-specified rule keeps the tuned refit only when its validation macro-F1 is higher. The frozen candidate is the untuned setting: learning rate 0.1, 31 leaves, minimum 50 samples per leaf, L2 1, 12 iterations, balanced class weights. That decision was written to `results/tables/06_frozen_decision.json` before the locked test was opened.

The Phase 5 validation macro-F1 of 0.482 is not the same number. That fit chose iteration 2 on the 2025 curve. The frozen refit does not look at 2025 while it chooses the iteration count.

The frozen model was scored once on the locked test and was not changed afterwards.

| | Accuracy | Macro-F1 | Weighted-F1 | Slow recall | Unresolved recall | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen booster | 0.600 | 0.494 | 0.629 | 0.400 | 0.418 | 0.724 | 0.519 |
| Multilayer perceptron | 0.606 | 0.495 | 0.634 | 0.458 | 0.345 | 0.731 | 0.529 |

Locked-test precision for the frozen booster is 0.829 for Resolved Fast, 0.210 for Resolved Slow, and 0.495 for Unresolved. Support is 287,131, 65,522, and 69,380. The low Resolved Slow precision means most requests predicted as slow are not slow.

![Frozen model, locked-test confusion matrix](../results/figures/06_confusion_locked_test.png)

The perceptron was retrained with the Phase 5 procedure before its test score. Its retrained validation macro-F1 is 0.476, the same value as Phase 5. Its locked-test confusion matrix is `results/figures/07_confusion_comparator_locked_test.png`.

## 11. Ablation experiments

The ablations use the frozen hyperparameters, the same chronological split, and the 2025 validation period. They were not scored on the locked test. One component changes at a time. The reference macro-F1 is 0.477.

| Experiment | Fit? | Macro-F1 | Change | Slow recall | Change | Unresolved recall | Change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Remove suburb, ward, and subcouncil | Yes | 0.469 | −0.008 | 0.434 | +0.002 | 0.475 | +0.033 |
| Remove work centre | No second model | 0.477 | 0 | 0.432 | 0 | 0.442 | 0 |
| No class weight | Yes | 0.376 | −0.101 | 0.344 | −0.089 | 0.037 | −0.405 |

![Ablation changes](../results/figures/07_ablation_deltas.png)

Removing the geographic fields changes macro-F1 by less than one percentage point. Unresolved recall rises by 0.033. The geographic fields are not useless on every class, and they are also not the main source of the macro-F1 score in this comparison.

Work centre was already absent, because the extract has no assignment time. Removing it does not change the inputs. No second model was fit. The zero change is by construction. It is not evidence about how a leakage-safe work-centre field would behave, because that field is not in the extract.

Turning class weights off, with no SMOTE and no other resampling, drops Unresolved recall from 0.442 to 0.037. Unresolved is 4.5% of the training rows, so an unweighted model can score well on the majority class and miss that class. The balanced weights are doing the work this project asked for on Unresolved recall.

## 12. McNemar's test

The comparison is the frozen booster and the multilayer perceptron on the same 422,033 locked-test cases. The perceptron was the strongest approved model other than the champion on the Phase 5 validation table. That choice was made before the locked-test scores were inspected. Paired predictions are in `results/tables/07_paired_predictions.parquet`. The file has the source row index, the true label, both predictions, and the champion probabilities. It does not have notification ids or coordinates.

Hypotheses, two-sided, alpha 0.05:

- Null: the two models have the same probability of being correct on a locked-test case.
- Alternative: those probabilities differ.

| Both correct | Champion only | Comparator only | Both wrong | Discordant |
| --- | --- | --- | --- | --- |
| 233,776 | 19,282 | 21,854 | 147,121 | 41,136 |

The discordant count is above 25, so the continuity-corrected chi-square approximation is the method that the code uses. The exact binomial test is reserved for fewer than 25 discordant cases. The statistic is (|19282 − 21854| − 1)² / 41136 = 160.6875. The p-value is 8.006e-37. That is below 0.05, so the paired error rates differ. The comparator is correct on 2,572 more of the discordant cases.

The test compares paired error rates. It does not show that one model is better on every class. The practical gaps, champion minus comparator, are −0.006 in accuracy and −0.001 in macro-F1. On this test set the perceptron has higher Resolved Slow recall (0.458 against 0.400) and lower Unresolved recall (0.345 against 0.418). A small accuracy gap can still have a very small p-value when the test set has 422,033 rows. The frozen model was not changed after this result.

## 13. Prediction interface

`streamlit run app/app.py` loads `models/frozen_resolution_pipeline.joblib`. The process does not fit a model and does not rebuild the encoder. Notebook 08 checks that file and does not refit it. The export compared the refit with the notebook 06 predictions: 422,033 of 422,033 labels matched.

The form asks for complaint type, suburb, ward, subcouncil, and the creation date. Dropdown values are categories kept from the training period. The date is used only to derive the weekday, the month, and the weekend flag. Work centre is not requested. Unknown categories are grouped with rare or unseen levels. Missing fields and dates outside 1 January 2023 to 6 October 2026 are rejected.

The page states that this is an academic prototype and not an official City of Cape Town system, and that predictions must support human review and must not be used to deny or reduce service. Probabilities are described as model confidence under balanced class weights, not as causes and not as the historical share of each class.

The demonstration checklist is `reports/demonstration_checklist.md`. Eight cases were run on the saved pipeline. The sample case, No Power in PHILIPPI on 12 June 2024, was also submitted in the browser and showed Resolved Fast with probabilities 0.598, 0.247, and 0.154. A second assembled input, using other frequent categories, was scored as Unresolved. That row is not a logged municipal case. Screenshot files were not saved.

## 14. Ethics, fairness, limitations, and recommendations

The City terms say the data are provided as is and that reliance is at the user's risk. This prototype is not an official City system. The label is a project rule, not a published service standard. A prediction does not establish why a request will be completed or will stay open.

Resolved Slow precision on the locked test is 0.210. If a department treated "Resolved Slow" as a reason to delay or refuse work, most of those flags would be wrong, and the harm would fall on the people waiting for the service. The interface warning exists for that reason. The model is a review aid.

No fairness audit of model error across suburbs or wards was run. The exploratory tables do show that area fields are messy: suburb `#` is 7.97% of analysis rows, subcouncil `Not assigned` is 35.34%, and ward `000` is a very common stored value. The geographic ablation changes validation macro-F1 by −0.008, which is not a disparity analysis. A later audit should compare recall for Resolved Slow and Unresolved across suburbs and wards on the validation period, and it should not retune the model on the locked test. That audit is future work, not a result of this draft.

Other limitations, all from the executed work:

- The layer is frozen. Open requests may have been completed later, so some Unresolved labels can be wrong.
- The 2025 class mix is not the 2023–2024 mix. Unresolved rises from 4.5% to 21.3%.
- Logistic regression did not converge.
- The embedding network is not approved.
- The work-centre ablation is not a fitted comparison.
- Balanced weights improve Unresolved recall and, on the logistic test score, leave Resolved Slow precision at 0.202.
- Macro ROC-AUC near 0.72 to 0.74 is better than chance and is not a basis for an automated decision.

Practical recommendations:

- Keep a person between the score and any operational action.
- Do not rank areas for reduced service from these scores.
- If the model is shown to staff, show the three probabilities and the seven-day definition, and say that the weights are balanced.
- Do not add work centre until an assignment time is known.
- Do not treat the pending embedding network as the delivered model unless a lecturer approves it and the locked test is not used to make that approval.

## 15. Conclusion and future work

On the measurements that were saved, a most-frequent prediction reaches validation macro-F1 0.273. Logistic regression reaches 0.460. Histogram gradient boosting reaches 0.482 in the Phase 5 early-stopped fit and is the approved champion family. A training-fold search did not beat the untuned refit on the 2025 validation macro-F1, so the frozen pipeline keeps those untuned settings with a 12-iteration cap. Its single locked-test macro-F1 is 0.494. The multilayer perceptron is close on macro-F1 and differs in which minority class it recalls. McNemar's test finds a difference in paired error rates and does not justify replacing the frozen model after seeing the test.

This report does not declare the project complete. The following are still open:

- Lecturer approval of the categorical embedding network. Until that decision, its scores stay out of the champion rule.
- A second fitted work-centre model. It cannot be leakage-safe with the current extract.
- A suburb and ward fairness audit of model error. It was not run.
- Stakeholder review with a service department. It was not run.
- Per-class ROC and precision-recall figures. Only the macro summaries were saved.
- A full re-execution of notebooks 01 to 07 from an empty environment in this phase. Those notebooks already contain executed outputs with no error cells, and their source cells do not hardcode a machine path. They were not run again while this draft was written. Notebook 08 was executed. It only checks the saved pipeline.
- A PDF export. The Markdown is the draft, and Pandoc was not available here.
- Saved screenshots of the form. The browser check and `reports/demonstration_checklist.md` are the demonstration evidence.

Future work that would not reuse the locked test for tuning: approve or reject the embedding architecture, calibrate the probabilities, audit minority-class recall by suburb and ward on the validation period, and revisit work centre only if an assignment timestamp becomes available.
