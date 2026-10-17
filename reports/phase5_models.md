# Phase 5 model experiments

Lecturer approval of the embedding network was recorded after this file was written. No approval date was supplied, and the architecture in this file was not changed. The scores below are the original run. The champion remains histogram gradient boosting because its validation macro-F1 is higher.

Notebooks 04 and 05 fit the traditional models and the neural models on the same chronological partitions. The provisional champion is chosen from validation macro-F1. The locked test is not scored in this phase.

## Split

- Train: 2023-01-01 to 2024-12-31, 2,616,595 labelled requests (66.2% Resolved Fast, 29.3% Resolved Slow, 4.5% Unresolved).
- Validation: 2025-02-01 to 2025-12-31, 936,469 labelled requests (69.3% Resolved Fast, 9.4% Resolved Slow, 21.3% Unresolved).
- Locked test, not scored: 2026-01-01 to 2026-04-30, 422,033 labelled requests (68.0% Resolved Fast, 15.5% Resolved Slow, 16.4% Unresolved).

The split dates are the Phase 4 dates. They were not moved to improve a model score.

## Encoding

The one-hot training matrix has 2,616,595 rows and 1,718 columns, with 18,316,165 stored values. A dense float32 copy would be 16.7 GiB. The sparse matrix is practical for logistic regression and the multilayer perceptron. A dense random forest on that matrix is not practical in this environment, so the tree model is histogram gradient boosting on native categorical codes.

Tree codes are fit on the training period. A level needs at least 200 training rows and the encoder keeps at most 253 levels, because the booster accepts at most 255 categories including the missing and rare levels.

- C3_Complaint_Type: 253 of 998 training levels kept, 254 encoded levels.
- Suburb: 253 of 28,394 training levels kept, 254 encoded levels.
- Ward: 218 of 219 training levels kept, 219 encoded levels.
- Sub_Council: 23 of 32 training levels kept, 25 encoded levels.
- created_day_of_week: 7 of 7 training levels kept, 7 encoded levels.
- created_month: 12 of 12 training levels kept, 12 encoded levels.

The embedding encoder keeps the same 200-row rare rule and does not apply the 255-level cap. Unseen categories use index 0, whose embedding stays at zero. Work centre is not an input.

## Imbalance

All approved models use balanced class weights computed from the training counts. Validation rows are not resampled. SMOTE is not used. The predictors are categorical codes or one-hot columns, and SMOTE on raw category strings would invent levels that were not logged.

## Models

Logistic regression is the interpretable baseline from Phase 4: saga, C = 1, L2, balanced class weights, max_iter = 200, tol = 0.001. Fit time was 247.9 seconds. The logistic solver stopped with a convergence warning before the tolerance was met.

Histogram gradient boosting uses log loss, learning rate 0.1, at most 31 leaves, minimum 50 samples per leaf, L2 regularisation 1, balanced class weights, and early stopping on the chronological validation macro-F1. Fit time was 17.9 seconds. Boosting trained 8 iterations and predictions use iteration 2, where validation macro-F1 on the early-stopping curve was 0.482. Early stopping halted training before the iteration cap.

The multilayer perceptron reads the training-only one-hot pipeline. It has three output logits, hidden sizes [128, 64], ReLU, dropout 0.3, AdamW with learning rate 0.001 and weight decay 0.0001, batch size 8192, and class-weighted cross-entropy. It has 228,483 parameters and 1,718 inputs. It ran 6 epochs, restored epoch 3, and stopped early. Fit time was 329.5 seconds.

The second neural architecture is a categorical embedding network. Lecturer approval has been recorded. No approval date was supplied, and the architecture was not changed. Embedding dimension 16, hidden size 64, dropout 0.3, AdamW learning rate 0.001, weight decay 0.0001, batch size 8192. It ran 7 epochs and restored epoch 4. Fit time was 236.2 seconds. Its validation macro-F1 is below histogram gradient boosting, so it is not the champion. No locked-test score was produced for it.

## Validation

- hist_gradient_boosting (approved): accuracy 0.591, macro-F1 0.482, weighted-F1 0.640, Resolved Slow recall 0.416, Unresolved recall 0.492, macro ROC-AUC 0.739, macro PR-AUC 0.506, fit 17.9 seconds.
- multilayer_perceptron (approved): accuracy 0.601, macro-F1 0.476, weighted-F1 0.647, Resolved Slow recall 0.452, Unresolved recall 0.394, macro ROC-AUC 0.739, macro PR-AUC 0.508, fit 329.5 seconds.
- logistic_regression (approved): accuracy 0.578, macro-F1 0.460, weighted-F1 0.630, Resolved Slow recall 0.460, Unresolved recall 0.367, macro ROC-AUC 0.729, macro PR-AUC 0.496, fit 247.9 seconds.
- embedding_network (approved; not the champion): accuracy 0.592, macro-F1 0.481, weighted-F1 0.642, Resolved Slow recall 0.435, Unresolved recall 0.462, macro ROC-AUC 0.741, macro PR-AUC 0.513, fit 236.2 seconds.

## Training and validation

- hist_gradient_boosting: training macro-F1 0.573, validation macro-F1 0.482, gap +0.091.
- multilayer_perceptron: training macro-F1 0.633, validation macro-F1 0.476, gap +0.156.
- logistic_regression: training macro-F1 0.578, validation macro-F1 0.460, gap +0.118.
- embedding_network (approved; not the champion): training macro-F1 0.586, validation macro-F1 0.481, gap +0.105.

Learning curves for the neural models show training and validation loss and macro-F1. Those training curves are computed with dropout turned off, on the full training period, so they can be compared with the validation curves. On both neural networks, training loss fell while validation loss rose across the epochs that were run. Early stopping restored the epoch with the highest validation macro-F1. The boosting curve shows validation macro-F1 at each iteration. The per-iteration boosting training score is the library's 10,000-row subset, so the training macro-F1 in the table above is the one computed on the full training period.

## Provisional champion

The provisional champion is hist_gradient_boosting. Its validation macro-F1 is 0.482. Validation accuracy 0.591, macro-F1 0.482, weighted-F1 0.640, Resolved Slow recall 0.416, Unresolved recall 0.492. This choice uses validation macro-F1 among the approved models only. The locked test was not scored and did not enter the decision.

The highest approved validation recall for Resolved Slow is 0.460 (logistic_regression). The macro-F1 leader's Resolved Slow recall is 0.416.
The highest approved validation recall for Unresolved is 0.492 (hist_gradient_boosting). The macro-F1 leader's Unresolved recall is 0.492.

The categorical embedding network has validation macro-F1 0.481, Resolved Slow recall 0.435, and Unresolved recall 0.462. Lecturer approval has been recorded, so those scores are included in the champion rule. Histogram gradient boosting remains higher at 0.482. No locked-test score was produced for the embedding network.

These Phase 5 settings were not grid-searched. The champion family was tuned later on training-period folds. This phase does not claim a locked-test result. The repository does not contain a marking brief that requires a separate search for logistic regression, the perceptron, or the embedding network.
