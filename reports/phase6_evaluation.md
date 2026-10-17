# Phase 6 evaluation

The provisional champion family is histogram gradient boosting, from the Phase 5 validation macro-F1 ranking. Pending embedding results did not enter that choice. Tuning used chronological folds inside the training period only.

## Search

Random draws: 8. Seed: 42. The untuned Phase 5 settings were scored on the same folds. The score is mean macro-F1. Early stopping inside a fold uses that fold's later slice.

The highest training-fold mean is trial 5 (parameter_sampler), mean macro-F1 0.544, median selected iteration 4.

## Validation comparison

- untuned: validation macro-F1 0.477, accuracy 0.591, Resolved Slow recall 0.432, Unresolved recall 0.442.
- tuned: validation macro-F1 0.470, accuracy 0.575, Resolved Slow recall 0.427, Unresolved recall 0.468.

The frozen candidate is `untuned`. The rule keeps the tuned refit only when its validation macro-F1 is higher than the untuned refit. A tie keeps the untuned settings. This choice was saved before the locked test was scored.

## Locked test

- untuned: accuracy 0.600, macro-F1 0.494, weighted-F1 0.629, Resolved Slow recall 0.400, Unresolved recall 0.418, macro ROC-AUC 0.724, macro PR-AUC 0.519.
- multilayer_perceptron: accuracy 0.606, macro-F1 0.495, weighted-F1 0.634, Resolved Slow recall 0.458, Unresolved recall 0.345, macro ROC-AUC 0.731, macro PR-AUC 0.529.

The frozen parameters were not changed after these test metrics were computed.

## Ablations

- remove_geographic_features: validation macro-F1 0.469 (change -0.008), Resolved Slow recall 0.434 (change +0.002), Unresolved recall 0.475 (change +0.033). Retrained without suburb, ward, and subcouncil. Complaint type and calendar fields stay.
- remove_work_centre: validation macro-F1 0.477 (change +0.000), Resolved Slow recall 0.432 (change +0.000), Unresolved recall 0.442 (change +0.000). Work Center is already excluded from the frozen pipeline because the extract has no assignment time. Removing it does not change the inputs. No second model was fit.
- no_class_weight: validation macro-F1 0.376 (change -0.101), Resolved Slow recall 0.344 (change -0.089), Unresolved recall 0.037 (change -0.405). Same features and frozen hyperparameters with class_weight left unset. SMOTE is not used.

## McNemar

Comparator: multilayer_perceptron. Both correct 233,776. Champion only 19,282. Comparator only 21,854. Both wrong 147,121.

Method: chi-square with continuity correction. Statistic: 160.6875. p-value: 8.006e-37. Threshold: 0.05. Conclusion: The paired error rates differ at the 0.05 threshold. This does not show that one model is better on every class.

The test compares paired error rates on the same cases. It does not show that one model is better on every class.

Accuracy difference (champion minus comparator): -0.0061. Macro-F1 difference: -0.0010. Those gaps are the practical comparison. A small gap can still have a small p-value when the test set is large.
