# Phase 4 baselines

Notebook 03 fits two baselines on the labelled file. No champion was tuned.

## Split

Random seed for the logistic solver: 42. The split itself is chronological and does not use the seed.

- Train: 2023-01-01 to 2024-12-31, 2,616,595 labelled requests.
- Validation: 2025-02-01 to 2025-12-31, 936,469 labelled requests. The validation window starts on 1 January 2025. January 2025 has no rows.
- Locked test: 2026-01-01 to 2026-04-30, 422,033 labelled requests.
- Unlabelled rows excluded: 16,001.
- Labelled rows created after 2026-04-30 excluded: 601,705.

Training class shares are 66.2% Resolved Fast, 29.3% Resolved Slow, and 4.5% Unresolved. Validation shares are 69.3% Resolved Fast, 9.4% Resolved Slow, and 21.3% Unresolved. Locked-test shares are 68.0% Resolved Fast, 15.5% Resolved Slow, and 16.4% Unresolved.

The mature end is 30 April 2026 because May 2026 is where the labelled mix breaks: Unresolved rises sharply and Resolved Slow shrinks, with the extract ending on 6 October 2026. The cut was set from that monthly pattern before the models were fit. It was not chosen to maximise a score.

## Predictors

Included: complaint type, suburb, ward, subcouncil, weekday, month, and a weekend indicator. Weekday and month are one-hot encoded. The weekend indicator is standardised with a scaler fit on the training rows.

Excluded: Work Center, because the extract has no assignment time; Created On Date as a raw timestamp; Completed Date; Changed on; resolution duration; the target; identifiers; and coordinates.

A training category is kept when it has at least 200 training rows. Rarer categories and categories first seen later become `(rare)`. Missing categorical values become `(missing)`.

- C3_Complaint_Type: 372 of 998 training categories kept (99.1% of non-missing training rows).
- Suburb: 1,080 of 28,394 training categories kept (95.2% of non-missing training rows).
- Ward: 218 of 219 training categories kept (100.0% of non-missing training rows).
- Sub_Council: 23 of 32 training categories kept (100.0% of non-missing training rows).
- created_day_of_week: 7 of 7 training categories kept (100.0% of non-missing training rows).
- created_month: 12 of 12 training categories kept (100.0% of non-missing training rows).

The encoded training matrix has 1,718 columns.

## Scores

Macro-F1 is the primary metric. The most-frequent dummy predicts the majority training class.

- Dummy validation: accuracy 0.693, macro-F1 0.273, weighted-F1 0.567.
- Dummy locked test: accuracy 0.680, macro-F1 0.270, weighted-F1 0.551.
- Logistic validation: accuracy 0.578, macro-F1 0.460, weighted-F1 0.630.
- Logistic locked test: accuracy 0.577, macro-F1 0.467, weighted-F1 0.610.

Locked-test logistic precision and recall:

- Resolved Fast: precision 0.823, recall 0.672.
- Resolved Slow: precision 0.202, recall 0.454.
- Unresolved: precision 0.515, recall 0.303.

Training macro-F1 was 0.265 for the dummy and 0.578 for logistic regression. The logistic training score is higher than the validation score. Validation and test were not resampled.

## Limitations

- The logistic solver stopped at the pre-set iteration cap before the tolerance was met.
- `C`, the rare-count rule, and the class weights were not searched.
- Balanced weights raise recall for Resolved Slow and Unresolved and lower precision for Resolved Slow.
- Work Center remains unused until its assignment time is known.
- The 2025 validation mix is substantially different from the 2023–2024 training mix.

