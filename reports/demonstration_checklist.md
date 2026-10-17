# Demonstration checklist

These cases were run on 17 October 2026 against `models/frozen_resolution_pipeline.joblib`. The full numeric record is `results/tables/08_demonstration_cases.csv`. None of these rows is a logged municipal request. A prediction of Unresolved is the model's class for the typed fields. It is not a City case file.

The sample row was also submitted through the running Streamlit form in a browser. The form showed Resolved Fast with probabilities 0.598, 0.247, and 0.154, which matches this table at three decimal places. Screenshot files were not saved.

| Case | Input | Result |
| --- | --- | --- |
| Demonstration sample | Complaint type No Power, suburb PHILIPPI, ward 000, subcouncil Sub-Council 16, created 2024-06-12. Frequent training categories. Placeholders `#` and `Not assigned` were skipped. | Scored. Resolved Fast. Probabilities 0.598, 0.247, 0.154. |
| Second set of frequent categories | Complaint type SEW: Blocked/Overflow, suburb PHILIPPI, ward 100, subcouncil Sub-Council 16, created 2024-06-12. Assembled for the form. | Scored. Unresolved. Probabilities 0.142, 0.166, 0.692. Not a logged unresolved case. |
| Unknown suburb | Same as the sample, with suburb `ZZZ Not A Cape Town Suburb`. | Scored. Resolved Fast. Probabilities 0.703, 0.164, 0.133. The suburb was grouped with rare or unseen levels. |
| Date after the training period | Same as the sample, created 2026-03-15. | Scored. Resolved Fast. Probabilities 0.735, 0.201, 0.063. The page notes that the booster was fit through 31 December 2024. |
| Missing complaint type | Sample fields with complaint type omitted. | Rejected. Message: "Choose a complaint type." |
| Date before the extract | Created 2022-12-31. | Rejected. The date must fall between 1 January 2023 and 6 October 2026. |
| Invalid date | Text `not-a-date`. | Rejected. The creation date is not a valid calendar date. |
| Work centre supplied | Sample fields plus Work Center Energy. | Rejected. Work centre is not an input to the fitted pipeline. |
