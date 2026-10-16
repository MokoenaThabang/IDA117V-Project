# Phase 3 exploratory findings

Notebook 02 describes `data/processed/service_requests_labelled.parquet` (4,592,803 rows after the Phase 2 duplicate rule). Created On Date runs from 1 January 2023 to 6 October 2026. The provisional label cut is seven calendar days. No model was trained. Figures are in `results/figures/03_01_target_distribution.png` through `03_06_resolution_duration.png`. Supporting counts are in `results/tables/03_*.csv`.

## Target balance

Of 4,576,802 labelled requests, 2,969,242 (64.9%) are Resolved Fast, 937,145 (20.5%) are Resolved Slow, and 670,415 (14.6%) are Unresolved. Another 16,001 rows have no label: 15,981 open requests younger than seven days, and 20 rows whose completion date is earlier than the creation date.

Resolved Fast is the majority class, so accuracy alone would reward a fast-only prediction. Macro-F1 is the metric that gives the three classes equal weight. Resolved Slow and Unresolved are the operational classes, and together they are about 35% of labelled requests. The imbalance also follows the duration distribution: the median completed duration is 2 days and the 75th percentile is 7 days, so the provisional cut places most recorded completions in Resolved Fast.

## Missing values and blanks that mean something

Complaint type, work centre, and Created On Date have no missing values. Subcouncil is missing on 4.38% of analysis rows, ward on 1.78%, and suburb on 0.98%. Notifications Created is missing on 0.08%.

A blank completion date is usually an open request. 3,906,387 rows have a recorded duration. 670,415 blanks are open requests old enough to be labelled Unresolved. 15,981 blanks are still inside the seven-day window and stay unlabelled. The 20 date reversals are invalid.

Placeholders are larger than the genuine gaps. Suburb is `#` on 365,919 rows (7.97%) and `GENERIC TOWN 0000` on 4,248 rows. Subcouncil is `Not assigned` on 1,622,881 rows (35.34%) and `0` on 8,465 rows. There are 44,905 distinct suburb values. These stored codes need a grouping rule before they are used as areas.

779,663 exact duplicate business rows were removed before this file was written. 309,758 rows still share a notification id with a row that differs in other fields; those rows were kept. 14,953 completions longer than 365 days were kept.

The coordinate flag, which does not store the coordinates, marks 22.55% missing, 54.76% outside the local metre envelope, and 22.69% inside it. The envelope is not an official coordinate reference system, so no map is published.

## Work centre

The analysis file has 2,076 work centres, and 2,075 of them appear among labelled requests. The chart shows the 12 centres with the most labelled requests. They cover 1,415,810 requests, 30.9% of labelled requests. Energy is the largest, with 294,487 labelled requests (62.5% Resolved Fast, 27.2% Resolved Slow, 10.3% Unresolved). Electricity first-line contractor centres in that set are mostly Resolved Fast (about 77% to 88%). WSR- CTA Debt Management is 97.4% Resolved Fast (103,615 requests).

1,216 centres have fewer than 100 labelled requests and together hold 21,425 requests (0.47%). The median centre has 45 labelled requests. Percentages for those small centres are left in the full table and off the chart.

Among centres with at least 5,000 labelled requests, Revenue Protection East is 98.5% Unresolved and Reinst Reg 4 Wat T/O is 87.8% Unresolved. Vending Investigations & Correspondence is 92.2% Resolved Slow. The centre stored as `Do not use` is 82.2% Resolved Slow. Work centre may have been assigned after logging, so these differences are descriptive.

## Complaint type

The analysis file has 1,154 complaint types. The chart uses the same count rule: the 12 types with the most labelled requests. They cover 2,014,815 requests, 44.0% of labelled requests.

No Power (474,818 requests) is 90.0% Resolved Fast. The two reconnection types in that set are 99.2% and 99.7% Resolved Fast. SEW: Blocked/Overflow (471,118) is close to the overall mix. Street Lights - All Lights Out is 31.3% Resolved Slow. Suspect Tampering is 15.4% Resolved Fast, 45.2% Resolved Slow, and 39.4% Unresolved. The truncated water-leak type WAT: Leak in Road/Pavement/Undergr is 35.8% Resolved Slow and 20.6% Unresolved.

Duplicate (73,682 requests) is 64.7% Resolved Slow, and only 3 of those requests are Unresolved. Among types with at least 10,000 labelled requests, names beginning `DO NOT USE` are often slow: DO NOT USE (Plumbing) is 72.2% Resolved Slow. High Unresolved shares in that size band include Doors: Internal / External (86.6%), Carpentry (69.1%), WAT: Re-instatement (67.9%), and Meter Replacement Project (58.7%). Retired and project-style names are still in the extract and need an explicit grouping rule later.

## Time

A typical month that contains rows has about 104,427 requests. January 2025 has none. February 2025 has 33,496 and September 2025 has 500. September 2025 is 90.4% Resolved Fast, on that base of 500. April 2024 is the high month, at 200,348 requests, and its leading complaint types are the same types that lead March 2024. July 2024 (130,058 requests) is 37.9% Resolved Slow and 1.9% Unresolved.

From May 2026 the Unresolved share rises: 38.6% in May, 46.9% in July, 56.9% in August, and 64.0% in September (65,200 of 101,929 labelled September requests). September 2026 has no Resolved Slow row. Its 36,729 recorded durations are all 2 days or less. October 2026 has 20,294 requests created on or before 6 October. 13,124 are inside the observation window and have no label. The 7,170 labelled October requests are all Resolved Fast, because only a completion inside those few days can be labelled. The late rise in Unresolved is the edge of a frozen extract.

## Duration

Of 3,906,387 completed requests, 28.9% were closed on the same calendar day. The source stores dates and not times, so duration 0 can still have taken hours. 76.0% of completions fall within seven days, which is exactly the Resolved Fast count. The 90th percentile is 32 days and the 99th is 236 days. The maximum is 1,135 days. 14,953 completions (0.4%) exceed 365 days, including 3,075 over 730 days and 300 over 1,000 days. The longest group includes Meter Replacement Project and several `DO NOT USE` names. Those rows are retained.

## Implications for the next notebooks

- Evaluate with macro-F1, and report recall for Resolved Slow and Unresolved.
- Keep the seven-day cut configurable. It matches this file's 75th percentile and is not a citywide service standard.
- Treat blank completion dates as open requests. Do not impute them as a predictor, and do not use duration, Completed Date, or Changed on as inputs.
- Group placeholder areas and retired complaint names before encoding them.
- Treat work centre as uncertain until it is clear the centre is known at logging.
- Keep a time-aware split. A random split would score the censored months at the end of the extract as if they were ordinary labels.
