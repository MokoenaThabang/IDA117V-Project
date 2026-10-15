# EDA interpretations (train only, n=2,120,469)

## fig01_target_distribution.png

Fast 1,429,124 (67.4%),
Slow 455,121 (21.5%),
Unresolved 236,224 (11.1%).
Macro-F1 is required; a majority dummy will look strong on accuracy.

## fig02_missingness.png

Ward is missing on 40.6% of train rows and Sub_Council on
41.6%. The pipeline maps those to MISSING rather than
imputing the mode. Coordinates are unused. Raw suburb streets never appear.

## fig03_department.png

Department (from work centre) is known at logging and changes the Fast/Slow/Unresolved
mix. Public housing and open space run much higher Unresolved rates in the audit —
partly ticket-closing practice, not only delay. This is an EDA chart, not a second model.

## fig04_complaint_type.png

Complaint type is the operational code at logging. Volume is dominated by power,
sewer, street-light and bin faults. The dummy ignores this mix; later models must beat it.

## fig05_temporal_train.png

The main split is stratified random, so train still covers 1 Jan 2023 – 20 May 2026.
Time drift is checked on the separate time split, not by peeking at test.
