# Phase 2 feasibility

The raw extract has 5372466 rows and 14 columns.
Created On Date runs from 2023-01-01 to 2026-10-06.
The observation cutoff is 2026-10-06, the latest created date in the file.
Exact business duplicates removed: 779663.
Before the duplicate rule, 709092 notification ids repeat across 1655615 rows.
After exact copies are removed, 142898 ids still repeat across 309758 rows. Those rows are kept.
Rows after that rule: 4592803.
Labelled rows: 4576802.
Resolved Fast: 2969242.
Resolved Slow: 937145.
Unresolved: 670415.
Excluded for completion before creation: 20.
Excluded for insufficient observation time: 15981.
Excluded for malformed or missing created dates: 0.
Excluded for malformed completed dates: 0.
Completed durations over 365 days retained: 14953.
Median completed duration: 2.0 days.
75th percentile of completed duration: 7.0 days.

The extract can support a three-class model. Resolved Fast is the largest labelled class. Resolved Slow and Unresolved both have hundreds of thousands of rows.

Limitations:
- The published layer is deprecated and was not connected to the live system. An open request may have been completed after the extract was taken.
- The project plan ends on 18 September 2026. Created dates in this file continue through 6 October 2026.
- Seven days is provisional. It matches the 75th percentile of completed durations in this file, but it is not a citywide resolution standard.
- Completions dated after the cutoff are not treated as completed. Some completion dates also fall in the year 2000.
- Coordinates are often missing or outside the local metre envelope, and they are withheld from the analysis file.
- Notification type is not a clean category and is withheld.
- Suburb has tens of thousands of distinct values. The most common stored value is the placeholder '#', and the first ObjectId rows use GENERIC TOWN 0000. It needs grouping before modelling.
- Work centre may be assigned after logging. It is left in the feature list with that warning.
- No model was trained in this notebook.
