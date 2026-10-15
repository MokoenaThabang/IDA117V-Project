# Data dictionary (modelling table)

Derived from the team governance draft. Raw portal columns are 14 fields; the model sees the seven below.

| Column | Type | Meaning | Use |
| --- | --- | --- | --- |
| `C3_Complaint_Type` | text | Fault / request type | Feature. Types with &lt;100 **train** rows grouped infrequent in `prep.py`. |
| `department` | text | 9 groups from `Work_Center` | Feature. Raw work centre is banned. |
| `suburb_group` | text | Suburb names with ≥500 **train** rows; else OTHER; `#` mapped to MISSING in the pipeline | Feature. Raw `Suburb` banned (can contain street-style text). |
| `Ward` | text | Ward `"1"`–`"116"` | Feature. Placeholders (`#`, `000`, …) → missing → MISSING. |
| `Sub_Council` | text | `Sub-Council 1`–`24` | Feature. Same missing handling. |
| `Created_On_Date` | datetime | Log date (Africa/Johannesburg date in the splits) | Feature source for month, weekday, weekend. Then dropped. |
| `target` | text | `Resolved Fast` / `Resolved Slow` / `Unresolved` | Label. Fast = completed ≤ **7** days; Slow = completed > 7 days; Unresolved = open after 7-day + 90-day maturity. **Never a feature.** |

**Banned (leakage / sensitive):** `Completed_Date`, `Changed_on`, `Notification_type`, `Notifications_Created`, `resolution_days`, `target`, raw `Suburb`, `Work_Center`, `Notification`. Enforced by `src/prep.py`.

**Not used:** X-Y coordinates (~22% missing raw).
