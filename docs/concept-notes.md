# Concept notes — Cape Town C3 resolution classifier

One page for the pre-modelling checkpoint. Roster, group number and lecturer: **TBD later** (do not invent).

| Field | Our answer |
| --- | --- |
| **Problem** | At logging, City crews cannot see which *repair/fault* C3 jobs will close within a week, which will drag, and which will stay open. |
| **Stakeholder** | City of Cape Town service departments (Water and Sanitation, Electricity, Energy, Solid waste, Public housing, Informal settlements, Open space, Health) that assign crews. |
| **Affected population** | Cape Town residents who report faults and depend on those services. |
| **Prediction unit** | One SAP C3 request (`Notification`) = one row, after duplicate collapse. |
| **Target** | `Resolved Fast` / `Resolved Slow` / `Unresolved`. Fast = completed in **≤ 7 whole days**; Slow = **> 7 days**; Unresolved = still open after maturity rules. |
| **Cut justification** | From the **resolution-time distribution**: 75.8% of resolved jobs finish within 7 days (median 2 days was rejected as harder to explain). Not a leftover placeholder. |
| **Maturity** | Drop open tickets **&lt; 7 days** old; then drop **all** tickets created in the last **90 days** before export (after 20 May 2026 on the team extract). |
| **Intended decision** | Surface likely Slow/Unresolved jobs at logging so dispatchers look there first. Not an instruction to close or refuse a ticket. |
| **Public value** | Earlier attention to request types, suburbs and departments that actually drag or stay open. |
| **Horizon / leakage** | Features known at logging only. See `src/prep.py` `BANNED`. |
| **Data** | City Open Data, lecturer-approved. https://odp-cctegis.opendata.arcgis.com/datasets/90711027d33940b5a06e96ad8a8f7ede_0/explore |
| **Size** | Raw **5,199,661**. Modelling **3,029,242** (67.4% Fast / 21.5% Slow / 11.1% Unresolved). Seed **42**: train 2,120,469 / val 454,386 / test 454,387. |
| **Metric** | Macro-F1; also Slow/Unresolved recall. FN (predict Fast, truly Slow/Unresolved) is the costly error. |
| **UI later** | Streamlit, not a notebook. Written due **20 October 2026**. Presentation slot **not yet announced**. |
