# Dataset card — City of Cape Town SAP C3 service requests

| Field | Entry |
| --- | --- |
| **Hub item (locked URL)** | https://odp-cctegis.opendata.arcgis.com/datasets/90711027d33940b5a06e96ad8a8f7ede_0/explore |
| **Owner** | City of Cape Town |
| **Team extract title** | Service Requests 2023 until 18 August 2026 |
| **Date downloaded (team)** | **8 October 2026, 16:38** — **5,199,661** rows |
| **Date obtained (this VM)** | **15 October 2026** — Drive splits (incl. `train.parquet`); portal fallback 5,372,466 rows |
| **Drive (preferred splits)** | https://drive.google.com/drive/folders/1ca8M2Gjq-sbe7I8wrt_WUyBm8523PwCq |
| **Licence** | City Open Data Terms of Use: free, “as is”, no warranties. See `data/licence.txt`. Lecturer **approved**. Not CC/Zindi. Raw extract not in git. |
| **Collection period** | Created 1 Jan 2023 – 18 Aug 2026 (team file). Modelling dates 1 Jan 2023 – 20 May 2026 after 90-day maturity. |
| **Unit** | One `Notification` after latest-`Changed_on` dedup. |
| **Modelling n** | **3,029,242** (repair/fault scope). Fast **2,041,606** (67.4%); Slow **650,173** (21.5%); Unresolved **337,463** (11.1%). |
| **Main split** | Stratified 70/15/15 on target, seed **42**: train 2,120,469 / val 454,386 / test 454,387. |
| **Time split** | Robustness only (31 May 2025 / 13 Dec 2025). Unresolved jumps 7.4% → ~19.8%. |
| **Target cut** | **7 calendar days** (~p75 of resolved durations; median 2 days not used). |
| **Features** | Complaint type, department, suburb_group, ward, sub-council, created date. |
| **Missingness** | Completed blank 13.0% (open). Train Ward 40.56% missing; Sub_Council 41.55%. Coords ~22% missing, unused. |
| **Sensitive** | No names/contacts. Raw suburb can look like an address — grouped; coords unused. |
| **Limitations** | Reported faults only; closing practice differs by department; freeze is not live SAP. |
