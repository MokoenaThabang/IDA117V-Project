# Dataset source

This file records what is known about the municipal extract used for the Cape Town service-request resolution classifier. Values that were not in the workspace are left blank on purpose.

| Field | Recorded value |
| --- | --- |
| Dataset owner | City of Cape Town |
| Dataset name | SAP C3 Notifications |
| Content | Municipal service requests |
| Portal | City of Cape Town Open Data Portal |
| Exact download URL | Not available in this workspace. Copy it from the portal page when the file is downloaded. |
| Download mechanism in the workspace | None. There is no script, token, or cached file that can fetch the extract. |
| Licence | Not available in this workspace. Copy the licence or terms of use shown on the portal page. Do not assume Creative Commons, Zindi, or the repository MIT licence. |
| Collection period (project plan) | 2023 until 18 September 2026 |
| Collection start | The plan states the year 2023. It does not state the start day or month. |
| Collection end (project plan) | 18 September 2026. This has not been checked against a downloaded file. |
| Download date | Not downloaded. |
| Local path | `data/raw/` (no extract is committed) |
| File name and format | Unknown until the file is saved. |

The repository code is MIT. That licence does not apply to the City dataset.

## Manual download

The workspace does not contain the dataset page URL. Use the portal search below rather than a guessed link.

1. Open the City of Cape Town Open Data Portal.
2. Search for **SAP C3 Notifications** covering service requests from **2023 until 18 September 2026**.
3. On the dataset page, confirm the owner is the City of Cape Town and the collection period matches the project plan before downloading.
4. Record the page URL, the licence or terms of use, and the calendar download date in the table above.
5. Save the original file, without renaming columns or dropping rows, in `data/raw/`.
6. If the portal provides more than one file, keep each original and set `DATA_RAW_FILE` in `.env` to the file the audit should read.
7. Do not commit the raw file.

After the file is in place, re-run `notebooks/01_data_acquisition_and_audit.ipynb` from the top. The notebook prints the real column names, row count, and missingness. Write those figures into this document from that output.

## Limitations known before the file is loaded

- The prediction is what is known when the request is logged. `Completed Date` may be used to build the label and must not be a model feature. `Changed on` may show later updates and is also excluded.
- Open requests logged near the end of the extract can look unresolved only because little time has passed. The observation window is configurable and is **not set** in Phase 1.
- The fast/slow day boundary is configurable and is **not set** in Phase 1. It must be chosen from the resolution-time distribution after the file is audited, using training data only.
- Other fields may encode the outcome or appear only after logging. Their names are not listed here, because the extract is not in the workspace.
- Raw files stay unmodified. Cleaning belongs in interim or processed data.

## What to fill in after download

- Exact source URL.
- Licence name and any link printed on the portal.
- Download date.
- File name, format, and file size.
- Actual column names and the collection start and end dates computed from the file.
- Any difference between the file and the planned period "2023 until 18 September 2026".
