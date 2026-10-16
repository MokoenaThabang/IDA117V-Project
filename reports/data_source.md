# Dataset source

Metadata below was read on **16 October 2026** from the City of Cape Town Open Data Portal and from the feature layer linked by that page. No raw file was saved in this repository. Counts and dates are from those service responses, not from a local table.

| Field | Recorded value |
| --- | --- |
| Dataset owner | City of Cape Town (`orgName` on the Hub dataset; ArcGIS owner `cctgis`) |
| Dataset name in the project plan | SAP C3 Notifications |
| Published title | Service Requests 2023 until  06 October 2026 (Deprecated) |
| Content | Service requests or service notifications stored in the City’s SAP C3 Notifications system |
| Portal | City of Cape Town Open Data Portal |
| Dataset page | https://odp-cctegis.opendata.arcgis.com/datasets/90711027d33940b5a06e96ad8a8f7ede_0/explore |
| Hub dataset id | `90711027d33940b5a06e96ad8a8f7ede_0` |
| ArcGIS item id | `90711027d33940b5a06e96ad8a8f7ede` |
| Item access | `public` |
| Item type | Feature Service |
| Feature layer | https://services6.arcgis.com/nyYfO9SxHU2ChQd9/arcgis/rest/services/Service_Requests_2023_until_20_May_2026/FeatureServer/0 |
| Licence on the item | Custom. The item stores https://www.capetown.gov.za/General/Terms-of-use-open-data |
| Licence page after redirect | https://www.capetown.gov.za/general/city-of-cape-town-open-data-terms-of-use/ |
| Collection period in the project plan | 2023 until 18 September 2026 |
| Created_On_Date range from the layer | 1 January 2023 through 6 October 2026, as interpreted below |
| Portal record count | 5,372,466 (`recordCount`, and the same count from `returnCountOnly`) |
| Portal item size | 1,825,087,488 bytes, as reported by the item. This is not a measured local file size. |
| Item created | 23 May 2026, 11:04 UTC |
| Item modified | 15 October 2026, 10:24 UTC |
| Local raw file | `data/raw/c3_notifications.parquet` (gitignored). 5,372,466 rows, the same count as the feature layer. |
| How the local file was obtained | On 16 October 2026 the Hub CSV export was still generating. The rows were read from the feature layer in ObjectId order and written to parquet. Columns were not renamed. Dates remain UTC epoch milliseconds, as returned by the service. |
| Download date of the local copy | 16 October 2026 |

The repository code is MIT. That licence does not apply to the City dataset.

## Licence

The terms page reached from the item’s licence link is titled “City of Cape Town Open Data terms of use”. On 16 October 2026 it said the City makes the data available without remuneration, on an “as is” basis, disclaims warranties including merchantability and fitness for a particular purpose, and does not warrant completeness, accuracy, reliability, suitability, or availability. It says reliance on the data is at the user’s own risk, and that the City is not liable for errors in the data or for decisions taken in reliance on it. The page title and those statements were read from the HTML of that terms page. This is not a Creative Commons or Zindi licence.

## What was queried

1. ArcGIS item JSON: `https://www.arcgis.com/sharing/rest/content/items/90711027d33940b5a06e96ad8a8f7ede?f=pjson`
2. Hub dataset JSON: `https://odp-cctegis.opendata.arcgis.com/api/v3/datasets/90711027d33940b5a06e96ad8a8f7ede_0`
3. Feature-layer `query` with `returnCountOnly=true` and with `outStatistics` for minimum and maximum of `Created_On_Date`, `Completed_Date`, and `Changed_on`, plus null counts for those three fields.
4. Hub download API for a CSV and a GeoJSON export.

The explore page’s HTML shell did not include the dataset text when fetched without a browser. The Hub record says `access` is `public` and `downloadable` is true.

## Date range

The statistic response returned epoch milliseconds, not a labelled time zone. Those instants are 22:00 UTC, which is midnight in Africa/Johannesburg (UTC+2):

| Field | UTC instant from the service | Johannesburg calendar time |
| --- | --- | --- |
| Minimum `Created_On_Date` | 2022-12-31 22:00 UTC | 2023-01-01 00:00 SAST |
| Maximum `Created_On_Date` | 2026-10-05 22:00 UTC | 2026-10-06 00:00 SAST |
| Minimum `Changed_on` | 2022-12-31 22:00 UTC | 2023-01-01 00:00 SAST |
| Maximum `Changed_on` | 2026-11-05 22:00 UTC | 2026-11-06 00:00 SAST |
| Minimum `Completed_Date` | 1999-12-31 22:00 UTC | 2000-01-01 00:00 SAST |
| Maximum `Completed_Date` | 2026-11-05 22:00 UTC | 2026-11-06 00:00 SAST |

The latest `Created_On_Date` agrees with the published title’s end of 6 October 2026. It is later than the project plan’s end of 18 September 2026. The feature-service name still says `until_20_May_2026`, which is older than both the title and the maximum created date.

`Completed_Date` and `Changed_on` extend past the created-date window. The earliest completion statistic is 1 January 2000. Those values were not removed. Whether they are data-entry errors has not been decided.

Null counts on the same day, out of 5,372,466 rows:

| Field | Null count | Share of rows |
| --- | --- | --- |
| `Created_On_Date` | 0 | 0% |
| `Completed_Date` | 785,115 | 14.61% |
| `Changed_on` | 116,320 | 2.17% |

Percents are the null count divided by 5,372,466, rounded to two decimal places.

## Fields published on the layer

| Field name | Alias | Type |
| --- | --- | --- |
| `Sub_Council` | Sub Council | String |
| `Ward` | Ward | String |
| `Suburb` | Suburb | String |
| `C3_Complaint_Type` | C3 Complaint Type | String |
| `Work_Center` | Work Center | String |
| `Notification` | Notification | Integer |
| `Notification_type` | Notification type | String |
| `X_Y_Co_ordinate_1` | X-Y Co ordinate 1 | Double |
| `X_Y_Co_ordinate_2` | X-Y Co ordinate 2 | Double |
| `Created_On_Date` | Created On Date | Date |
| `Changed_on` | Changed on | Date |
| `Completed_Date` | Completed Date | Date |
| `Notifications_Created` | Notifications Created | Integer |
| `ObjectId` | ObjectId | OID |

`Created On Date` / `Created_On_Date` is known at logging. `Completed Date` / `Completed_Date` may be used later to build the label and must not be a model feature. `Changed on` / `Changed_on` is excluded for the same reason. The other fields are not yet classified as safe or unsafe. `Notification` is an identifier. Coordinates and complaint type need a leakage review after a local file is loaded. No column was renamed.

## Publisher status

The item description says the dataset is deprecated and frozen as of 6 October 2026, and that it is not connected to the live system. A request can be listed as incomplete because it was incomplete when the extract was loaded and may have been completed later.

The same description points to a different public feature service, “Service Requests Jan 2023 to Present”:

https://cctegis.maps.arcgis.com/home/item.html?id=da3b6d46845e4671bde500e8de7e1d3e

That title was checked against the ArcGIS item on 16 October 2026. This project stays on the page that was supplied, which is the deprecated layer.

The description also still contains the unfilled phrases “[MONTH YEAR — e.g. January 2027]” and “[Your name / team email]”, so it does not state a retirement month or a contact address. It includes a “Historic Data” list of years from 2024 back to 2011. That list does not match the `Created_On_Date` minimum returned by the layer, which is 1 January 2023.

## Local copy

The Hub CSV was not ready, so `python -m src.fetch_c3` saved the feature-layer attributes to `data/raw/c3_notifications.parquet` on 16 October 2026. That file is the extract notebook 01 audits. It is gitignored. The CSV endpoint, at 19:59 UTC the same day, had not finished:

The Hub CSV download endpoint returned HTTP 202 on 16 October 2026 at 19:59 UTC:

`https://odp-cctegis.opendata.arcgis.com/api/download/v1/items/90711027d33940b5a06e96ad8a8f7ede/csv?layers=0`

The body said status `Pending` and “Up to date download file is being generated. Please check back again later.” The GeoJSON request at the same time said status `PagingData`, progress 11%, and `recordCount` 605,000. That figure is generation progress, not the layer count of 5,372,466.

The full table was not paged out of the feature service and was not saved under `data/raw/`.

## How to save the file locally

1. Open the dataset page linked above.
2. When the portal offers a download, save the original file without renaming columns or dropping rows.
3. Put that file in `data/raw/`. Prefer CSV.
4. If more than one file is saved, set `DATA_RAW_FILE` in `.env`.
5. Do not commit the file.
6. Re-run `notebooks/01_data_acquisition_and_audit.ipynb` from the top. Write the local file name, format, byte size, and download date into the table at the top of this document, and compare its column names with the field list above.
