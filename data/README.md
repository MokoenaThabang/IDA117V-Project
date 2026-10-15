# Data — how to obtain (do not commit the raw C3 extract)

Licence: City of Cape Town [Open Data Terms of Use](https://www.capetown.gov.za/general/city-of-cape-town-open-data-terms-of-use/) — free, “as is”, no warranties. Verbatim copy: `data/licence.txt`. Download recorded **15 October 2026** (this VM) and **8 October 2026 16:38** (team extract). **Not** Creative Commons / Zindi. Lecturer approved this municipal source.

Modelling parquets are **~47 MB together** but still City operational data. They stay **out of git**. Place them in `data/processed/`.

## Preferred: Google Drive (team splits, including `train.parquet`)

Folder (anyone with the link):  
https://drive.google.com/drive/folders/1ca8M2Gjq-sbe7I8wrt_WUyBm8523PwCq

```bash
python -m pip install gdown
python -m gdown --folder "https://drive.google.com/drive/folders/1ca8M2Gjq-sbe7I8wrt_WUyBm8523PwCq" -O data/processed/drive/
mkdir -p data/processed
cp data/processed/drive/117V/*.parquet data/processed/   # or whatever subfolder gdown creates
# required: train.parquet val.parquet test.parquet
# optional robustness: time_train.parquet time_val.parquet time_test.parquet
```

Also copy `prep.py` and `app_choices.json` from that folder if this repo copy is missing.

## Fallback: Open Data Portal extract

Exact item:  
https://odp-cctegis.opendata.arcgis.com/datasets/90711027d33940b5a06e96ad8a8f7ede_0/explore

Hub CSV export may sit in `PagingData` for hours. This repo pages the FeatureServer instead:

```bash
python src/download_extract.py
# writes data/raw/parts/*.parquet (gitignored). n was 5,372,466 on 15 Oct 2026.
```

Do **not** rebuild splits from the portal dump unless the Drive files are unavailable. Target, 7-day cut, maturity rules and seed **42** are defined in `src/prep.py` plus the team governance note (Sections 3 and 5).

## What each processed file is

| File | Role | n (team Drive, 15 Oct 2026) |
| --- | --- | ---: |
| `train.parquet` | Main train (stratified 70%, seed 42) | 2,120,469 |
| `val.parquet` | Main validation | 454,386 |
| `test.parquet` | **Locked** test | 454,387 |
| `time_*.parquet` | Robustness split (class mix drifts) | 3,029,242 total |

Columns: `C3_Complaint_Type`, `department`, `suburb_group`, `Ward`, `Sub_Council`, `Created_On_Date`, `target`.

`data/sample/anonymised_sample.csv` is a 2,000-row draw from train (seed 42): no Notification IDs, no coordinates, no raw street-style suburb text.
