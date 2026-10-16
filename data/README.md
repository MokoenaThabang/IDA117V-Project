# Data

Keep this tree empty of extracts. Git tracks the folders; it must not track the City C3 dump or modelling parquets.

## How to obtain

1. **Team Drive (preferred splits)**  
   https://drive.google.com/drive/folders/1ca8M2Gjq-sbe7I8wrt_WUyBm8523PwCq  
   Download into `data/raw/` on your machine.

2. **City of Cape Town Open Data Hub (authoritative source)**  
   https://odp-cctegis.opendata.arcgis.com/datasets/90711027d33940b5a06e96ad8a8f7ede_0/explore  
   Hub item: Service Requests 2023 until 06 October 2026 (Deprecated).

## Licence

City of Cape Town [Open Data Terms of Use](https://www.capetown.gov.za/general/city-of-cape-town-open-data-terms-of-use/). Free, “as is”, no warranties; the City is not liable for decisions made on the data. **Not** Creative Commons or Zindi.

**Do not commit the raw extract.** `data/raw/` is gitignored except for `.gitkeep`.

## Layout

| Path | Purpose |
| --- | --- |
| `data/raw/` | Local downloads only (empty in git) |
| Dataset card / data dictionary | Written later with Q1–Q2, not in this scaffold |
