# IDA117V — Cape Town C3 resolution classifier

Group project for **IDA117V (Introduction to Data Science, S2 2026)**  
Theme: *Data Science for Better Service Delivery in South Africa*

At the moment a City of Cape Town SAP C3 service request is logged, classify likely resolution status:

**Resolved Fast / Resolved Slow / Unresolved**

Stakeholder: City of Cape Town service departments that assign repair crews. This repository is a **structure-only scaffold**. Question notebooks, models, and the Streamlit app are added later.

## How to obtain the data

Do **not** commit the raw extract or large modelling parquets.

1. **Preferred (team splits):** Google Drive folder  
   https://drive.google.com/drive/folders/1ca8M2Gjq-sbe7I8wrt_WUyBm8523PwCq  
   Place downloaded files under `data/raw/` (gitignored).
2. **Authoritative source (City Open Data Hub):**  
   [Service Requests 2023 until 06 October 2026 (Deprecated)](https://odp-cctegis.opendata.arcgis.com/datasets/90711027d33940b5a06e96ad8a8f7ede_0/explore)  
   https://odp-cctegis.opendata.arcgis.com/datasets/90711027d33940b5a06e96ad8a8f7ede_0/explore

**Licence:** City of Cape Town [Open Data Terms of Use](https://www.capetown.gov.za/general/city-of-cape-town-open-data-terms-of-use/) — free, “as is”, no warranties. This is **not** Creative Commons or Zindi. Record the download date in the dataset card when that artefact is written. **Do not commit the raw extract.**

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch is optional and commented in `requirements.txt` until neural models are added.

## Seeds

Use **42** for every split, sampler, and estimator (`src/config.py`). Do not change the seed without a team decision.

## Execution order

Placeholders only. Fill each step when the matching question is approved.

| Step | Question | Artefact | Status |
| --- | --- | --- | --- |
| 1 | Q1 — Problem, stakeholder, data & ethics | `notebooks/q1.ipynb` | placeholder |
| 2 | Q2 — Data audit, EDA, visualisation & pipeline | `notebooks/q2.ipynb` | placeholder |
| 3 | Q3 — Baseline & models | `notebooks/q3.ipynb` | placeholder |
| 4 | Q4 — Evaluation, comparison & overfitting | `notebooks/q4.ipynb` | placeholder |
| 5 | Q5 — Champion optimisation | `notebooks/q5.ipynb` | placeholder |
| 6 | Q6 — Ablation & McNemar | `notebooks/q6.ipynb` | placeholder |
| 7 | Q7 — User-interface deployment | `app/` (Streamlit) | placeholder |
| 8 | Q8 — Conclusions, reproducibility & team defence | `notebooks/q8.ipynb`, `reports/` | placeholder |

Saved pipeline (later): `models/`  
Figures and tables (later): `reports/`

## Layout

See the folder list in this repository. `src/config.py` holds the seed and banned column names only — no training code.

## Licence (code)

Repository code is MIT (see `LICENSE`). Dataset terms are the City Open Data Terms of Use, which are separate from the code licence.
