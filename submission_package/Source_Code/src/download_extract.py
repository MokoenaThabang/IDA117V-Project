"""Download the City of Cape Town SAP C3 extract via the FeatureServer.

Hub CSV export was pending (async generate). This pager uses ObjectId ranges
so the pull can resume after a drop. Raw files stay local (see data/README.md).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

LAYER_URL = (
    "https://services6.arcgis.com/nyYfO9SxHU2ChQd9/arcgis/rest/services/"
    "Service_Requests_2023_until_20_May_2026/FeatureServer/0"
)
ITEM_URL = (
    "https://odp-cctegis.opendata.arcgis.com/datasets/"
    "90711027d33940b5a06e96ad8a8f7ede_0/explore"
)
PAGE_SIZE = 32_000
TIMEOUT = 120
MAX_RETRIES = 6
RAW_DIR = Path("data/raw")
PARTS_DIR = RAW_DIR / "parts"
MANIFEST = RAW_DIR / "download_manifest.json"
FIELDS = [
    "ObjectId",
    "Sub_Council",
    "Ward",
    "Suburb",
    "C3_Complaint_Type",
    "Work_Center",
    "Notification",
    "Notification_type",
    "X_Y_Co_ordinate_1",
    "X_Y_Co_ordinate_2",
    "Created_On_Date",
    "Changed_on",
    "Completed_Date",
    "Notifications_Created",
]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "IDA117V-student-extract/0.1"})
    return s


def query_json(session: requests.Session, params: dict) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(f"{LAYER_URL}/query", params=params, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload
        except Exception as exc:  # noqa: BLE001 — retry network/service errors
            last_err = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"query failed after retries: {last_err}")


def epoch_ms_to_iso(value):
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def features_to_table(features: list[dict]) -> pa.Table:
    rows = [f.get("attributes", f) for f in features]
    df = pd.DataFrame(rows, columns=FIELDS)
    for col in ("Created_On_Date", "Changed_on", "Completed_Date"):
        df[col] = pd.to_datetime(df[col], unit="ms", utc=True, errors="coerce")
    return pa.Table.from_pandas(df, preserve_index=False)


def already_done() -> set[int]:
    done = set()
    if PARTS_DIR.exists():
        for p in PARTS_DIR.glob("part_*.parquet"):
            try:
                done.add(int(p.stem.split("_")[1]))
            except ValueError:
                continue
    return done


def download(start_id: int | None = None, end_id: int | None = None) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    session = _session()
    count_payload = query_json(
        session, {"where": "1=1", "returnCountOnly": "true", "f": "json"}
    )
    server_count = int(count_payload["count"])
    stats = query_json(
        session,
        {
            "where": "1=1",
            "outStatistics": json.dumps(
                [
                    {
                        "statisticType": "min",
                        "onStatisticField": "ObjectId",
                        "outStatisticFieldName": "min_id",
                    },
                    {
                        "statisticType": "max",
                        "onStatisticField": "ObjectId",
                        "outStatisticFieldName": "max_id",
                    },
                ]
            ),
            "f": "json",
        },
    )
    attrs = stats["features"][0]["attributes"]
    min_id = int(start_id or attrs["min_id"])
    max_id = int(end_id or attrs["max_id"])
    started = datetime.now(timezone.utc).isoformat()
    done = already_done()
    n_written = 0
    lo = min_id
    while lo <= max_id:
        hi = min(lo + PAGE_SIZE - 1, max_id)
        if lo in done:
            lo = hi + 1
            continue
        where = f"ObjectId >= {lo} AND ObjectId <= {hi}"
        payload = query_json(
            session,
            {
                "where": where,
                "outFields": ",".join(FIELDS),
                "returnGeometry": "false",
                "resultRecordCount": PAGE_SIZE,
                "resultType": "standard",
                "f": "json",
            },
        )
        features = payload.get("features", [])
        if features:
            table = features_to_table(features)
            out = PARTS_DIR / f"part_{lo:08d}.parquet"
            pq.write_table(table, out, compression="zstd")
            n_written += table.num_rows
            print(f"wrote {out.name} rows={table.num_rows} cumulative_new={n_written}", flush=True)
        else:
            print(f"empty range {lo}-{hi}", flush=True)
        lo = hi + 1

    local_rows = 0
    for p in sorted(PARTS_DIR.glob("part_*.parquet")):
        local_rows += pq.ParquetFile(p).metadata.num_rows
    manifest = {
        "item_url": ITEM_URL,
        "layer_url": LAYER_URL,
        "download_started_utc": started,
        "download_finished_utc": datetime.now(timezone.utc).isoformat(),
        "server_count": server_count,
        "local_rows": local_rows,
        "objectid_min": min_id,
        "objectid_max": max_id,
        "page_size": PAGE_SIZE,
        "format": "parquet-parts",
        "note": "Frozen Hub item Service Requests 2023 until 06 October 2026 (Deprecated).",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if local_rows != server_count:
        raise SystemExit(
            f"row-count mismatch: local={local_rows} server={server_count}"
        )


if __name__ == "__main__":
    download()
