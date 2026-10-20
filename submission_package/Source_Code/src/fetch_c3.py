"""Download the published C3 feature layer into data/raw.

The Hub CSV export was still generating when this was written. This module
reads the public feature layer in ObjectId pages and writes a parquet file.
It does not print row values. The raw file stays gitignored.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.config import FEATURE_LAYER_URL, RAW_DATA_DIR

QUERY_URL = FEATURE_LAYER_URL.rstrip("/") + "/query"
RAW_PARQUET_NAME = "c3_notifications.parquet"
OID_START = 1
OID_END = 5_373_162
PAGE_SIZE = 1000

SCHEMA = pa.schema(
    [
        ("Sub_Council", pa.string()),
        ("Ward", pa.string()),
        ("Suburb", pa.string()),
        ("C3_Complaint_Type", pa.string()),
        ("Work_Center", pa.string()),
        ("Notification", pa.int64()),
        ("Notification_type", pa.string()),
        ("X_Y_Co_ordinate_1", pa.float64()),
        ("X_Y_Co_ordinate_2", pa.float64()),
        ("Created_On_Date", pa.int64()),
        ("Changed_on", pa.int64()),
        ("Completed_Date", pa.int64()),
        ("Notifications_Created", pa.int64()),
        ("ObjectId", pa.int64()),
    ]
)
COLUMNS = [field.name for field in SCHEMA]


def _query(where: str, attempts: int = 5) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "where": where,
            "outFields": ",".join(COLUMNS),
            "returnGeometry": "false",
            "orderByFields": "ObjectId",
            "resultRecordCount": str(PAGE_SIZE),
            "f": "json",
        }
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(QUERY_URL + "?" + params, timeout=120) as response:
                payload = json.load(response)
            if "error" in payload:
                raise RuntimeError(str(payload["error"]))
            return [feature["attributes"] for feature in payload.get("features", [])]
        except Exception as exc:  # noqa: BLE001 — retry transient portal errors
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Query failed for {where}: {last_error}")


def _column(rows: list[dict], name: str, kind: str) -> list:
    values = []
    for row in rows:
        value = row.get(name)
        if value is None or value == "":
            values.append(None)
            continue
        if kind == "int":
            values.append(int(value))
        elif kind == "float":
            values.append(float(value))
        else:
            values.append(str(value))
    return values


def _table(rows: list[dict]) -> pa.Table:
    kind = {
        "Notification": "int",
        "X_Y_Co_ordinate_1": "float",
        "X_Y_Co_ordinate_2": "float",
        "Created_On_Date": "int",
        "Changed_on": "int",
        "Completed_Date": "int",
        "Notifications_Created": "int",
        "ObjectId": "int",
    }
    arrays = []
    for field in SCHEMA:
        arrays.append(pa.array(_column(rows, field.name, kind.get(field.name, "str")), type=field.type))
    return pa.table(arrays, schema=SCHEMA)


def _download_span(span: tuple[int, int, Path]) -> tuple[int, int]:
    start, end, part_path = span
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    oid = start
    try:
        while oid <= end:
            upper = min(oid + PAGE_SIZE - 1, end)
            rows = _query(f"ObjectId >= {oid} AND ObjectId <= {upper}")
            if rows:
                table = _table(rows)
                if writer is None:
                    writer = pq.ParquetWriter(part_path, SCHEMA, compression="zstd")
                writer.write_table(table)
                rows_written += table.num_rows
            oid = upper + 1
    finally:
        if writer is not None:
            writer.close()
    return start, rows_written


def download_published_layer(
    destination: Path | None = None,
    workers: int = 6,
) -> Path:
    """Save the feature layer to data/raw and return the parquet path."""
    target = RAW_DATA_DIR / RAW_PARQUET_NAME if destination is None else destination
    target.parent.mkdir(parents=True, exist_ok=True)
    part_dir = target.parent / "_c3_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    width = OID_END - OID_START + 1
    span_size = (width + workers - 1) // workers
    spans = []
    for index in range(workers):
        start = OID_START + index * span_size
        end = min(OID_END, start + span_size - 1)
        if start <= end:
            spans.append((start, end, part_dir / f"part_{index:02d}.parquet"))
    print(f"Downloading ObjectId {OID_START}..{OID_END} with {len(spans)} workers", flush=True)
    total = 0
    with ThreadPoolExecutor(max_workers=len(spans)) as pool:
        futures = [pool.submit(_download_span, span) for span in spans]
        for future in as_completed(futures):
            start, count = future.result()
            total += count
            print(f"worker start {start} rows {count} cumulative {total}", flush=True)
    parts = sorted(part_dir.glob("part_*.parquet"))
    if not parts:
        raise RuntimeError("The feature layer download produced no rows.")
    writer = pq.ParquetWriter(target, SCHEMA, compression="zstd")
    try:
        for part in parts:
            writer.write_table(pq.read_table(part))
    finally:
        writer.close()
    for part in parts:
        part.unlink()
    part_dir.rmdir()
    print(f"Wrote {target} rows {total}", flush=True)
    return target


def main() -> int:
    download_published_layer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
