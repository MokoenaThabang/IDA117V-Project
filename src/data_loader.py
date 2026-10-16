"""Read-only access to the raw service-request extract.

The loader does not download data and does not rename columns. If the file
is missing, it stops with manual instructions instead of creating a substitute.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, RAW_DATA_FILE, PROJECT_ROOT

# Formats this phase knows how to read. Excel is omitted because openpyxl
# is not a project dependency; export a CSV if the portal provides a workbook.
SUPPORTED_SUFFIXES = {".csv", ".parquet", ".json"}

MANUAL_DOWNLOAD_INSTRUCTIONS = """
No raw City of Cape Town extract is available to this project yet.

The project plan names the dataset, but this workspace does not contain an
exact download URL or a working download mechanism. Do not invent one.

1. Open the City of Cape Town Open Data Portal.
2. Search for SAP C3 Notifications: service requests from 2023 until 18 September 2026.
3. On the dataset page, confirm the owner and the collection period before downloading.
4. Copy the page URL, the licence or terms shown there, and the download date into reports/data_source.md.
5. Save the original file, unchanged, in data/raw/.
6. If you download more than one file, set DATA_RAW_FILE in .env to the file this audit should read.
7. Do not commit the raw file. data/raw/ is gitignored except for .gitkeep.

Then re-run the audit from the top.
""".strip()


class RawDataNotFoundError(FileNotFoundError):
    """Raised when data/raw has no supported extract."""


def manual_download_instructions() -> str:
    """Return the steps for obtaining the extract without a hard-coded URL."""
    return MANUAL_DOWNLOAD_INSTRUCTIONS


def list_raw_files(directory: Path | None = None) -> list[Path]:
    """List supported files in the raw directory. Hidden files are skipped."""
    raw_dir = RAW_DATA_DIR if directory is None else directory
    if not raw_dir.is_dir():
        return []
    files = [
        path
        for path in raw_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=lambda path: path.name.lower())


def resolve_raw_file() -> Path:
    """Choose the raw file, or raise if the choice is missing or ambiguous."""
    if RAW_DATA_FILE:
        path = Path(RAW_DATA_FILE).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise RawDataNotFoundError(
                f"DATA_RAW_FILE does not point to a file: {path}\n\n"
                f"{MANUAL_DOWNLOAD_INSTRUCTIONS}"
            )
        return path

    files = list_raw_files()
    if not files:
        raise RawDataNotFoundError(MANUAL_DOWNLOAD_INSTRUCTIONS)
    if len(files) > 1:
        names = ", ".join(path.name for path in files)
        raise ValueError(
            "More than one raw file is present "
            f"({names}). Set DATA_RAW_FILE in .env to the file that should be audited. "
            "The loader will not concatenate or choose one on its own."
        )
    return files[0]


def load_raw_table(path: Path | None = None) -> pd.DataFrame:
    """Load the raw table. The file on disk is only read, never written."""
    source = resolve_raw_file() if path is None else path
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(
            f"Unsupported raw file type {suffix!r} for {source.name}. "
            f"Supported types: {supported}. Leave the raw file unchanged."
        )
    if suffix == ".csv":
        try:
            return pd.read_csv(source)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Could not read {source.name} as UTF-8. "
                "Save a UTF-8 copy outside data/raw, or record the encoding "
                "before any conversion. Do not overwrite the original download."
            ) from exc
    if suffix == ".json":
        return pd.read_json(source)
    try:
        return pd.read_parquet(source)
    except ImportError as exc:
        raise ImportError(
            "Reading a parquet extract needs pyarrow. It is not pinned directly; "
            "the Streamlit dependency normally installs it. Export CSV if parquet "
            "cannot be read, and keep the original download unchanged."
        ) from exc


def audit_frame(frame: pd.DataFrame) -> dict:
    """Summarise a loaded frame. Counts come only from that frame."""
    missing_count = {str(column): int(frame[column].isna().sum()) for column in frame.columns}
    row_count = int(len(frame))
    missing_percent = {
        column: (count / row_count * 100.0 if row_count else 0.0)
        for column, count in missing_count.items()
    }
    return {
        "row_count": row_count,
        "column_count": int(frame.shape[1]),
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_count": missing_count,
        "missing_percent": missing_percent,
    }


def main() -> int:
    """Print either the raw-file audit or the manual download steps."""
    try:
        path = resolve_raw_file()
    except (RawDataNotFoundError, ValueError) as exc:
        print(exc)
        return 1
    frame = load_raw_table(path)
    summary = audit_frame(frame)
    print(f"Raw file: {path}")
    print(f"Rows: {summary['row_count']}")
    print(f"Columns ({summary['column_count']}):")
    for name in summary["columns"]:
        dtype = summary["dtypes"][name]
        missing = summary["missing_count"][name]
        print(f"  - {name} ({dtype}), missing {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
