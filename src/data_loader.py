"""Read-only access to the raw service-request extract.

The loader does not download data and does not rename columns. If the file
is missing, it stops with manual instructions instead of creating a substitute.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    FAST_SLOW_THRESHOLD_DAYS,
    LOCAL_GRID_X_MAX,
    LOCAL_GRID_X_MIN,
    LOCAL_GRID_Y_MAX,
    LOCAL_GRID_Y_MIN,
    OBSERVATION_WINDOW_DAYS,
    PRECISE_LOCATION_COLUMNS,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
    TABLES_DIR,
    UNUSUAL_DURATION_DAYS,
)
from src.preprocessing import (
    build_resolution_target,
    date_problem_masks,
    require_date_columns,
    select_model_features,
)

# CSV and Excel are the manual-download formats. Parquet is also accepted
# because the feature-service copy is stored that way. JSON remains readable.
SUPPORTED_SUFFIXES = {".csv", ".parquet", ".json", ".xlsx", ".xls"}

MANUAL_DOWNLOAD_INSTRUCTIONS = """
No raw file is in data/raw yet.

Dataset page:
https://odp-cctegis.opendata.arcgis.com/datasets/90711027d33940b5a06e96ad8a8f7ede_0/explore

Published title: Service Requests 2023 until  06 October 2026 (Deprecated)
Portal owner: City of Cape Town
Licence link on the item: https://www.capetown.gov.za/General/Terms-of-use-open-data

On 16 October 2026 the portal CSV export was still being generated, so this
repository has no local copy. Counts, fields, and date limits already read
from the feature service are in reports/data_source.md.

1. Open the dataset page above and download the original file when the portal offers it.
2. Save that file, unchanged, in data/raw/. CSV is the preferred format.
3. If you save more than one file, set DATA_RAW_FILE in .env to the file this audit should read.
4. Do not commit the raw file. data/raw/ is gitignored except for .gitkeep.
5. Do not trim the raw file to the project-plan end date of 18 September 2026.
   The published Created On Date values continue through 6 October 2026.

Then re-run the audit from the top.
""".strip()


class RawDataNotFoundError(FileNotFoundError):
    """Raised when data/raw has no supported extract."""


def manual_download_instructions() -> str:
    """Return the steps for saving the portal file into data/raw."""
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
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(source)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel needs openpyxl (for .xlsx). It is listed in requirements.txt. "
                "Old .xls files are not supported; save the sheet as .xlsx or CSV "
                "without changing the raw download."
            ) from exc
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


# Meanings and proposed uses for columns that appear on the published layer.
# Keys are case-folded with spaces replaced by underscores.
_COLUMN_GUIDE = {
    "sub_council": (
        "Subcouncil recorded on the request.",
        "Candidate predictor. Coarse area, not a precise point.",
    ),
    "ward": (
        "Ward recorded on the request.",
        "Candidate predictor. Coarse area, not a precise point.",
    ),
    "suburb": (
        "Suburb recorded on the request.",
        "Candidate predictor. Coarse area, not a precise point.",
    ),
    "c3_complaint_type": (
        "Complaint or request category.",
        "Candidate predictor if it is known when the request is logged.",
    ),
    "work_center": (
        "Work centre named on the request.",
        "Uncertain predictor. Use only if the centre is known at logging rather than assigned later.",
    ),
    "notification": (
        "Service-request identifier.",
        "Identifier. Not a model input.",
    ),
    "notification_type": (
        "Mixed field. On the published layer many rows repeat the notification id and other rows hold free text.",
        "Not a model input. Do not print the free-text values.",
    ),
    "x_y_co_ordinate_1": (
        "First coordinate attribute.",
        "Precise location. Not a model input and not shown in tables or figures.",
    ),
    "x_y_co_ordinate_2": (
        "Second coordinate attribute.",
        "Precise location. Not a model input and not shown in tables or figures.",
    ),
    "created_on_date": (
        "Date the request was logged.",
        "Known at logging. Used for the observation cutoff and allowed as a time feature.",
    ),
    "changed_on": (
        "Date of a later change to the request.",
        "Post-logging field. Not a model input.",
    ),
    "completed_date": (
        "Date the request was completed, when it has been completed.",
        "Used only to build the label. Not a model input.",
    ),
    "notifications_created": (
        "Count stored in Notifications Created. On the published layer almost every row is 1.",
        "Low-variance candidate predictor. Not used to build the label and not an identifier.",
    ),
    "objectid": (
        "Feature-service row identifier.",
        "System identifier. Not a model input. Ignored when checking business duplicates.",
    ),
}

_SENSITIVE_NAMES = {name.casefold() for name in (*PRECISE_LOCATION_COLUMNS, "Notification", "Notification_type", "ObjectId")}


def _guide_key(column: str) -> str:
    return str(column).casefold().replace(" ", "_")


def public_columns(columns) -> list[str]:
    """Columns safe to print. Coordinates, identifiers, and free text are removed."""
    return [str(column) for column in columns if str(column).casefold() not in _SENSITIVE_NAMES]


def representative_rows(frame: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
    """Return the first rows with sensitive columns removed."""
    columns = public_columns(frame.columns)
    return frame.loc[:, columns].head(rows).copy()


def data_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe the actual columns. Meanings are blank when the name is unknown."""
    row_count = int(len(frame))
    records = []
    for column in frame.columns:
        meaning, proposed = _COLUMN_GUIDE.get(
            _guide_key(str(column)),
            ("No description is recorded for this column.", "Review the values before using it."),
        )
        missing = int(frame[column].isna().sum())
        records.append(
            {
                "column": str(column),
                "pandas_dtype": str(frame[column].dtype),
                "meaning": meaning,
                "missing_count": missing,
                "missing_percent": round((missing / row_count * 100.0) if row_count else 0.0, 4),
                "distinct_values": int(frame[column].nunique(dropna=True)),
                "proposed_use": proposed,
            }
        )
    return pd.DataFrame(records)


def _find_column(columns, *names: str) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in columns}
    for name in names:
        found = lookup.get(name.casefold())
        if found is not None:
            return found
    return None


def duplicate_report(frame: pd.DataFrame) -> dict[str, int]:
    """Count duplicate rows and repeated notification ids. Nothing is deleted."""
    business_columns = [column for column in frame.columns if str(column).casefold() != "objectid"]
    report = {
        "exact_duplicate_rows_including_objectid": int(frame.duplicated().sum()),
        "exact_business_duplicate_rows": int(frame.duplicated(subset=business_columns).sum()),
        "notification_ids_with_repeats": 0,
        "rows_in_repeated_notification_ids": 0,
    }
    notification = _find_column(frame.columns, "Notification")
    if notification is None:
        return report
    counts = frame[notification].value_counts(dropna=True)
    repeated = counts[counts > 1]
    report["notification_ids_with_repeats"] = int(len(repeated))
    report["rows_in_repeated_notification_ids"] = int(repeated.sum())
    return report


def _coordinate_quality(frame: pd.DataFrame) -> pd.Series:
    """Flag coordinate pairs. Individual coordinate values are not returned."""
    x_name = _find_column(frame.columns, "X_Y_Co_ordinate_1")
    y_name = _find_column(frame.columns, "X_Y_Co_ordinate_2")
    quality = pd.Series("not_present", index=frame.index, dtype="object")
    if x_name is None or y_name is None:
        return quality
    x_values = pd.to_numeric(frame[x_name], errors="coerce")
    y_values = pd.to_numeric(frame[y_name], errors="coerce")
    missing = x_values.isna() | y_values.isna()
    plausible = x_values.between(LOCAL_GRID_X_MIN, LOCAL_GRID_X_MAX) & y_values.between(
        LOCAL_GRID_Y_MIN, LOCAL_GRID_Y_MAX
    )
    quality.loc[~missing] = "outside_local_grid"
    quality.loc[missing] = "missing"
    quality.loc[plausible] = "plausible_local_grid"
    return quality


def build_analysis_dataset(
    frame: pd.DataFrame,
    threshold_days: int | None = None,
    observation_days: int | None = None,
    reference_date=None,
) -> tuple[pd.DataFrame, dict]:
    """Apply the documented cleaning rules and attach the resolution label.

    Exact copies of the business columns, ignoring ObjectId, are collapsed
    to the first row. Repeated notification ids that are not exact copies
    are kept. Completed Date, Changed on, coordinates, and Notification type
    are omitted from the returned table. Resolution duration is stored for
    the audit and is blocked from model features.
    """
    if threshold_days is None:
        threshold_days = FAST_SLOW_THRESHOLD_DAYS
    if observation_days is None:
        observation_days = OBSERVATION_WINDOW_DAYS
    roles = require_date_columns(frame)
    created_name = str(roles["created_on_date"])
    completed_name = str(roles["completed_date"])
    changed_name = roles["changed_on"]

    business_columns = [column for column in frame.columns if str(column).casefold() != "objectid"]
    exact_with_id = int(frame.duplicated().sum())
    business_duplicate = frame.duplicated(subset=business_columns, keep="first")
    removed_duplicates = int(business_duplicate.sum())
    working = frame.loc[~business_duplicate].copy()

    created, _missing_created, _malformed_created = date_problem_masks(working[created_name])
    if reference_date is None:
        if not bool(created.notna().any()):
            raise ValueError(
                "No Created On Date could be parsed, so the observation cutoff cannot be set. "
                f"Actual columns: {', '.join(map(str, frame.columns))}."
            )
        reference_date = created.max()

    labels = build_resolution_target(
        working,
        created_column=created_name,
        completed_column=completed_name,
        fast_slow_threshold_days=threshold_days,
        observation_window_days=observation_days,
        reference_date=reference_date,
    )

    changed_before_created = 0
    malformed_changed = 0
    if changed_name is not None:
        changed, _missing_changed, malformed_changed_mask = date_problem_masks(working[changed_name])
        changed_before_created = int(
            (
                changed.notna()
                & created.notna()
                & changed.dt.normalize().lt(created.dt.normalize())
            ).sum()
        )
        malformed_changed = int(malformed_changed_mask.sum())

    notification_name = _find_column(working.columns, "Notification")
    type_name = _find_column(working.columns, "Notification_type")
    id_repeated = pd.Series(False, index=working.index)
    repeated_id_count = 0
    rows_in_repeated = 0
    type_matches_id = pd.Series(pd.NA, index=working.index, dtype="boolean")
    if notification_name is not None:
        value_counts = working[notification_name].value_counts(dropna=True)
        repeated_values = value_counts[value_counts > 1].index
        id_repeated = working[notification_name].isin(repeated_values)
        repeated_id_count = int(len(repeated_values))
        rows_in_repeated = int(id_repeated.sum())
        if type_name is not None:
            type_matches_id = (
                working[type_name].astype("string").str.strip()
                == working[notification_name].astype("string").str.strip()
            )

    coordinate_quality = _coordinate_quality(working)
    created_min = created.min()
    created_max = created.max()
    completed, _c_missing, _c_bad = date_problem_masks(working[completed_name])

    drop_names = [
        name
        for name in (completed_name, changed_name, type_name, _find_column(working.columns, *PRECISE_LOCATION_COLUMNS))
        if name is not None and name in working.columns
    ]
    # _find_column with multiple names returns only one. Drop both coordinate columns explicitly.
    for coordinate_name in PRECISE_LOCATION_COLUMNS:
        found = _find_column(working.columns, coordinate_name)
        if found is not None and found not in drop_names:
            drop_names.append(found)
    output = working.drop(columns=drop_names)
    output[created_name] = created
    output["resolution_outcome"] = labels["resolution_outcome"].to_numpy()
    output["resolution_days"] = labels["resolution_days"].to_numpy()
    output["label_status"] = labels["label_status"].to_numpy()
    output["duration_flag"] = labels["duration_flag"].to_numpy()
    output["coordinate_quality"] = coordinate_quality.to_numpy()
    output["notification_id_repeated"] = id_repeated.to_numpy()
    output["notification_type_matches_id"] = type_matches_id.to_numpy()

    completed_duration = labels["resolution_days"]
    quantile_source = completed_duration.dropna()
    quantiles = {}
    if not quantile_source.empty:
        for label, probability in (("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p99", 0.99)):
            quantiles[label] = float(quantile_source.quantile(probability))
    within_threshold = int((quantile_source <= threshold_days).sum()) if len(quantile_source) else 0
    exclusion_counts = {
        str(status): int(count)
        for status, count in labels["label_status"].value_counts(dropna=False).items()
    }
    target_counts = {
        str(label): int(count)
        for label, count in labels["resolution_outcome"].value_counts(dropna=False).items()
    }
    report = {
        "raw_rows": int(len(frame)),
        "rows_after_duplicate_rule": int(len(output)),
        "exact_duplicate_rows_including_objectid": exact_with_id,
        "exact_business_duplicate_rows_removed": removed_duplicates,
        "notification_ids_with_repeats": repeated_id_count,
        "rows_in_repeated_notification_ids": rows_in_repeated,
        "reference_date": str(pd.Timestamp(reference_date).date()),
        "threshold_days": int(threshold_days),
        "observation_days": int(observation_days),
        "completed_after_reference": int(
            (
                completed.notna()
                & completed.dt.normalize().gt(pd.Timestamp(reference_date).normalize())
            ).sum()
        ),
        "completed_before_2023": int(
            (completed.notna() & completed.dt.normalize().lt(pd.Timestamp("2023-01-01"))).sum()
        ),
        "created_min": None if pd.isna(created_min) else str(pd.Timestamp(created_min).date()),
        "created_max": None if pd.isna(created_max) else str(pd.Timestamp(created_max).date()),
        "completed_min": None if not bool(completed.notna().any()) else str(completed.min().date()),
        "completed_max": None if not bool(completed.notna().any()) else str(completed.max().date()),
        "exclusion_counts": exclusion_counts,
        "target_counts": target_counts,
        "unusual_long_durations_retained": int((labels["duration_flag"] == "unusual_long").sum()),
        "completed_duration_count": int(len(quantile_source)),
        "completed_within_threshold": within_threshold,
        "duration_quantiles_days": quantiles,
        "changed_before_created": changed_before_created,
        "malformed_changed_on": malformed_changed,
        "coordinate_quality_counts": {
            str(status): int(count) for status, count in coordinate_quality.value_counts().items()
        },
        "notification_type_matches_id": int(type_matches_id.fillna(False).sum())
        if notification_name is not None and type_name is not None
        else 0,
        "notification_type_other": int((~type_matches_id.fillna(False)).sum())
        if notification_name is not None and type_name is not None
        else 0,
        "model_features": select_model_features(output.columns),
        "unusual_duration_rule_days": int(UNUSUAL_DURATION_DAYS),
    }
    return output, report


def save_analysis_dataset(frame: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the analysis table to parquet, or CSV if parquet cannot be written."""
    destination = PROCESSED_DATA_DIR / "service_requests_labelled.parquet" if path is None else path
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(destination, index=False)
        return destination
    except (ImportError, ValueError, OSError):
        csv_path = destination.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        return csv_path


def write_audit_tables(dictionary: pd.DataFrame, report: dict) -> dict[str, Path]:
    """Save aggregate tables. These files do not contain coordinates or free text."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    dictionary_path = TABLES_DIR / "02_data_dictionary.csv"
    dictionary.to_csv(dictionary_path, index=False)
    paths["dictionary"] = dictionary_path

    missingness = dictionary.loc[:, ["column", "missing_count", "missing_percent"]]
    missing_path = TABLES_DIR / "02_missingness.csv"
    missingness.to_csv(missing_path, index=False)
    paths["missingness"] = missing_path

    target_rows = []
    for key, value in sorted(report["target_counts"].items(), key=lambda item: str(item[0])):
        outcome = "(no label)" if str(key) in {"None", "nan", "<NA>"} else key
        target_rows.append({"outcome": outcome, "rows": value})
    target_path = TABLES_DIR / "02_target_distribution.csv"
    pd.DataFrame(target_rows).to_csv(target_path, index=False)
    paths["target"] = target_path

    exclusion_rows = [
        {"reason": key, "rows": value} for key, value in sorted(report["exclusion_counts"].items())
    ]
    exclusion_path = TABLES_DIR / "02_exclusion_counts.csv"
    pd.DataFrame(exclusion_rows).to_csv(exclusion_path, index=False)
    paths["exclusions"] = exclusion_path

    decisions = [
        {
            "decision": "Remove exact duplicate business rows, ignoring ObjectId. Keep the first copy.",
            "rows_affected": report["exact_business_duplicate_rows_removed"],
        },
        {
            "decision": "Keep rows that share a notification id when the other columns differ.",
            "rows_affected": report["rows_in_repeated_notification_ids"],
        },
        {
            "decision": "Exclude malformed Created On Date values from the label.",
            "rows_affected": report["exclusion_counts"].get("malformed_created_date", 0),
        },
        {
            "decision": "Exclude rows with no Created On Date.",
            "rows_affected": report["exclusion_counts"].get("missing_created_date", 0),
        },
        {
            "decision": "Exclude malformed Completed Date values. Do not treat them as open.",
            "rows_affected": report["exclusion_counts"].get("malformed_completed_date", 0),
        },
        {
            "decision": "Exclude completion dates earlier than creation dates.",
            "rows_affected": report["exclusion_counts"].get("completion_before_creation", 0),
        },
        {
            "decision": "Exclude requests created after the observation cutoff.",
            "rows_affected": report["exclusion_counts"].get("created_after_reference", 0),
        },
        {
            "decision": "Leave open requests younger than the observation window unlabelled.",
            "rows_affected": report["exclusion_counts"].get("insufficient_follow_up", 0),
        },
        {
            "decision": "Keep completed requests longer than the unusual-duration flag. Do not delete them.",
            "rows_affected": report["unusual_long_durations_retained"],
        },
        {
            "decision": "Flag coordinates outside the local metre envelope. Do not delete those rows.",
            "rows_affected": report["coordinate_quality_counts"].get("outside_local_grid", 0),
        },
        {
            "decision": "Omit Completed Date, Changed on, coordinates, and Notification type from the analysis file.",
            "rows_affected": report["rows_after_duplicate_rule"],
        },
    ]
    decision_path = TABLES_DIR / "02_cleaning_decisions.csv"
    pd.DataFrame(decisions).to_csv(decision_path, index=False)
    paths["decisions"] = decision_path

    duration_path = TABLES_DIR / "02_duration_summary.csv"
    duration_row = {
        "completed_rows": report["completed_duration_count"],
        "within_threshold": report["completed_within_threshold"],
        "threshold_days": report["threshold_days"],
        "unusual_long_retained": report["unusual_long_durations_retained"],
        **report["duration_quantiles_days"],
    }
    pd.DataFrame([duration_row]).to_csv(duration_path, index=False)
    paths["duration"] = duration_path
    return paths


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
