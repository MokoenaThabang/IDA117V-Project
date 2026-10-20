"""Aggregate the labelled service-request table for exploratory analysis.

These functions describe the Phase 2 analysis file. They do not impute
values, drop long durations, or train a model. Completed Date is not in
that file; a blank resolution duration is interpreted from ``label_status``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, TARGET_LABELS, UNUSUAL_DURATION_DAYS

# Columns kept for charts. Identifiers are not required and are not loaded.
ANALYSIS_COLUMNS = (
    "Sub_Council",
    "Ward",
    "Suburb",
    "C3_Complaint_Type",
    "Work_Center",
    "Created_On_Date",
    "Notifications_Created",
    "resolution_outcome",
    "resolution_days",
    "label_status",
    "duration_flag",
    "coordinate_quality",
    "notification_id_repeated",
)

PREDICTOR_COLUMNS = (
    "Sub_Council",
    "Ward",
    "Suburb",
    "C3_Complaint_Type",
    "Work_Center",
    "Created_On_Date",
    "Notifications_Created",
)

PREDICTOR_LABELS = {
    "Sub_Council": "Subcouncil",
    "Ward": "Ward",
    "Suburb": "Suburb",
    "C3_Complaint_Type": "Complaint type",
    "Work_Center": "Work centre",
    "Created_On_Date": "Created on date",
    "Notifications_Created": "Notifications created",
}

# Stored text that is present but does not name a place.
SUBURB_PLACEHOLDERS = ("#", "GENERIC TOWN 0000")
SUBCOUNCIL_UNASSIGNED = "Not assigned"

# The five required charts use this many of the largest groups.
DEFAULT_TOP_GROUPS = 12
# Proportions below this labelled count are reported in tables only.
SMALL_GROUP_REQUESTS = 100

DURATION_BINS = (
    ("Same calendar day (0 days)", 0, 0),
    ("1 day", 1, 1),
    ("2 to 7 days", 2, 7),
    ("8 to 30 days", 8, 30),
    ("31 to 90 days", 31, 90),
    ("91 to 365 days", 91, 365),
    ("More than 365 days", 366, None),
)


class LabelledDataNotFoundError(FileNotFoundError):
    """Raised when the Phase 2 analysis file has not been written."""


def labelled_dataset_path() -> Path:
    """Return the parquet analysis file, or the CSV fallback if that is all that exists."""
    parquet_path = PROCESSED_DATA_DIR / "service_requests_labelled.parquet"
    csv_path = PROCESSED_DATA_DIR / "service_requests_labelled.csv"
    if parquet_path.is_file():
        return parquet_path
    if csv_path.is_file():
        return csv_path
    raise LabelledDataNotFoundError(
        "The labelled analysis file is not in data/processed/. "
        "Run notebooks/01_data_acquisition_and_audit.ipynb first. "
        "Expected data/processed/service_requests_labelled.parquet."
    )


def column_names(path: Path | None = None) -> list[str]:
    """Read column names without loading the table."""
    source = labelled_dataset_path() if path is None else path
    if source.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(source).schema_arrow.names)
    header = pd.read_csv(source, nrows=0)
    return [str(column) for column in header.columns]


def load_labelled_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the analysis file, leaving identifier columns out of the frame."""
    source = labelled_dataset_path() if path is None else path
    if not source.is_file():
        raise LabelledDataNotFoundError(
            f"Labelled dataset not found: {source}. "
            "Run notebooks/01_data_acquisition_and_audit.ipynb first."
        )
    names = column_names(source)
    required = [name for name in ANALYSIS_COLUMNS if name not in names]
    if required:
        raise ValueError(
            "The labelled file is missing columns needed for this EDA: "
            + ", ".join(required)
            + ". Actual columns: "
            + ", ".join(names)
            + "."
        )
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source, columns=list(ANALYSIS_COLUMNS))
    else:
        frame = pd.read_csv(source, usecols=list(ANALYSIS_COLUMNS))
        frame["Created_On_Date"] = pd.to_datetime(frame["Created_On_Date"])
    return frame


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "Cannot summarise this frame. Missing columns: "
            + ", ".join(missing)
            + ". Actual columns: "
            + ", ".join(map(str, frame.columns))
            + "."
        )


def labelled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows with a resolution class. Recent open requests are excluded."""
    _require_columns(frame, ("resolution_outcome",))
    return frame.loc[frame["resolution_outcome"].notna()].copy()


def target_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Count the three classes. Unlabelled rows are reported on their own row."""
    _require_columns(frame, ("resolution_outcome",))
    labelled = frame["resolution_outcome"].notna()
    labelled_count = int(labelled.sum())
    rows = []
    for outcome in TARGET_LABELS:
        count = int((frame["resolution_outcome"] == outcome).sum())
        rows.append(
            {
                "outcome": outcome,
                "requests": count,
                "percent_of_labelled": (count / labelled_count * 100.0) if labelled_count else 0.0,
                "percent_of_analysis_file": count / len(frame) * 100.0 if len(frame) else 0.0,
            }
        )
    unlabelled = int((~labelled).sum())
    rows.append(
        {
            "outcome": "(no label)",
            "requests": unlabelled,
            "percent_of_labelled": pd.NA,
            "percent_of_analysis_file": unlabelled / len(frame) * 100.0 if len(frame) else 0.0,
        }
    )
    return pd.DataFrame(rows)


def predictor_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Missingness of fields that can be known at logging.

    A blank completion date is not included. That blank is a status, not a
    missing predictor, and is counted by ``completion_status``.
    """
    _require_columns(frame, PREDICTOR_COLUMNS)
    row_count = int(len(frame))
    records = []
    for column in PREDICTOR_COLUMNS:
        missing = int(frame[column].isna().sum())
        records.append(
            {
                "column": column,
                "display_name": PREDICTOR_LABELS[column],
                "missing_count": missing,
                "missing_percent": (missing / row_count * 100.0) if row_count else 0.0,
                "kind": "genuine_missing",
            }
        )
    return pd.DataFrame(records)


def completion_status(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate a recorded duration from a meaningful blank completion date."""
    _require_columns(frame, ("resolution_days", "label_status", "resolution_outcome"))
    recorded = frame["resolution_days"].notna()
    blank = ~recorded
    unresolved = blank & frame["resolution_outcome"].eq("Unresolved")
    too_recent = blank & frame["label_status"].eq("insufficient_follow_up")
    invalid = blank & frame["label_status"].eq("completion_before_creation")
    other_blank = blank & ~unresolved & ~too_recent & ~invalid
    specs = (
        ("completion_recorded", "Recorded completion", recorded, "duration_known"),
        (
            "blank_unresolved",
            "Open at least 7 days (Unresolved)",
            unresolved,
            "meaningful_blank",
        ),
        (
            "blank_insufficient_follow_up",
            "Open fewer than 7 days (no label)",
            too_recent,
            "meaningful_blank",
        ),
        (
            "completion_before_creation",
            "Completion before creation (excluded)",
            invalid,
            "invalid_date",
        ),
        ("other_blank_completion", "Other blank completion", other_blank, "review"),
    )
    row_count = int(len(frame))
    records = []
    for key, label, mask, kind in specs:
        count = int(mask.sum())
        if key == "other_blank_completion" and count == 0:
            continue
        records.append(
            {
                "status": key,
                "display_name": label,
                "requests": count,
                "percent_of_analysis_file": (count / row_count * 100.0) if row_count else 0.0,
                "kind": kind,
            }
        )
    return pd.DataFrame(records)


def _text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("").astype("string").str.strip()


def data_quality_table(frame: pd.DataFrame, exact_duplicates_removed: int | None = None) -> pd.DataFrame:
    """Audit counts that can be recomputed from the analysis file.

    ``exact_duplicates_removed`` comes from the Phase 2 cleaning log. Those
    rows are not in this file, so this function does not invent the count.
    """
    _require_columns(
        frame,
        (
            "resolution_outcome",
            "label_status",
            "coordinate_quality",
            "notification_id_repeated",
            "Suburb",
            "Sub_Council",
            "resolution_days",
        ),
    )
    row_count = int(len(frame))
    suburb = _text_column(frame, "Suburb")
    subcouncil = _text_column(frame, "Sub_Council")
    records = [
        ("analysis_rows", "Rows in the analysis file", row_count, "analysis_file"),
        (
            "labelled_rows",
            "Rows with a resolution class",
            int(frame["resolution_outcome"].notna().sum()),
            "analysis_file",
        ),
        (
            "insufficient_follow_up",
            "Open requests excluded because they are younger than 7 days",
            int(frame["label_status"].eq("insufficient_follow_up").sum()),
            "analysis_file",
        ),
        (
            "completion_before_creation",
            "Rows excluded because completion is earlier than creation",
            int(frame["label_status"].eq("completion_before_creation").sum()),
            "analysis_file",
        ),
        (
            "unusual_long_duration",
            "Completed requests longer than 365 days, retained",
            int((frame["resolution_days"] > UNUSUAL_DURATION_DAYS).sum()),
            "analysis_file",
        ),
        (
            "repeated_notification_id_rows",
            "Rows that still share a notification id after exact copies were removed",
            int(frame["notification_id_repeated"].fillna(False).sum()),
            "analysis_file",
        ),
        (
            "suburb_placeholder_hash",
            "Suburb stored as '#'",
            int(suburb.eq(SUBURB_PLACEHOLDERS[0]).sum()),
            "analysis_file",
        ),
        (
            "suburb_generic_town",
            "Suburb stored as 'GENERIC TOWN 0000'",
            int(suburb.str.casefold().eq(SUBURB_PLACEHOLDERS[1].casefold()).sum()),
            "analysis_file",
        ),
        (
            "subcouncil_not_assigned",
            "Subcouncil stored as 'Not assigned'",
            int(subcouncil.eq(SUBCOUNCIL_UNASSIGNED).sum()),
            "analysis_file",
        ),
    ]
    for quality, label in (
        ("missing", "Coordinate pair missing"),
        ("outside_local_grid", "Coordinate pair outside the local metre envelope"),
        ("plausible_local_grid", "Coordinate pair inside the local metre envelope"),
    ):
        records.append(
            (
                f"coordinate_{quality}",
                label,
                int(frame["coordinate_quality"].eq(quality).sum()),
                "analysis_file",
            )
        )
    if exact_duplicates_removed is not None:
        records.insert(
            1,
            (
                "exact_business_duplicates_removed",
                "Exact duplicate business rows removed before this file was written",
                int(exact_duplicates_removed),
                "phase2_cleaning_log",
            ),
        )
    table = pd.DataFrame(records, columns=["item", "description", "requests", "source"])
    table["percent_of_analysis_file"] = [
        (count / row_count * 100.0) if row_count and source == "analysis_file" else pd.NA
        for count, source in zip(table["requests"], table["source"], strict=True)
    ]
    return table


def phase2_duplicate_removals(path: Path) -> int:
    """Read the exact-duplicate count from the Phase 2 cleaning log."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase 2 cleaning log not found: {path}. "
            "Run the audit notebook before interpreting duplicate removals."
        )
    log = pd.read_csv(path)
    if "decision" not in log.columns or "rows_affected" not in log.columns:
        raise ValueError(
            f"{path.name} does not have decision and rows_affected columns. "
            f"Actual columns: {', '.join(map(str, log.columns))}."
        )
    match = log["decision"].astype("string").str.startswith("Remove exact duplicate business rows")
    if int(match.sum()) != 1:
        raise ValueError(
            f"{path.name} does not contain exactly one exact-duplicate decision. "
            "The EDA will not guess that count."
        )
    return int(log.loc[match, "rows_affected"].iloc[0])


def group_outcomes(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Count each resolution class inside one grouping column.

    Only labelled rows enter the proportions. The result is sorted by
    labelled request count, largest first.
    """
    _require_columns(frame, (column, "resolution_outcome"))
    labelled = labelled_rows(frame)
    counts = (
        labelled.groupby(column, dropna=False)["resolution_outcome"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for outcome in TARGET_LABELS:
        if outcome not in counts.columns:
            counts[outcome] = 0
    counts = counts.loc[:, list(TARGET_LABELS)]
    counts["requests"] = counts.sum(axis=1).astype(int)
    for outcome in TARGET_LABELS:
        counts[outcome] = counts[outcome].astype(int)
        counts[f"{outcome} percent"] = counts[outcome] / counts["requests"] * 100.0
    counts.index.name = column
    counts = counts.reset_index()
    return counts.sort_values(["requests", column], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def largest_groups(outcomes: pd.DataFrame, top_n: int = DEFAULT_TOP_GROUPS) -> pd.DataFrame:
    """Keep the largest groups. The rule is count, not outcome rate."""
    if top_n <= 0:
        raise ValueError("top_n must be a positive whole number.")
    if "requests" not in outcomes.columns:
        raise ValueError("The outcome table needs a requests column.")
    selected = outcomes.sort_values(["requests"], ascending=False).head(top_n).copy()
    selected.insert(0, "rank", range(1, len(selected) + 1))
    selected["selection_rule"] = f"largest {top_n} groups by labelled request count"
    return selected


def small_group_summary(outcomes: pd.DataFrame, minimum_requests: int = SMALL_GROUP_REQUESTS) -> dict:
    """Describe groups that are too small for a stable proportion."""
    small = outcomes.loc[outcomes["requests"] < minimum_requests]
    total = int(outcomes["requests"].sum())
    small_rows = int(small["requests"].sum())
    return {
        "minimum_requests": int(minimum_requests),
        "small_groups": int(len(small)),
        "groups": int(len(outcomes)),
        "requests_in_small_groups": small_rows,
        "percent_of_labelled_in_small_groups": (small_rows / total * 100.0) if total else 0.0,
    }


def monthly_activity(frame: pd.DataFrame) -> pd.DataFrame:
    """Count requests and class shares for every month from the first to the last.

    A month with no created dates is kept and marked as a gap. Class
    percentages use labelled rows only.
    """
    _require_columns(frame, ("Created_On_Date", "resolution_outcome", "label_status"))
    created = pd.to_datetime(frame["Created_On_Date"])
    if created.notna().sum() == 0:
        raise ValueError("Created On Date has no parsed values, so monthly volume cannot be built.")
    month = created.dt.to_period("M")
    working = frame.loc[month.notna()].copy()
    working["month"] = month.loc[month.notna()].astype(str)
    start = month.min()
    end = month.max()
    index = pd.period_range(start, end, freq="M").astype(str)

    counts = working.groupby("month").size().reindex(index, fill_value=0)
    labelled = working.loc[working["resolution_outcome"].notna()]
    class_counts = (
        labelled.groupby("month")["resolution_outcome"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(index, fill_value=0)
    )
    for outcome in TARGET_LABELS:
        if outcome not in class_counts.columns:
            class_counts[outcome] = 0
    insufficient = (
        working.loc[working["label_status"].eq("insufficient_follow_up")]
        .groupby("month")
        .size()
        .reindex(index, fill_value=0)
    )
    table = pd.DataFrame({"month": index, "requests": counts.to_numpy().astype(int)})
    table["gap"] = table["requests"].eq(0)
    labelled_count = class_counts[list(TARGET_LABELS)].sum(axis=1).astype(int)
    table["labelled_requests"] = labelled_count.to_numpy()
    table["insufficient_follow_up"] = insufficient.to_numpy().astype(int)
    for outcome in TARGET_LABELS:
        column_name = {
            "Resolved Fast": "resolved_fast",
            "Resolved Slow": "resolved_slow",
            "Unresolved": "unresolved",
        }[outcome]
        table[column_name] = class_counts[outcome].to_numpy().astype(int)
        percent = table[column_name] / table["labelled_requests"].where(table["labelled_requests"].gt(0)) * 100.0
        table[f"{column_name}_percent"] = percent.astype(float)
    return table


def duration_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Bin completed durations. Rows without a duration are excluded here."""
    _require_columns(frame, ("resolution_days",))
    days = frame["resolution_days"].dropna()
    total = int(len(days))
    records = []
    for label, low, high in DURATION_BINS:
        if high is None:
            mask = days >= low
        else:
            mask = days.between(low, high)
        count = int(mask.sum())
        records.append(
            {
                "bin": label,
                "requests": count,
                "percent_of_completed": (count / total * 100.0) if total else 0.0,
            }
        )
    return pd.DataFrame(records)


def extreme_duration_tables(frame: pd.DataFrame, long_days: int = UNUSUAL_DURATION_DAYS) -> dict[str, pd.DataFrame]:
    """Summarise very long completed durations without listing identifiers."""
    _require_columns(frame, ("resolution_days", "C3_Complaint_Type", "Work_Center", "Created_On_Date"))
    days = frame["resolution_days"]
    completed = days.notna()
    long = frame.loc[completed & days.gt(long_days)].copy()
    completed_days = days.loc[completed]
    summary = pd.DataFrame(
        [
            {"metric": "completed_requests", "value": int(completed.sum())},
            {"metric": "median_days", "value": float(completed_days.median()) if len(completed_days) else pd.NA},
            {"metric": "p75_days", "value": float(completed_days.quantile(0.75)) if len(completed_days) else pd.NA},
            {"metric": "p90_days", "value": float(completed_days.quantile(0.90)) if len(completed_days) else pd.NA},
            {"metric": "p99_days", "value": float(completed_days.quantile(0.99)) if len(completed_days) else pd.NA},
            {"metric": "maximum_days", "value": float(completed_days.max()) if len(completed_days) else pd.NA},
            {"metric": "same_calendar_day", "value": int(completed_days.eq(0).sum())},
            {"metric": "within_7_days", "value": int(completed_days.le(7).sum())},
            {"metric": f"longer_than_{long_days}_days", "value": int(len(long))},
            {"metric": "longer_than_730_days", "value": int(completed_days.gt(730).sum())},
            {"metric": "longer_than_1000_days", "value": int(completed_days.gt(1000).sum())},
        ]
    )
    by_complaint = (
        long.groupby("C3_Complaint_Type", dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(10)
        .rename("requests")
        .reset_index()
    )
    by_centre = (
        long.groupby("Work_Center", dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(10)
        .rename("requests")
        .reset_index()
    )
    return {"summary": summary, "by_complaint_type": by_complaint, "by_work_centre": by_centre}


def write_eda_tables(tables: dict[str, pd.DataFrame], directory: Path) -> dict[str, Path]:
    """Save aggregate tables. Values are counts and rates, not row-level records."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths
