"""Leakage checks and the resolution-time label.

The label uses Created On Date and Completed Date. Completed Date and
Changed on are not model features. Thresholds are arguments: Phase 1 does
not estimate them from the extract, and this module does not set defaults.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from src.config import (
    CONSTRUCTED_COLUMNS,
    IDENTIFIER_COLUMNS,
    LEAKAGE_OR_POST_EVENT_COLUMNS,
    PLANNED_CHANGED_ON_COLUMN,
    PLANNED_COMPLETED_DATE_COLUMN,
    PLANNED_CREATED_DATE_COLUMN,
    PRECISE_LOCATION_COLUMNS,
    SERVICE_CHANGED_ON_COLUMN,
    SERVICE_COMPLETED_DATE_COLUMN,
    SERVICE_CREATED_DATE_COLUMN,
    TARGET_LABELS,
    UNUSUAL_DURATION_DAYS,
)

# Whole calendar days at or below the threshold are Resolved Fast.
# This comparison is a project rule, not a result measured from the extract.
FAST_LABEL = TARGET_LABELS[0]
SLOW_LABEL = TARGET_LABELS[1]
UNRESOLVED_LABEL = TARGET_LABELS[2]


def _blocked_names() -> set[str]:
    """Case-folded names that must not be used as model inputs."""
    names = (
        list(LEAKAGE_OR_POST_EVENT_COLUMNS)
        + list(CONSTRUCTED_COLUMNS)
        + list(IDENTIFIER_COLUMNS)
        + list(PRECISE_LOCATION_COLUMNS)
    )
    return {name.casefold() for name in names}


def select_model_features(columns: list[str] | pd.Index) -> list[str]:
    """Drop known leakage and constructed columns. Other columns are kept.

    Keeping a column here does not mean it is safe. Fields that encode the
    outcome, or that are filled in after logging, still need a manual check
    once the real column list is known.
    """
    blocked = _blocked_names()
    return [column for column in columns if str(column).casefold() not in blocked]


def match_planned_columns(columns: list[str] | pd.Index) -> dict[str, dict[str, str | None]]:
    """Match known date fields with a case-insensitive exact match.

    Each role accepts the project-brief alias and the feature-service field
    name. Both were read from the published layer. Any other spelling is
    reported as missing rather than renamed.
    """
    lookup: dict[str, str] = {}
    for column in columns:
        lookup[str(column).casefold()] = str(column)
    planned = {
        "created_on_date": (PLANNED_CREATED_DATE_COLUMN, SERVICE_CREATED_DATE_COLUMN),
        "completed_date": (PLANNED_COMPLETED_DATE_COLUMN, SERVICE_COMPLETED_DATE_COLUMN),
        "changed_on": (PLANNED_CHANGED_ON_COLUMN, SERVICE_CHANGED_ON_COLUMN),
    }
    report: dict[str, dict[str, str | None]] = {}
    for role, names in planned.items():
        actual = None
        for name in names:
            actual = lookup.get(name.casefold())
            if actual is not None:
                break
        report[role] = {
            "planned_name": names[0],
            "service_name": names[1],
            "actual_name": actual,
        }
    return report


def planned_column_report(columns: list[str] | pd.Index) -> pd.DataFrame:
    """Return the planned-name match as a table for the audit notebook."""
    rows = []
    for role, info in match_planned_columns(columns).items():
        rows.append(
            {
                "role": role,
                "planned_name": info["planned_name"],
                "service_name": info["service_name"],
                "found": info["actual_name"] is not None,
                "actual_name": info["actual_name"],
            }
        )
    return pd.DataFrame(rows)


def parse_portal_datetime(series: pd.Series) -> pd.Series:
    """Parse a date column to naive Africa/Johannesburg timestamps.

    Feature-service dates are integer UTC epoch milliseconds. A value such as
    1672524000000 is 2022-12-31 22:00 UTC, which is 2023-01-01 00:00 in Cape
    Town. Converting to Johannesburg keeps that request on 1 January.
    Text dates and existing datetimes are parsed as recorded. Unparseable
    values become missing. Nothing is imputed.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        tz = getattr(parsed.dtype, "tz", None)
        if tz is not None:
            parsed = parsed.dt.tz_convert("Africa/Johannesburg").dt.tz_localize(None)
        return parsed

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_present = numeric.notna()
    if bool(numeric_present.any()) and float(numeric.loc[numeric_present].abs().median()) >= 1e11:
        parsed = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        parsed = parsed.dt.tz_convert("Africa/Johannesburg").dt.tz_localize(None)
        text_rows = ~numeric_present & series.notna()
        if bool(text_rows.any()):
            text = pd.to_datetime(series, errors="coerce")
            parsed = parsed.where(~text_rows, text)
        return parsed

    parsed = pd.to_datetime(series, errors="coerce")
    tz = getattr(parsed.dtype, "tz", None)
    if tz is not None:
        parsed = parsed.dt.tz_convert("Africa/Johannesburg").dt.tz_localize(None)
    return parsed


def _blank_mask(series: pd.Series) -> pd.Series:
    """True where the original cell is null or an empty string."""
    missing = series.isna()
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        text = series.astype("string").str.strip()
        missing = missing | text.isna() | text.eq("")
    return missing


def date_problem_masks(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return parsed timestamps, missing mask, and malformed mask."""
    missing = _blank_mask(series)
    parsed = parse_portal_datetime(series)
    malformed = ~missing & parsed.isna()
    return parsed, missing, malformed


def require_date_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    """Map date roles to actual column names, or stop with the real columns."""
    found = {role: info["actual_name"] for role, info in match_planned_columns(frame.columns).items()}
    missing_roles = [
        role for role in ("created_on_date", "completed_date") if found[role] is None
    ]
    if missing_roles:
        raise KeyError(
            "Target construction stopped because required date columns are missing. "
            f"Missing roles: {', '.join(missing_roles)}. "
            "Accepted names are 'Created On Date' or 'Created_On_Date', and "
            "'Completed Date' or 'Completed_Date'. "
            f"Actual columns: {', '.join(map(str, frame.columns))}."
        )
    return found


def _as_naive_datetime(series: pd.Series) -> pd.Series:
    """Parse dates. Invalid values become missing. Time zones keep their clock time.

    Dropping the zone without converting avoids moving an evening Cape Town
    timestamp onto the previous UTC calendar day. Invalid text is not replaced
    with an invented date.
    """
    parsed = pd.to_datetime(series, errors="coerce")
    tz = getattr(parsed.dtype, "tz", None)
    if tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _as_timestamp(value: date | datetime | pd.Timestamp | str) -> pd.Timestamp:
    """Normalise the study reference to midnight on that calendar day."""
    stamp = pd.Timestamp(value)
    tz = getattr(stamp, "tzinfo", None)
    if tz is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def build_resolution_target(
    frame: pd.DataFrame,
    created_column: str,
    completed_column: str,
    fast_slow_threshold_days: int,
    observation_window_days: int,
    reference_date: date | datetime | pd.Timestamp | str,
) -> pd.DataFrame:
    """Build labels from creation and completion dates.

    Returns a new frame with three columns aligned to ``frame.index``:
    ``resolution_outcome``, ``resolution_days``, and ``label_status``.
    The input frame is not modified. Completed Date is not copied into the
    result, so it cannot be used as a feature by accident.

    Rules, all in whole calendar days after Johannesburg conversion:
    - Duration at or below ``fast_slow_threshold_days`` is Resolved Fast.
      A longer completed duration is Resolved Slow, including very long
      durations. Those long rows are flagged and kept.
    - A request with no completion on or before ``reference_date`` is still
      open at that date. It is Unresolved only when its age is at least
      ``observation_window_days``. Younger open requests are excluded
      (``insufficient_follow_up``). They are not treated as failures.
    - A completion after the reference date is treated as not yet known,
      then the same open-request rule is applied.
    - Missing creation dates, malformed dates, creation after the reference
      date, and completion before creation are excluded rather than labelled.

    ``label_status`` is ``labelled`` when ``resolution_outcome`` is set.
    Otherwise it is the exclusion reason. ``duration_flag`` is
    ``unusual_long`` when a labelled completion exceeds
    ``UNUSUAL_DURATION_DAYS``. That flag does not remove the row.
    """
    if fast_slow_threshold_days <= 0 or observation_window_days <= 0:
        raise ValueError("Thresholds must be positive whole numbers of days.")
    missing = [name for name in (created_column, completed_column) if name not in frame.columns]
    if missing:
        raise KeyError(
            "Cannot build the target because these columns are not in the frame: "
            + ", ".join(missing)
        )

    created, missing_created, malformed_created = date_problem_masks(frame[created_column])
    completed, missing_completed, malformed_completed = date_problem_masks(frame[completed_column])
    reference_day = _as_timestamp(reference_date)
    created_day = created.dt.normalize()
    completed_day = completed.dt.normalize()
    age_days = (reference_day - created_day).dt.days
    duration_days = (completed_day - created_day).dt.days

    status = pd.Series("labelled", index=frame.index, dtype="object")
    # Build from a Python list so missing outcomes stay None. Passing None
    # straight to the Series constructor turns them into NaN in pandas 3.
    outcome = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    stored_duration = pd.Series(float("nan"), index=frame.index, dtype="float64")
    duration_flag = pd.Series([None] * len(frame), index=frame.index, dtype="object")

    status.loc[malformed_created] = "malformed_created_date"
    status.loc[status.eq("labelled") & missing_created] = "missing_created_date"

    created_after = status.eq("labelled") & created_day.gt(reference_day)
    status.loc[created_after] = "created_after_reference"

    status.loc[status.eq("labelled") & malformed_completed] = "malformed_completed_date"

    negative = status.eq("labelled") & completed.notna() & completed_day.lt(created_day)
    status.loc[negative] = "completion_before_creation"

    # Still open on the reference day: no usable completion on or before it.
    open_at_reference = status.eq("labelled") & (
        missing_completed | completed_day.gt(reference_day)
    )
    too_recent = open_at_reference & age_days.lt(observation_window_days)
    unresolved = open_at_reference & age_days.ge(observation_window_days)
    status.loc[too_recent] = "insufficient_follow_up"
    outcome.loc[unresolved] = UNRESOLVED_LABEL

    resolved = status.eq("labelled") & completed.notna() & ~open_at_reference
    fast = resolved & duration_days.le(fast_slow_threshold_days)
    slow = resolved & duration_days.gt(fast_slow_threshold_days)
    outcome.loc[fast] = FAST_LABEL
    outcome.loc[slow] = SLOW_LABEL
    stored_duration.loc[fast | slow] = duration_days.loc[fast | slow].astype(float)
    duration_flag.loc[slow & duration_days.gt(UNUSUAL_DURATION_DAYS)] = "unusual_long"

    leftover = status.eq("labelled") & outcome.isna()
    if bool(leftover.any()):
        raise RuntimeError(
            "Some rows were neither labelled nor excluded. Check the target rules."
        )

    return pd.DataFrame(
        {
            "resolution_outcome": outcome,
            "resolution_days": stored_duration,
            "label_status": status,
            "duration_flag": duration_flag,
        },
        index=frame.index,
    )
