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
    LEAKAGE_OR_POST_EVENT_COLUMNS,
    PLANNED_CHANGED_ON_COLUMN,
    PLANNED_COMPLETED_DATE_COLUMN,
    PLANNED_CREATED_DATE_COLUMN,
    TARGET_LABELS,
)

# Whole calendar days at or below the threshold are Resolved Fast.
# This comparison is a project rule, not a result measured from the extract.
FAST_LABEL = TARGET_LABELS[0]
SLOW_LABEL = TARGET_LABELS[1]
UNRESOLVED_LABEL = TARGET_LABELS[2]


def _blocked_names() -> set[str]:
    """Case-folded names that must not be used as model inputs."""
    names = list(LEAKAGE_OR_POST_EVENT_COLUMNS) + list(CONSTRUCTED_COLUMNS)
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
    """Match brief column names to the extract with a case-insensitive exact match.

    A missing match means the extract uses another name, or the file is not
    the expected dataset. It is not filled in with a guess.
    """
    lookup: dict[str, str] = {}
    for column in columns:
        lookup[str(column).casefold()] = str(column)
    planned = {
        "created_on_date": PLANNED_CREATED_DATE_COLUMN,
        "completed_date": PLANNED_COMPLETED_DATE_COLUMN,
        "changed_on": PLANNED_CHANGED_ON_COLUMN,
    }
    report: dict[str, dict[str, str | None]] = {}
    for role, planned_name in planned.items():
        report[role] = {
            "planned_name": planned_name,
            "actual_name": lookup.get(planned_name.casefold()),
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
                "found": info["actual_name"] is not None,
                "actual_name": info["actual_name"],
            }
        )
    return pd.DataFrame(rows)


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

    Rules, all in whole calendar days:
    - Duration at or below ``fast_slow_threshold_days`` is Resolved Fast.
      A longer completed duration is Resolved Slow.
    - A request with no completion on or before ``reference_date`` is still
      open at that date. It is Unresolved only when its age is at least
      ``observation_window_days``. Younger open requests are excluded
      (``insufficient_follow_up``), because they may still be completed.
    - A completion after the reference date is treated as not yet known at
      the reference instant, then the same open-request rule is applied.
    - Missing creation dates, creation after the reference date, and
      completion before creation are excluded rather than given a class.

    ``label_status`` is ``labelled`` when ``resolution_outcome`` is set.
    Otherwise it is the exclusion reason.
    """
    if fast_slow_threshold_days <= 0 or observation_window_days <= 0:
        raise ValueError("Thresholds must be positive whole numbers of days.")
    missing = [name for name in (created_column, completed_column) if name not in frame.columns]
    if missing:
        raise KeyError(
            "Cannot build the target because these columns are not in the frame: "
            + ", ".join(missing)
        )

    created = _as_naive_datetime(frame[created_column])
    completed = _as_naive_datetime(frame[completed_column])
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

    missing_created = created.isna()
    status.loc[missing_created] = "missing_created_date"

    created_after = ~missing_created & created_day.gt(reference_day)
    status.loc[created_after] = "created_after_reference"

    negative = status.eq("labelled") & completed.notna() & completed.lt(created)
    status.loc[negative] = "completion_before_creation"

    # Still open on the reference day: no usable completion date on or before it.
    open_at_reference = status.eq("labelled") & (
        completed.isna() | completed_day.gt(reference_day)
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
        },
        index=frame.index,
    )
