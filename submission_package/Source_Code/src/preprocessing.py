"""Leakage checks, the resolution-time label, and the baseline preprocessor.

The label uses Created On Date and Completed Date. Completed Date, Changed
on, resolution duration, and the target are not model features. The baseline
preprocessor is fit on the training period only.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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


# Calendar fields are derived from Created On Date before the split. The raw
# timestamp is not a predictor. Rare-category counts are learned later, on
# the training rows only.
DAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
SOURCE_CATEGORICAL_COLUMNS = (
    "C3_Complaint_Type",
    "Suburb",
    "Ward",
    "Sub_Council",
)
CATEGORICAL_FEATURES = SOURCE_CATEGORICAL_COLUMNS + (
    "created_day_of_week",
    "created_month",
)
NUMERIC_FEATURES = ("created_is_weekend",)
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Requests created after this date are left out of every modelling split.
# May 2026 is where Resolved Slow collapses and Unresolved jumps, while the
# extract itself ends on 6 October 2026. The cut is a pre-specified
# evaluation rule, not a threshold tuned on the test score.
TRAIN_END = pd.Timestamp("2024-12-31")
VALIDATION_END = pd.Timestamp("2025-12-31")
MATURE_END = pd.Timestamp("2026-04-30")
RARE_CATEGORY_MIN_COUNT = 200
RARE_CATEGORY_LABEL = "(rare)"
MISSING_CATEGORY_LABEL = "(missing)"

_FEATURE_DECISIONS = (
    (
        "C3_Complaint_Type",
        "include",
        "Complaint category stored when the request is logged.",
    ),
    (
        "Suburb",
        "include",
        "Coarse place name on the request. Stored placeholders such as '#' are kept as their own values.",
    ),
    (
        "Ward",
        "include",
        "Ward on the request. This is a coarse area, not a coordinate.",
    ),
    (
        "Sub_Council",
        "include",
        "Subcouncil on the request. The stored value 'Not assigned' is kept as text.",
    ),
    (
        "created_day_of_week",
        "include",
        "Day name derived from Created On Date. One-hot encoded because weekday order is not a linear scale.",
    ),
    (
        "created_month",
        "include",
        "Month name derived from Created On Date. One-hot encoded so December is not treated as far from January.",
    ),
    (
        "created_is_weekend",
        "include",
        "1 when Created On Date is Saturday or Sunday, otherwise 0. This is the only numeric predictor and is standardised on the training rows.",
    ),
    (
        "Created_On_Date",
        "exclude",
        "Used to build the chronological split and the calendar fields. The raw timestamp is not a predictor.",
    ),
    (
        "Work_Center",
        "exclude",
        "Present on the request, but the extract has no assignment time. It is not treated as known at logging.",
    ),
    (
        "Notifications_Created",
        "exclude",
        "Not one of the logging-time candidate features for this baseline.",
    ),
    (
        "Notification",
        "exclude",
        "Request identifier.",
    ),
    (
        "Notification_type",
        "exclude",
        "Identifier mixed with free text.",
    ),
    (
        "ObjectId",
        "exclude",
        "Row identifier.",
    ),
    (
        "Completed_Date",
        "exclude",
        "Known only after completion.",
    ),
    (
        "Changed_on",
        "exclude",
        "Updated after logging.",
    ),
    (
        "resolution_days",
        "exclude",
        "Computed from Completed Date minus Created On Date.",
    ),
    (
        "resolution_outcome",
        "exclude",
        "Target. Never a predictor.",
    ),
    (
        "label_status",
        "exclude",
        "Constructed from the label rule.",
    ),
    (
        "duration_flag",
        "exclude",
        "Constructed from resolution duration.",
    ),
    (
        "X_Y_Co_ordinate_1",
        "exclude",
        "Precise location. Withheld from the model.",
    ),
    (
        "X_Y_Co_ordinate_2",
        "exclude",
        "Precise location. Withheld from the model.",
    ),
)


def feature_decision_table() -> pd.DataFrame:
    """Return the logging-time decision for each candidate or blocked field."""
    return pd.DataFrame(
        [
            {"column": column, "decision": decision, "reason": reason}
            for column, decision, reason in _FEATURE_DECISIONS
        ]
    )


def _reject_blocked_features(columns: list[str], *, allow_work_centre: bool = False) -> None:
    """Raise when a predictor list contains leakage, identifiers, or the target.

    ``allow_work_centre`` is only for the explicit sensitivity comparison.
    The fitted champion and the app keep the default, which still rejects it.
    """
    blocked = _blocked_names()
    blocked.update({"work_center", "created_on_date", "created on date", "notifications_created"})
    if allow_work_centre:
        blocked.discard("work_center")
    found = [column for column in columns if str(column).casefold() in blocked]
    if found:
        raise ValueError(
            "These columns cannot be baseline predictors: "
            + ", ".join(found)
            + ". Completed Date, Changed on, resolution duration, the target, "
            "identifiers, coordinates, and Work Center are excluded."
        )


def add_calendar_features(
    frame: pd.DataFrame,
    created_column: str = SERVICE_CREATED_DATE_COLUMN,
) -> pd.DataFrame:
    """Add weekday, month, and weekend fields. The input frame is not modified."""
    if created_column not in frame.columns:
        raise ValueError(
            f"Cannot derive calendar features without {created_column}. "
            f"Actual columns: {', '.join(map(str, frame.columns))}."
        )
    created = pd.to_datetime(frame[created_column], errors="coerce")
    output = frame.copy()
    day_lookup = dict(enumerate(DAY_NAMES))
    month_lookup = dict(enumerate(MONTH_NAMES, start=1))
    output["created_day_of_week"] = created.dt.dayofweek.map(day_lookup).astype("object")
    output["created_month"] = created.dt.month.map(month_lookup).astype("object")
    weekend = pd.Series(np.nan, index=frame.index, dtype="float64")
    known = created.notna()
    weekend.loc[known] = created.loc[known].dt.dayofweek.ge(5).astype(float)
    output["created_is_weekend"] = weekend
    return output


def logging_feature_frame(
    frame: pd.DataFrame,
    created_column: str = SERVICE_CREATED_DATE_COLUMN,
) -> pd.DataFrame:
    """Return only the predictors used by the baseline.

    The raw creation timestamp, the target, and Work Center are not included.
    """
    missing = [column for column in SOURCE_CATEGORICAL_COLUMNS if column not in frame.columns]
    if missing or created_column not in frame.columns:
        required = list(SOURCE_CATEGORICAL_COLUMNS) + [created_column]
        absent = [column for column in required if column not in frame.columns]
        raise ValueError(
            "Cannot build the logging-time feature frame. Missing columns: "
            + ", ".join(absent)
            + ". Actual columns: "
            + ", ".join(map(str, frame.columns))
            + "."
        )
    dated = add_calendar_features(frame, created_column)
    features = dated.loc[:, list(MODEL_FEATURES)].copy()
    _reject_blocked_features(list(features.columns))
    return features


def _class_counts(outcomes: pd.Series) -> dict[str, int]:
    counts = outcomes.value_counts(dropna=False)
    return {label: int(counts.get(label, 0)) for label in TARGET_LABELS}


def chronological_splits(
    frame: pd.DataFrame,
    created_column: str = SERVICE_CREATED_DATE_COLUMN,
    target_column: str = "resolution_outcome",
    train_end: pd.Timestamp = TRAIN_END,
    validation_end: pd.Timestamp = VALIDATION_END,
    mature_end: pd.Timestamp = MATURE_END,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Split labelled rows by Created On Date.

    Train is through ``train_end``, validation through ``validation_end``,
    and the locked test through ``mature_end``. Later labelled rows are a
    censored tail and are not placed in any modelling split. Unlabelled rows
    are excluded. Nothing is resampled.
    """
    if created_column not in frame.columns or target_column not in frame.columns:
        missing = [
            column
            for column in (created_column, target_column)
            if column not in frame.columns
        ]
        raise ValueError(
            "Cannot split this frame. Missing columns: "
            + ", ".join(missing)
            + ". Actual columns: "
            + ", ".join(map(str, frame.columns))
            + "."
        )
    train_day = pd.Timestamp(train_end).normalize()
    validation_day = pd.Timestamp(validation_end).normalize()
    mature_day = pd.Timestamp(mature_end).normalize()
    if not (train_day < validation_day < mature_day):
        raise ValueError(
            "Split dates must increase: train_end < validation_end < mature_end. "
            f"Got {train_day.date()}, {validation_day.date()}, {mature_day.date()}."
        )

    created = pd.to_datetime(frame[created_column], errors="coerce").dt.normalize()
    labelled = frame[target_column].notna() & created.notna()
    mature = labelled & created.le(mature_day)
    masks = {
        "train": mature & created.le(train_day),
        "validation": mature & created.gt(train_day) & created.le(validation_day),
        "test": mature & created.gt(validation_day) & created.le(mature_day),
    }
    splits = {name: frame.loc[mask].copy() for name, mask in masks.items()}
    for name, part in splits.items():
        if part.empty:
            raise ValueError(
                f"The {name} split has no labelled rows. "
                f"Boundaries are {train_day.date()}, {validation_day.date()}, {mature_day.date()}."
            )

    rows = []
    for name, part in splits.items():
        part_created = pd.to_datetime(part[created_column])
        counts = _class_counts(part[target_column])
        total = int(len(part))
        rows.append(
            {
                "split": name,
                "rows": total,
                "start_date": str(part_created.min().date()),
                "end_date": str(part_created.max().date()),
                **{label: counts[label] for label in TARGET_LABELS},
                **{
                    f"{label} percent": (counts[label] / total * 100.0) if total else 0.0
                    for label in TARGET_LABELS
                },
            }
        )
    summary = pd.DataFrame(rows)
    exclusions = pd.DataFrame(
        [
            {
                "reason": "unlabelled_or_missing_created_date",
                "rows": int((~labelled).sum()),
            },
            {
                "reason": "labelled_after_mature_end",
                "rows": int((labelled & created.gt(mature_day)).sum()),
            },
            {
                "reason": "train",
                "rows": int(masks["train"].sum()),
            },
            {
                "reason": "validation",
                "rows": int(masks["validation"].sum()),
            },
            {
                "reason": "test",
                "rows": int(masks["test"].sum()),
            },
        ]
    )
    return splits, summary, exclusions


class GroupRareCategories(TransformerMixin, BaseEstimator):
    """Replace infrequent categories with one shared label.

    Counts are learned in ``fit``. A category that is new at transform time
    uses the same label, because it also has no training support. Missing
    values stay missing so a later imputer can give them their own level.
    """

    def __init__(self, min_count: int = RARE_CATEGORY_MIN_COUNT, other_label: str = RARE_CATEGORY_LABEL):
        self.min_count = min_count
        self.other_label = other_label

    def fit(self, X, y=None):
        if int(self.min_count) < 1:
            raise ValueError(f"min_count must be at least 1. Got {self.min_count}.")
        frame = self._frame(X, fitting=True)
        self.n_features_in_ = frame.shape[1]
        self.columns_ = [str(column) for column in frame.columns]
        self.kept_categories_ = {}
        self.training_counts_ = {}
        for column in self.columns_:
            text = self._clean_text(frame[column])
            if bool(text.eq(self.other_label).any()):
                raise ValueError(
                    f"Column {column} already contains the grouping label {self.other_label!r}. "
                    "Choose a label that is not a stored category."
                )
            counts = text.dropna().value_counts()
            self.training_counts_[column] = {str(name): int(count) for name, count in counts.items()}
            kept = {str(name) for name, count in counts.items() if int(count) >= int(self.min_count)}
            self.kept_categories_[column] = kept
        return self

    def transform(self, X):
        self._check_fitted()
        frame = self._frame(X, fitting=False)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} columns and received {frame.shape[1]}."
            )
        output = np.empty(frame.shape, dtype=object)
        for index, column in enumerate(self.columns_):
            text = self._clean_text(frame.iloc[:, index])
            missing = text.isna()
            known = text.isin(self.kept_categories_[column])
            grouped = text.astype("object")
            grouped = grouped.where(known | missing, self.other_label)
            grouped = grouped.where(~missing, np.nan)
            output[:, index] = grouped.to_numpy()
        return output

    def get_feature_names_out(self, input_features=None):
        self._check_fitted()
        return np.asarray(self.columns_, dtype=object)

    def rare_summary(self) -> pd.DataFrame:
        """Count how many training categories were kept or grouped."""
        self._check_fitted()
        rows = []
        for column in self.columns_:
            counts = self.training_counts_[column]
            kept = self.kept_categories_[column]
            total = int(sum(counts.values()))
            kept_rows = int(sum(count for name, count in counts.items() if name in kept))
            rows.append(
                {
                    "column": column,
                    "training_rows": total,
                    "distinct_categories": len(counts),
                    "kept_categories": len(kept),
                    "grouped_categories": len(counts) - len(kept),
                    "min_count": int(self.min_count),
                    "kept_row_percent": (kept_rows / total * 100.0) if total else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def _check_fitted(self) -> None:
        if not hasattr(self, "kept_categories_"):
            raise AttributeError("GroupRareCategories must be fit before transform.")

    def _frame(self, X, fitting: bool) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        array = np.asarray(X, dtype=object)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if fitting:
            names = [f"feature_{index}" for index in range(array.shape[1])]
        else:
            names = list(getattr(self, "columns_", []))
            if len(names) != array.shape[1]:
                names = [f"feature_{index}" for index in range(array.shape[1])]
        return pd.DataFrame(array, columns=names)

    @staticmethod
    def _clean_text(series: pd.Series) -> pd.Series:
        text = series.astype("string").str.strip()
        return text.mask(text.eq(""), pd.NA)


def build_baseline_preprocessor(
    categorical_columns: tuple[str, ...] | list[str] | None = None,
    numeric_columns: tuple[str, ...] | list[str] | None = None,
    min_count: int = RARE_CATEGORY_MIN_COUNT,
):
    """Build a ColumnTransformer that is fit on training rows only.

    Categorical columns are grouped, imputed with a constant missing level,
    and one-hot encoded. Unseen one-hot levels are ignored. Numeric columns
    are standardised. The transformer does not look at ``y``.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical = list(CATEGORICAL_FEATURES if categorical_columns is None else categorical_columns)
    numeric = list(NUMERIC_FEATURES if numeric_columns is None else numeric_columns)
    _reject_blocked_features(categorical + numeric)
    if not categorical and not numeric:
        raise ValueError("The baseline preprocessor needs at least one feature column.")

    transformers = []
    if categorical:
        categorical_pipe = Pipeline(
            steps=[
                ("rare", GroupRareCategories(min_count=min_count)),
                (
                    "impute",
                    SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY_LABEL),
                ),
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical))
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def make_most_frequent_baseline():
    """Predict the class that is most common in the training rows."""
    from sklearn.dummy import DummyClassifier

    from src.config import RANDOM_SEED

    return DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)


def make_logistic_baseline():
    """L2 logistic regression with balanced class weights.

    ``C`` stays at the library default. ``tol`` and ``max_iter`` are set so
    the sparse solver can finish on this extract. They were not chosen by
    searching validation scores. Balanced weights are used because the
    training years contain many more Resolved Fast rows than Unresolved rows.
    """
    from sklearn.linear_model import LogisticRegression

    from src.config import RANDOM_SEED

    return LogisticRegression(
        C=1.0,
        solver="saga",
        class_weight="balanced",
        max_iter=200,
        tol=1e-3,
        random_state=RANDOM_SEED,
    )
