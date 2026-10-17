"""Score one logging-time request with the saved pipeline.

The fitted encoder and booster are loaded from disk. Nothing in this module
fits a model or learns a category list.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.preprocessing import (
    MISSING_CATEGORY_LABEL,
    MODEL_FEATURES,
    RARE_CATEGORY_LABEL,
    SERVICE_CREATED_DATE_COLUMN,
    SOURCE_CATEGORICAL_COLUMNS,
    logging_feature_frame,
)
from src.training import _clean_category, probabilities_in_label_order

PIPELINE_FILENAME = "frozen_resolution_pipeline.joblib"
PIPELINE_FORMAT = "ida117v-frozen-hgb-v1"
SUPPORTED_DATE_START = date(2023, 1, 1)
SUPPORTED_DATE_END = date(2026, 10, 6)
TRAINING_DATE_END = date(2024, 12, 31)

# Collected by the form. Work centre is intentionally absent.
REQUEST_FIELDS = SOURCE_CATEGORICAL_COLUMNS + (SERVICE_CREATED_DATE_COLUMN,)
FIELD_LABELS = {
    "C3_Complaint_Type": "Complaint type",
    "Suburb": "Suburb",
    "Ward": "Ward",
    "Sub_Council": "Subcouncil",
    SERVICE_CREATED_DATE_COLUMN: "Request creation date",
}
EXCLUDED_INPUTS = (
    "Work_Center",
    "Work Centre",
    "work_centre",
    "Completed_Date",
    "Completed Date",
    "Changed_on",
    "Changed on",
    "resolution_days",
    "resolution_outcome",
    "Notification",
    "ObjectId",
)

ACADEMIC_NOTICE = (
    "This is an academic prototype for an IDA117V project. "
    "It is not an official City of Cape Town system."
)
SERVICE_WARNING = (
    "Predictions must support human review. "
    "They must not be used to deny or reduce service to an area."
)
WORK_CENTRE_NOTICE = (
    "Work centre is not collected. The fitted pipeline excludes it because "
    "the extract has no assignment time, so it is not treated as known when a request is logged."
)


class InputError(ValueError):
    """The request cannot be scored as entered."""


def pipeline_path(directory: Path | None = None) -> Path:
    folder = config.MODELS_DIR if directory is None else Path(directory)
    return folder / PIPELINE_FILENAME


def load_pipeline(path: Path | None = None) -> dict:
    """Load the exported pipeline. This does not train or preprocess-fit."""
    artifact_path = pipeline_path() if path is None else Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            "The fitted pipeline was not found at "
            f"{artifact_path}. Run python -m app.export_frozen_pipeline "
            "from the project folder to write it. This application does not train a model."
        )
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or artifact.get("format") != PIPELINE_FORMAT:
        raise RuntimeError(
            f"{artifact_path} is not a {PIPELINE_FORMAT} pipeline file."
        )
    features = list(artifact.get("features", []))
    if features != list(MODEL_FEATURES):
        raise RuntimeError(
            "The saved pipeline features do not match the training feature list. "
            f"Saved: {features}. Training: {list(MODEL_FEATURES)}."
        )
    if "Work_Center" in features:
        raise RuntimeError("The saved pipeline includes Work Center, which training excluded.")
    encoder = artifact.get("encoder")
    model = artifact.get("model")
    if encoder is None or model is None:
        raise RuntimeError("The pipeline file has no encoder or model.")
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("The saved model has no predict_proba method.")
    if bool(getattr(model, "early_stopping", True)):
        raise RuntimeError("The saved booster still has early stopping turned on.")
    options = artifact.get("category_options")
    if not isinstance(options, dict):
        raise RuntimeError("The pipeline file has no category options.")
    for column in SOURCE_CATEGORICAL_COLUMNS:
        values = options.get(column)
        if not values:
            raise RuntimeError(f"The pipeline file has no {column} options.")
    sample = artifact.get("sample_input")
    if not isinstance(sample, dict):
        raise RuntimeError("The pipeline file has no demonstration sample.")
    return artifact


def _reject_excluded_fields(payload: dict) -> None:
    present = [name for name in EXCLUDED_INPUTS if name in payload]
    if present:
        raise InputError(
            "These fields are not inputs to the fitted pipeline: "
            + ", ".join(present)
            + ". Work centre, completion fields, identifiers, and the outcome are excluded."
        )


def _required_text(payload: dict, column: str) -> str:
    if column not in payload:
        raise InputError(f"Choose a {FIELD_LABELS[column].lower()}.")
    value = payload[column]
    if value is None:
        raise InputError(f"Choose a {FIELD_LABELS[column].lower()}.")
    text = str(value).strip()
    if text == "" or text.lower() == "select...":
        raise InputError(f"Choose a {FIELD_LABELS[column].lower()}.")
    return text


def parse_created_date(value) -> date:
    """Return a calendar day inside the project extract, or raise InputError."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise InputError("Enter the date the request was created.")
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    elif isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise InputError("The creation date is not a valid calendar date.")
        parsed = value.date()
    else:
        stamp = pd.to_datetime(value, errors="coerce")
        if stamp is None or pd.isna(stamp):
            raise InputError(
                "The creation date is not a valid calendar date. Use a date such as 2024-06-12."
            )
        parsed = pd.Timestamp(stamp).date()
    if parsed < SUPPORTED_DATE_START or parsed > SUPPORTED_DATE_END:
        raise InputError(
            "The creation date must fall between 1 January 2023 and 6 October 2026, "
            "the period covered by the project extract."
        )
    return parsed


def validate_request(payload: dict) -> dict:
    """Check one request and return the fields the pipeline is allowed to see."""
    if not isinstance(payload, dict):
        raise InputError("The request must be a set of named fields.")
    _reject_excluded_fields(payload)
    cleaned = {column: _required_text(payload, column) for column in SOURCE_CATEGORICAL_COLUMNS}
    cleaned[SERVICE_CREATED_DATE_COLUMN] = parse_created_date(payload.get(SERVICE_CREATED_DATE_COLUMN))
    return cleaned


def _unknown_categories(encoder, features: pd.DataFrame) -> list[str]:
    unknown = []
    internal = {MISSING_CATEGORY_LABEL, RARE_CATEGORY_LABEL}
    for column in encoder.categorical_columns_:
        if column not in SOURCE_CATEGORICAL_COLUMNS:
            continue
        text = _clean_category(features[column]).iloc[0]
        if pd.isna(text):
            unknown.append(column)
            continue
        known = {name for name in encoder.code_maps_[column] if name not in internal}
        if str(text) not in known:
            unknown.append(column)
    return unknown


def _class_meaning(label: str, threshold_days: int) -> str:
    meanings = {
        "Resolved Fast": (
            f"Under the project label, Resolved Fast means a request completed "
            f"within {threshold_days} calendar days of creation."
        ),
        "Resolved Slow": (
            f"Under the project label, Resolved Slow means a request completed "
            f"after more than {threshold_days} calendar days."
        ),
        "Unresolved": (
            f"Under the project label, Unresolved means a request that was still open "
            f"after the {threshold_days}-day observation window."
        ),
    }
    if label not in meanings:
        raise RuntimeError(f"The model returned an unexpected class: {label}.")
    return meanings[label]


def predict_request(pipeline: dict, payload: dict) -> dict:
    """Validate one request and score it with the loaded encoder and model."""
    cleaned = validate_request(payload)
    frame = pd.DataFrame(
        [
            {
                **{column: cleaned[column] for column in SOURCE_CATEGORICAL_COLUMNS},
                SERVICE_CREATED_DATE_COLUMN: pd.Timestamp(cleaned[SERVICE_CREATED_DATE_COLUMN]),
            }
        ]
    )
    features = logging_feature_frame(frame)
    if list(features.columns) != list(MODEL_FEATURES):
        raise RuntimeError("The logging-time feature frame does not match the training columns.")
    encoder = pipeline["encoder"]
    model = pipeline["model"]
    coded = encoder.transform(features)
    predicted = str(model.predict(coded)[0])
    probabilities = probabilities_in_label_order(
        model.predict_proba(coded),
        model.classes_,
    )[0]
    probability_map = {
        label: float(probabilities[index]) for index, label in enumerate(config.TARGET_LABELS)
    }
    unknown = _unknown_categories(encoder, features)
    notes = []
    created = cleaned[SERVICE_CREATED_DATE_COLUMN]
    if created > TRAINING_DATE_END:
        notes.append(
            "This creation date is after 31 December 2024. The booster was fit on requests "
            "through that date. Later dates were used only for evaluation, not for choosing settings."
        )
    if unknown:
        labels = ", ".join(FIELD_LABELS[column] for column in unknown)
        notes.append(
            f"{labels} was not a kept training category. The saved encoder groups that value "
            "with rare or unseen levels. This is a coarser input, not a new category the model learned."
        )
    threshold = int(pipeline.get("label_threshold_days", config.FAST_SLOW_THRESHOLD_DAYS))
    model_inputs = {column: features.iloc[0][column] for column in MODEL_FEATURES}
    model_inputs["created_is_weekend"] = float(model_inputs["created_is_weekend"])
    return {
        "prediction": predicted,
        "probabilities": probability_map,
        "meaning": _class_meaning(predicted, threshold),
        "model_inputs": model_inputs,
        "unknown_categories": unknown,
        "notes": notes,
        "academic_notice": ACADEMIC_NOTICE,
        "service_warning": SERVICE_WARNING,
    }
