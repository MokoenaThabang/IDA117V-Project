"""Write the frozen Phase 6 pipeline to models/ for the Streamlit app.

This script fits the already chosen hyperparameters on the training period.
It does not search, and it does not change the settings after the locked test.
The Streamlit app does not call this module.
"""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from src import config
from src.optimisation import fit_frozen_boosting
from src.preprocessing import (
    MISSING_CATEGORY_LABEL,
    MODEL_FEATURES,
    RARE_CATEGORY_LABEL,
    SOURCE_CATEGORICAL_COLUMNS,
    chronological_splits,
    logging_feature_frame,
)
from src.training import probabilities_in_label_order
from app.predict import (
    PIPELINE_FORMAT,
    SUPPORTED_DATE_END,
    SUPPORTED_DATE_START,
    TRAINING_DATE_END,
    pipeline_path,
)


def _kept_levels(encoder, column: str) -> list[str]:
    internal = {MISSING_CATEGORY_LABEL, RARE_CATEGORY_LABEL}
    levels = [name for name in encoder.code_maps_[column] if name not in internal]
    return sorted(levels, key=str.casefold)


# Stored placeholders documented in the feature decisions. They stay in the
# dropdowns. The demonstration sample skips them so the form shows a named
# category rather than a placeholder.
SAMPLE_PLACEHOLDERS = {"#", "Not assigned"}


def _most_common_kept(series: pd.Series, kept: list[str]) -> str:
    allowed = set(kept) - SAMPLE_PLACEHOLDERS
    text = series.astype("string").str.strip()
    counts = text[text.isin(allowed)].value_counts()
    if counts.empty:
        raise RuntimeError("No training value matched the kept category list.")
    return str(counts.index[0])


def _sample_input(train: pd.DataFrame, encoder) -> dict:
    options = {column: _kept_levels(encoder, column) for column in SOURCE_CATEGORICAL_COLUMNS}
    sample = {
        column: _most_common_kept(train[column], options[column])
        for column in SOURCE_CATEGORICAL_COLUMNS
    }
    sample["Created_On_Date"] = "2024-06-12"
    sample["description"] = (
        "Demonstration only. Each field is a frequent training category, skipping the stored "
        "placeholders '#' and 'Not assigned'. Those placeholders remain in the dropdowns. "
        "The fields are entered together so the form can be tried. "
        "This is not a logged municipal request and it is not an unresolved case."
    )
    return sample


def build_artifact(force: bool = False) -> dict:
    destination = pipeline_path()
    if destination.is_file() and not force:
        raise FileExistsError(
            f"{destination} already exists. Pass --force to write it again. "
            "The app will not do that itself."
        )
    frozen_path = config.TABLES_DIR / "06_frozen_decision.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(f"Missing frozen decision: {frozen_path}.")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if list(frozen["features"]) != list(MODEL_FEATURES):
        raise RuntimeError("The frozen decision features do not match the training feature list.")
    if frozen.get("early_stopping", True):
        raise RuntimeError("The frozen decision still has early stopping turned on.")
    if "Work_Center" in frozen["features"]:
        raise RuntimeError("The frozen decision includes Work Center.")

    columns = list(SOURCE_CATEGORICAL_COLUMNS) + ["Created_On_Date", "resolution_outcome"]
    frame = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "service_requests_labelled.parquet",
        columns=columns,
    )
    splits, _summary, _exclusions = chronological_splits(frame)
    del frame
    train = splits["train"]
    features = logging_feature_frame(train)
    labels = train["resolution_outcome"].to_numpy()
    encoder, model, fit_seconds = fit_frozen_boosting(
        features,
        labels,
        frozen["parameters"],
        max_iter=int(frozen["max_iter"]),
    )
    if bool(model.early_stopping):
        raise RuntimeError("The refit booster has early stopping turned on.")
    if int(model.max_iter) != int(frozen["max_iter"]):
        raise RuntimeError("The refit iteration cap does not match the frozen decision.")

    comparison = _compare_locked_test(encoder, model, splits["test"])
    sample = _sample_input(train, encoder)
    artifact = {
        "format": PIPELINE_FORMAT,
        "encoder": encoder,
        "model": model,
        "features": list(MODEL_FEATURES),
        "category_options": {
            column: _kept_levels(encoder, column) for column in SOURCE_CATEGORICAL_COLUMNS
        },
        "sample_input": sample,
        "parameters": frozen["parameters"],
        "max_iter": int(frozen["max_iter"]),
        "class_weight": frozen["class_weight"],
        "selected": frozen["selected"],
        "champion_family": frozen["champion_family"],
        "label_threshold_days": int(config.FAST_SLOW_THRESHOLD_DAYS),
        "observation_window_days": int(config.OBSERVATION_WINDOW_DAYS),
        "supported_date_start": SUPPORTED_DATE_START.isoformat(),
        "supported_date_end": SUPPORTED_DATE_END.isoformat(),
        "training_date_end": TRAINING_DATE_END.isoformat(),
        "fit_seconds": float(fit_seconds),
        "locked_test_comparison": comparison,
        "excluded_fields": ["Work_Center"],
        "work_centre_reason": (
            "Excluded because the extract has no assignment time. "
            "The app does not collect it."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    print(f"Wrote {destination}")
    print(
        "Locked-test prediction agreement: "
        f"{comparison['matching_predictions']} / {comparison['rows']}."
    )
    print("Demonstration sample:", {key: sample[key] for key in SOURCE_CATEGORICAL_COLUMNS})
    return artifact


def _compare_locked_test(encoder, model, test: pd.DataFrame) -> dict:
    """Check the refit against the predictions already stored by notebook 06."""
    saved_path = config.TABLES_DIR / "06_test_predictions.parquet"
    if not saved_path.is_file():
        return {"compared": False, "reason": f"Missing {saved_path.name}."}
    saved = pd.read_parquet(saved_path)
    features = logging_feature_frame(test)
    coded = encoder.transform(features)
    predicted = model.predict(coded)
    probabilities = probabilities_in_label_order(model.predict_proba(coded), model.classes_)
    fresh = pd.DataFrame(
        {
            "source_index": test.index.to_numpy(),
            "refit_predicted": predicted,
            "refit_p_resolved_fast": probabilities[:, 0],
            "refit_p_resolved_slow": probabilities[:, 1],
            "refit_p_unresolved": probabilities[:, 2],
        }
    )
    merged = saved.merge(fresh, on="source_index", how="inner")
    if len(merged) != len(saved) or len(merged) != len(fresh):
        raise RuntimeError(
            "The refit rows do not align with the saved locked-test predictions. "
            f"Saved {len(saved)}, refit {len(fresh)}, aligned {len(merged)}."
        )
    label_matches = int((merged["champion_predicted"] == merged["refit_predicted"]).sum())
    probability_gap = float(
        (
            merged[["p_resolved_fast", "p_resolved_slow", "p_unresolved"]].to_numpy()
            - merged[["refit_p_resolved_fast", "refit_p_resolved_slow", "refit_p_unresolved"]].to_numpy()
        )
        .__abs__()
        .max()
    )
    if label_matches != len(merged) or probability_gap > 1e-6:
        raise RuntimeError(
            "The refit does not reproduce the locked-test predictions from notebook 06. "
            f"Matching labels {label_matches} of {len(merged)}. "
            f"Largest probability gap {probability_gap}."
        )
    return {
        "compared": True,
        "rows": int(len(merged)),
        "matching_predictions": label_matches,
        "max_probability_gap": probability_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the frozen resolution pipeline.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing pipeline file.",
    )
    args = parser.parse_args()
    build_artifact(force=args.force)


if __name__ == "__main__":
    main()
