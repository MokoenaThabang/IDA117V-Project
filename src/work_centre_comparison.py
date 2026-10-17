"""Fit the frozen booster with and without Work Center.

The champion excluded Work Center because the extract has no assignment time.
That exclusion, by itself, is not an ablation. This module fits both settings
on the training period and scores the 2025 validation period. It does not
read the locked test and it does not change the saved champion.
"""

from __future__ import annotations

import json
import time

import pandas as pd

from src import config
from src.evaluation import classification_scores, ranking_scores
from src.optimisation import _new_booster, fit_frozen_boosting, frozen_iteration
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    SOURCE_CATEGORICAL_COLUMNS,
    chronological_splits,
    logging_feature_frame,
)
from src.training import TreeCategoryEncoder, probabilities_in_label_order, save_json


def _score(model, encoder, features, labels) -> tuple[dict, dict]:
    coded = encoder.transform(features)
    predicted = model.predict(coded)
    probabilities = probabilities_in_label_order(model.predict_proba(coded), model.classes_)
    scores = classification_scores(labels, predicted)
    ranking = ranking_scores(labels, probabilities)
    return scores, ranking


def _rows(name: str, scores: dict, ranking: dict, fit_seconds: float) -> tuple[dict, list[dict]]:
    summary = {
        "model": name,
        "split": "validation",
        "accuracy": scores["accuracy"],
        "macro_f1": scores["macro_f1"],
        "weighted_f1": scores["weighted_f1"],
        "macro_roc_auc": ranking["macro_roc_auc"],
        "macro_average_precision": ranking["macro_average_precision"],
        "fit_seconds": fit_seconds,
    }
    per_class = []
    for row in scores["per_class"]:
        per_class.append({"model": name, "split": "validation", **row})
        summary[f"recall_{row['label'].lower().replace(' ', '_')}"] = row["recall"]
    return summary, per_class


def run_comparison() -> dict:
    """Refit both settings. Returns the written-file summary."""
    frozen_path = config.TABLES_DIR / "06_frozen_decision.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    columns = list(SOURCE_CATEGORICAL_COLUMNS) + [
        "Work_Center",
        "Created_On_Date",
        "resolution_outcome",
    ]
    frame = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "service_requests_labelled.parquet",
        columns=columns,
    )
    splits, _summary, _exclusions = chronological_splits(frame)
    del frame
    train = splits["train"]
    validation = splits["validation"]
    y_train = train["resolution_outcome"].to_numpy()
    y_validation = validation["resolution_outcome"].to_numpy()

    without_features = logging_feature_frame(train)
    without_validation = logging_feature_frame(validation)
    without_encoder, without_model, without_seconds = fit_frozen_boosting(
        without_features,
        y_train,
        frozen["parameters"],
        max_iter=int(frozen["max_iter"]),
    )
    without_scores, without_ranking = _score(
        without_model, without_encoder, without_validation, y_validation
    )

    with_train = without_features.copy()
    with_train["Work_Center"] = train["Work_Center"].to_numpy()
    with_validation = without_validation.copy()
    with_validation["Work_Center"] = validation["Work_Center"].to_numpy()
    categorical = list(CATEGORICAL_FEATURES) + ["Work_Center"]
    with_encoder = TreeCategoryEncoder(
        categorical_columns=categorical,
        allow_work_centre=True,
    )
    with_encoder.fit(with_train)
    coded_train = with_encoder.transform(with_train)
    with_model = _new_booster(
        frozen["parameters"],
        categorical,
        max_iter=frozen_iteration(int(frozen["max_iter"])),
        early_stopping=False,
        class_weight=frozen["class_weight"],
    )
    started = time.perf_counter()
    with_model.fit(coded_train, y_train)
    with_seconds = time.perf_counter() - started
    with_scores, with_ranking = _score(
        with_model, with_encoder, with_validation, y_validation
    )

    saved = pd.read_csv(config.TABLES_DIR / "06_candidate_metrics.csv")
    saved_row = saved.loc[
        saved["model"].eq(frozen["selected"]) & saved["split"].eq("validation")
    ].iloc[0]
    macro_gap = abs(float(without_scores["macro_f1"]) - float(saved_row["macro_f1"]))

    without_summary, without_class = _rows(
        "without_work_centre", without_scores, without_ranking, without_seconds
    )
    with_summary, with_class = _rows(
        "with_work_centre", with_scores, with_ranking, with_seconds
    )
    without_summary["delta_macro_f1_versus_without"] = 0.0
    with_summary["delta_macro_f1_versus_without"] = (
        float(with_scores["macro_f1"]) - float(without_scores["macro_f1"])
    )
    comparison = pd.DataFrame([without_summary, with_summary])
    per_class = pd.DataFrame(without_class + with_class)
    config_payload = {
        "purpose": (
            "Sensitivity comparison. The champion excluded Work Center because "
            "the extract has no assignment time. Both models were fit for this table. "
            "The champion file was not replaced."
        ),
        "leakage_status": (
            "Work Center is present on the request. No assignment timestamp is in the extract, "
            "so this column is not known to be available at logging. "
            "The with-work-centre model is not a logging-time champion."
        ),
        "split": "validation 2025, same chronological split as Phase 4",
        "locked_test_scored": False,
        "parameters": frozen["parameters"],
        "max_iter": int(frozen["max_iter"]),
        "class_weight": frozen["class_weight"],
        "early_stopping": False,
        "seed": int(config.RANDOM_SEED),
        "without_features": list(without_features.columns),
        "with_features": list(with_train.columns),
        "saved_champion_validation_macro_f1": float(saved_row["macro_f1"]),
        "without_model_minus_saved_champion_macro_f1": float(macro_gap),
        "champion_changed": False,
    }
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = config.TABLES_DIR / "09_work_centre_comparison.csv"
    per_class_path = config.TABLES_DIR / "09_work_centre_per_class.csv"
    config_path = config.TABLES_DIR / "09_work_centre_config.json"
    comparison.to_csv(comparison_path, index=False)
    per_class.to_csv(per_class_path, index=False)
    save_json(config_payload, config_path)
    print(comparison.round(4).to_string(index=False))
    print(f"Absolute macro-F1 gap versus the saved champion validation row: {macro_gap:.6f}")
    print(f"Wrote {comparison_path}")
    return config_payload


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()
