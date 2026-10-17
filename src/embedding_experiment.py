"""Fit and record the approved embedding network.

The architecture is the one already specified in ``EMBEDDING_PARAMETERS``.
Lecturer approval was recorded from the project instruction. No approval
date and no new architecture were supplied. Work centre is not an input.

The champion rule is validation macro-F1 among approved models. This module
does not read the locked test.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import torch

from src import config
from src.neural import predict_embedding, train_embedding_network
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    SOURCE_CATEGORICAL_COLUMNS,
    chronological_splits,
    logging_feature_frame,
)
from src.training import (
    APPROVED_APPROVAL,
    EmbeddingCategoryEncoder,
    provisional_champion,
    save_json,
    summarise_predictions,
)
from src.visualization import plot_learning_curves

PHASE5_METRICS = config.TABLES_DIR / "05_split_metrics.csv"
FROZEN_METRICS = config.TABLES_DIR / "06_candidate_metrics.csv"


def champion_decision_from_metrics(
    metrics: pd.DataFrame,
    frozen_validation_macro_f1: float,
) -> dict:
    """Apply the validation rule once the embedding network is approved.

    Pending rows are still ignored. The locked test is not a column this
    function reads. A second training run is not a reason to replace the
    model that won this comparison.
    """
    required = {"model", "split", "approval", "macro_f1", "recall_unresolved", "recall_resolved_slow"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError("Champion metrics are missing columns: " + ", ".join(sorted(missing)) + ".")
    comparable = metrics.copy()
    embedding = comparable["model"].eq("embedding_network")
    if not embedding.any():
        raise ValueError("The metrics table has no embedding_network rows.")
    comparable.loc[embedding, "approval"] = APPROVED_APPROVAL
    chosen = provisional_champion(comparable)
    validation = comparable.loc[comparable["split"].eq("validation")]
    embedding_score = float(validation.loc[validation["model"].eq("embedding_network"), "macro_f1"].iloc[0])
    champion_name = str(chosen["model"])
    return {
        "rule": (
            "Highest validation macro-F1 among approved models. "
            "Ties break on Unresolved recall, then Resolved Slow recall. "
            "The locked test is not an input."
        ),
        "champion_model": champion_name,
        "champion_validation_macro_f1": float(chosen["macro_f1"]),
        "embedding_validation_macro_f1": embedding_score,
        "frozen_boosting_validation_macro_f1": float(frozen_validation_macro_f1),
        "champion_changed": champion_name != "hist_gradient_boosting",
        "delivered_model": "frozen histogram gradient boosting pipeline",
        "locked_test_used": False,
        "approval_date": None,
        "architecture_changed": False,
        "work_centre_included": False,
    }


def _frozen_validation_macro_f1() -> float:
    saved = pd.read_csv(FROZEN_METRICS)
    row = saved.loc[saved["model"].eq("untuned") & saved["split"].eq("validation")]
    if row.empty:
        raise RuntimeError("The frozen untuned validation row is missing.")
    return float(row.iloc[0]["macro_f1"])


def run_experiment() -> dict:
    """Fit the approved network on the training period and score validation."""
    source_columns = list(SOURCE_CATEGORICAL_COLUMNS) + ["Created_On_Date", "resolution_outcome"]
    frame = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "service_requests_labelled.parquet",
        columns=source_columns,
    )
    splits, _summary, _exclusions = chronological_splits(frame)
    del frame
    del splits["test"]
    train = splits["train"]
    validation = splits["validation"]
    features_train = logging_feature_frame(train)
    features_validation = logging_feature_frame(validation)
    if "Work_Center" in features_train.columns or "Work_Center" in CATEGORICAL_FEATURES:
        raise RuntimeError("Work centre is not an embedding input.")
    y_train = train["resolution_outcome"].to_numpy()
    y_validation = validation["resolution_outcome"].to_numpy()
    del splits

    encoder = EmbeddingCategoryEncoder().fit(features_train)
    train_encoded = encoder.transform(features_train)
    validation_encoded = encoder.transform(features_validation)
    result = train_embedding_network(
        train_encoded["codes"],
        train_encoded["numeric"],
        y_train,
        validation_encoded["codes"],
        validation_encoded["numeric"],
        y_validation,
        encoder.cardinalities_,
        verbose=True,
    )
    batch_size = int(result["parameters"]["batch_size"])
    summaries = []
    class_rows = []
    for split_name, encoded, labels in (
        ("train", train_encoded, y_train),
        ("validation", validation_encoded, y_validation),
    ):
        predicted, probabilities, _loss = predict_embedding(
            result["model"],
            encoded["codes"],
            encoded["numeric"],
            None,
            batch_size,
        )
        summary, per_class = summarise_predictions(
            labels,
            predicted,
            probabilities,
            model="embedding_network",
            split=split_name,
            approval=APPROVED_APPROVAL,
            fit_seconds=result["fit_seconds"],
        )
        summaries.append(summary)
        class_rows.append(per_class)

    metrics = pd.concat(summaries, ignore_index=True)
    per_class = pd.concat(class_rows, ignore_index=True)
    history = result["history"]
    phase5 = pd.read_csv(PHASE5_METRICS)
    decision = champion_decision_from_metrics(phase5, _frozen_validation_macro_f1())
    refit_validation = float(metrics.loc[metrics["split"].eq("validation"), "macro_f1"].iloc[0])
    decision.update(
        {
            "approval_source": (
                "The project instruction stated that the lecturer approved the embedding network. "
                "No approval date and no additional architecture details were supplied."
            ),
            "score_used_for_champion_rule": "results/tables/05_split_metrics.csv",
            "saved_refit_validation_macro_f1": refit_validation,
            "saved_refit_is_a_second_vote": False,
            "features": list(MODEL_FEATURES),
            "best_epoch": int(result["best_epoch"]),
            "epochs_ran": int(result["epochs_ran"]),
            "n_parameters": int(result["n_parameters"]),
            "fit_seconds": float(result["fit_seconds"]),
        }
    )
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "embedding_network.pt"
    encoder_path = config.MODELS_DIR / "embedding_encoder.joblib"
    torch.save(
        {
            "state_dict": result["model"].state_dict(),
            "cardinalities": result["cardinalities"],
            "parameters": {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in result["parameters"].items()
            },
            "best_epoch": int(result["best_epoch"]),
            "epochs_ran": int(result["epochs_ran"]),
            "class_weights": result["class_weights"],
        },
        model_path,
    )
    joblib.dump(encoder, encoder_path)
    metrics.to_csv(config.TABLES_DIR / "10_embedding_metrics.csv", index=False)
    per_class.to_csv(config.TABLES_DIR / "10_embedding_per_class.csv", index=False)
    history.to_csv(config.TABLES_DIR / "10_embedding_history.csv", index=False)
    encoder.level_summary().to_csv(config.TABLES_DIR / "10_embedding_levels.csv", index=False)
    plot_learning_curves(
        history,
        "Approved embedding network",
        "10_embedding_learning_curves.png",
    )
    save_json(decision, config.TABLES_DIR / "10_embedding_champion_decision.json")
    config_payload = {
        "command": "python -m src.embedding_experiment",
        "features": list(MODEL_FEATURES),
        "work_centre_included": False,
        "fit_on": "train",
        "scored_splits": ["train", "validation"],
        "locked_test_scored": False,
        "model_file": "models/embedding_network.pt",
        "encoder_file": "models/embedding_encoder.joblib",
        "parameters": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in result["parameters"].items()
        },
    }
    save_json(config_payload, config.TABLES_DIR / "10_embedding_config.json")
    print(json.dumps(decision, indent=2))
    print(metrics.round(4).to_string(index=False))
    return decision


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
