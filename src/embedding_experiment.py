"""Fit and record the approved embedding network.

The architecture is the one already specified in ``EMBEDDING_PARAMETERS``.
Lecturer approval was recorded from the project instruction. No approval
date and no new architecture were supplied. Work centre is not an input.

The champion rule is validation macro-F1 among approved models. This module
does not read the locked test.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from src import config
from src.neural import EmbeddingNet, predict_embedding, train_embedding_network
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


def artifact_directory(checkpoint_exists: bool, overwrite_checkpoint: bool) -> Path:
    """Keep an existing checkpoint unless an explicit overwrite was requested."""
    if checkpoint_exists and not overwrite_checkpoint:
        return config.PROJECT_ROOT / "evidence" / "reproducibility" / "embedding_retrain"
    return config.PROJECT_ROOT


def _load_train_validation() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load logging features. The locked test split is deleted before encoding."""
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
    y_train = train["resolution_outcome"]
    y_validation = validation["resolution_outcome"]
    del splits
    return features_train, features_validation, y_train, y_validation


def _score_encoded(model, encoder_result, batch_size: int, fit_seconds: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    class_rows = []
    for split_name, encoded, split_labels in encoder_result:
        predicted, probabilities, _loss = predict_embedding(
            model,
            encoded["codes"],
            encoded["numeric"],
            None,
            batch_size,
        )
        summary, per_class = summarise_predictions(
            split_labels,
            predicted,
            probabilities,
            model="embedding_network",
            split=split_name,
            approval=APPROVED_APPROVAL,
            fit_seconds=fit_seconds,
        )
        summaries.append(summary)
        class_rows.append(per_class)
    return pd.concat(summaries, ignore_index=True), pd.concat(class_rows, ignore_index=True)


def metric_differences(reproduced: pd.DataFrame, saved: pd.DataFrame, columns: list[str]) -> dict:
    """Maximum absolute gap by split for columns present in both tables."""
    gaps = {}
    for split_name in ("train", "validation"):
        left = reproduced.loc[reproduced["split"].eq(split_name)]
        right = saved.loc[saved["split"].eq(split_name)]
        if left.empty or right.empty:
            gaps[split_name] = {"rows_present": False}
            continue
        split_gaps = {}
        for column in columns:
            if column not in left.columns or column not in right.columns:
                continue
            split_gaps[column] = abs(float(left.iloc[0][column]) - float(right.iloc[0][column]))
        gaps[split_name] = split_gaps
    return gaps


def run_experiment(destination: Path | None = None) -> dict:
    """Fit the approved network on the training period and score validation."""
    features_train, features_validation, y_train, y_validation = _load_train_validation()
    y_train = y_train.to_numpy()
    y_validation = y_validation.to_numpy()
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
    encoded_splits = (
        ("train", train_encoded, y_train),
        ("validation", validation_encoded, y_validation),
    )
    metrics, per_class = _score_encoded(
        result["model"], encoded_splits, batch_size, result["fit_seconds"]
    )
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
    destination = config.PROJECT_ROOT if destination is None else destination
    official = destination.resolve() == config.PROJECT_ROOT.resolve()
    model_dir = config.MODELS_DIR if official else destination / "models"
    table_dir = config.TABLES_DIR if official else destination / "tables"
    figure_dir = config.FIGURES_DIR if official else destination / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": result["model"].state_dict(),
        "cardinalities": result["cardinalities"],
        "parameters": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in result["parameters"].items()
        },
        "best_epoch": int(result["best_epoch"]),
        "epochs_ran": int(result["epochs_ran"]),
        "class_weights": result["class_weights"],
    }
    model_path = model_dir / "embedding_network.pt"
    encoder_path = model_dir / "embedding_encoder.joblib"
    torch.save(checkpoint, model_path)
    joblib.dump(encoder, encoder_path)
    metrics.to_csv(table_dir / "10_embedding_metrics.csv", index=False)
    per_class.to_csv(table_dir / "10_embedding_per_class.csv", index=False)
    history.to_csv(table_dir / "10_embedding_history.csv", index=False)
    encoder.level_summary().to_csv(table_dir / "10_embedding_levels.csv", index=False)
    _figure, figure_path = plot_learning_curves(
        history,
        "Approved embedding network",
        "10_embedding_learning_curves.png",
        directory=figure_dir,
    )
    save_json(decision, table_dir / "10_embedding_champion_decision.json")
    config_payload = {
        "command": "python -m src.embedding_experiment",
        "features": list(MODEL_FEATURES),
        "work_centre_included": False,
        "fit_on": "train",
        "scored_splits": ["train", "validation"],
        "locked_test_scored": False,
        "official_artifacts_replaced": official,
        "model_file": str(model_path.relative_to(config.PROJECT_ROOT)),
        "encoder_file": str(encoder_path.relative_to(config.PROJECT_ROOT)),
        "learning_curve": str(figure_path.relative_to(config.PROJECT_ROOT)),
        "parameters": checkpoint["parameters"],
    }
    save_json(config_payload, table_dir / "10_embedding_config.json")
    print(json.dumps(decision, indent=2))
    print(metrics.round(4).to_string(index=False))
    return {
        "decision": decision,
        "metrics": metrics,
        "history": history,
        "destination": str(destination),
        "official_artifacts_replaced": official,
        "learning_curve": str(figure_path.relative_to(config.PROJECT_ROOT)),
    }


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_saved_checkpoint() -> dict:
    """Score the saved checkpoint on train and validation. Do not write the checkpoint."""
    model_path = config.MODELS_DIR / "embedding_network.pt"
    encoder_path = config.MODELS_DIR / "embedding_encoder.joblib"
    if not model_path.is_file() or not encoder_path.is_file():
        raise FileNotFoundError("The saved embedding checkpoint is missing.")
    features_train, features_validation, y_train, y_validation = _load_train_validation()
    encoder = joblib.load(encoder_path)
    train_encoded = encoder.transform(features_train)
    validation_encoded = encoder.transform(features_validation)
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    parameters = payload["parameters"]
    model = EmbeddingNet(
        cardinalities=list(payload["cardinalities"]),
        embedding_dim=int(parameters["embedding_dim"]),
        hidden_size=int(parameters["hidden_size"]),
        dropout=float(parameters["dropout"]),
        n_numeric=int(train_encoded["numeric"].shape[1]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metrics, per_class = _score_encoded(
        model,
        (
            ("train", train_encoded, y_train.to_numpy()),
            ("validation", validation_encoded, y_validation.to_numpy()),
        ),
        int(parameters["batch_size"]),
        float("nan"),
    )
    destination = config.PROJECT_ROOT / "evidence" / "reproducibility" / "embedding_saved_eval"
    table_dir = destination / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(table_dir / "10_embedding_metrics.csv", index=False)
    per_class.to_csv(table_dir / "10_embedding_per_class.csv", index=False)
    saved = pd.read_csv(config.TABLES_DIR / "10_embedding_metrics.csv")
    gaps = metric_differences(
        metrics,
        saved,
        ["accuracy", "macro_f1", "weighted_f1", "recall_resolved_fast", "recall_resolved_slow", "recall_unresolved"],
    )
    return {"metrics": metrics, "gaps_against_saved_table": gaps, "table_dir": str(table_dir)}


def _history_gaps(reproduced: pd.DataFrame, saved: pd.DataFrame) -> dict:
    columns = ["train_loss", "validation_loss", "train_macro_f1", "validation_macro_f1"]
    if len(reproduced) != len(saved):
        return {"epoch_rows": {"reproduced": int(len(reproduced)), "saved": int(len(saved))}}
    gaps = {}
    for column in columns:
        gaps[column] = float((reproduced[column] - saved[column]).abs().max())
    return gaps


def reproduce(overwrite_checkpoint: bool = False) -> dict:
    """Retrain beside the saved checkpoint and compare. The locked test is not read."""
    model_path = config.MODELS_DIR / "embedding_network.pt"
    encoder_path = config.MODELS_DIR / "embedding_encoder.joblib"
    before = (_file_sha256(model_path), _file_sha256(encoder_path))
    saved_eval = evaluate_saved_checkpoint() if model_path.is_file() and encoder_path.is_file() else None
    import gc

    gc.collect()
    destination = artifact_directory(model_path.is_file(), overwrite_checkpoint)
    trained = run_experiment(destination)
    after = (_file_sha256(model_path), _file_sha256(encoder_path))
    if not overwrite_checkpoint and before != after:
        raise RuntimeError("The saved embedding checkpoint was overwritten.")
    saved_metrics = pd.read_csv(config.TABLES_DIR / "10_embedding_metrics.csv")
    saved_history = pd.read_csv(config.TABLES_DIR / "10_embedding_history.csv")
    metric_gaps = metric_differences(
        trained["metrics"],
        saved_metrics,
        ["accuracy", "macro_f1", "weighted_f1", "recall_resolved_fast", "recall_resolved_slow", "recall_unresolved"],
    )
    history_gaps = _history_gaps(trained["history"], saved_history)
    record = {
        "command": "python -m src.embedding_experiment",
        "retrains_model": True,
        "saved_checkpoint_overwritten": bool(overwrite_checkpoint),
        "checkpoint_sha256_unchanged": before == after,
        "locked_test_scored": False,
        "champion_changed": False,
        "champion_rule": trained["decision"]["rule"],
        "champion_model": trained["decision"]["champion_model"],
        "corrected_learning_curve": "results/figures/10_embedding_learning_curves.png",
        "retrain_learning_curve": trained["learning_curve"],
        "retrain_destination": trained["destination"],
        "saved_checkpoint_metric_gaps": None if saved_eval is None else saved_eval["gaps_against_saved_table"],
        "retrain_metric_gaps_against_saved_table": metric_gaps,
        "retrain_history_gaps_against_saved_table": history_gaps,
        "retrain_validation_macro_f1": trained["decision"]["saved_refit_validation_macro_f1"],
        "recorded_validation_macro_f1": float(
            saved_metrics.loc[saved_metrics["split"].eq("validation"), "macro_f1"].iloc[0]
        ),
    }
    save_json(record, config.TABLES_DIR / "12_embedding_reproduction.json")
    print(json.dumps(record, indent=2))
    return record


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce the approved embedding network without using the locked test.")
    parser.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help="Replace models/embedding_network.pt. Omitted runs write beside that file.",
    )
    args = parser.parse_args()
    reproduce(overwrite_checkpoint=args.overwrite_checkpoint)


if __name__ == "__main__":
    main()
