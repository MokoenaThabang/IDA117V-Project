"""Question 6.2 and the common locked-test ROC figure.

The champion stays the frozen histogram gradient boosting pipeline. Its
locked-test probabilities are read from the file notebook 06 already saved.
The embedding checkpoint is loaded and applied with the encoder that was fit
on the training period. Nothing in this module fits the embedding network on
the locked test, and nothing here changes the champion.

Logistic regression, the multilayer perceptron, and the most-frequent dummy
are scored on the same locked-test rows only when a fresh training-period fit
reproduces the saved validation macro-F1. A model that fails that check is
left off the ROC figure.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from src import config
from src.evaluation import classification_scores
from src.neural import EmbeddingNet, predict_embedding, predict_feedforward, train_feedforward
from src.optimisation import chronological_split_id
from src.preprocessing import (
    SOURCE_CATEGORICAL_COLUMNS,
    build_baseline_preprocessor,
    chronological_splits,
    logging_feature_frame,
    make_logistic_baseline,
    make_most_frequent_baseline,
)
from src.statistics import (
    assert_matches_split,
    mcnemar_test,
    paired_correctness_counts,
    paired_correctness_table,
)
from src.training import (
    APPROVED_APPROVAL,
    class_indices,
    probabilities_in_label_order,
    save_json,
    summarise_predictions,
)
from src.visualization import save_figure, use_project_style

QUESTION62_COMPARATOR = "embedding_network"
HISTORICAL_COMPARATOR = "multilayer_perceptron"
CHAMPION = "hist_gradient_boosting"
INTERPRETATION = (
    "The test compares paired error outcomes for the frozen champion and the "
    "embedding network on the same locked-test observations. It does not select "
    "the champion, and it does not by itself establish general superiority."
)
VALIDATION_TOLERANCE = 1e-4

SAVED_VALIDATION_MACRO_F1 = {
    "embedding_network": 0.4806872719871502,
    "logistic_regression": 0.4602680703697125,
    "multilayer_perceptron": 0.4764661870392401,
    "dummy_most_frequent": 0.2727882405417154,
}


def macro_ovr_curve(y_true, probabilities) -> dict:
    """Average the one-vs-rest ROC curves on a shared false-positive grid."""
    labels = list(config.TARGET_LABELS)
    actual = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError(
            "Probabilities must have one column per class. "
            f"Got shape {scores.shape}."
        )
    if len(actual) != len(scores):
        raise ValueError("Labels and probabilities have different lengths.")
    if not np.isfinite(scores).all():
        raise ValueError("Probabilities contain a non-finite value.")
    grid = np.linspace(0.0, 1.0, 201)
    class_rows = []
    interpolated = []
    for index, label in enumerate(labels):
        positive = actual == label
        support = int(positive.sum())
        if support == 0 or support == len(actual):
            class_rows.append(
                {
                    "label": label,
                    "support": support,
                    "roc_auc": np.nan,
                    "curve_defined": False,
                }
            )
            continue
        false_positive, true_positive, _thresholds = roc_curve(positive, scores[:, index])
        class_auc = float(auc(false_positive, true_positive))
        class_rows.append(
            {
                "label": label,
                "support": support,
                "roc_auc": class_auc,
                "curve_defined": True,
            }
        )
        curve = np.interp(grid, false_positive, true_positive)
        curve[0] = 0.0
        interpolated.append(curve)
    if not interpolated:
        raise ValueError("No class has a defined ROC curve.")
    mean_true_positive = np.mean(np.vstack(interpolated), axis=0)
    mean_true_positive[-1] = 1.0
    binary = label_binarize(actual, classes=labels)
    if binary.shape[1] == 1:
        raise ValueError("ROC needs more than one class in the evaluation rows.")
    return {
        "fpr": grid,
        "tpr": mean_true_positive,
        "macro_roc_auc": float(roc_auc_score(binary, scores, average="macro")),
        "micro_roc_auc": float(roc_auc_score(binary, scores, average="micro")),
        "classes": class_rows,
    }


def _load_splits():
    columns = list(SOURCE_CATEGORICAL_COLUMNS) + ["Created_On_Date", "resolution_outcome"]
    frame = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "service_requests_labelled.parquet",
        columns=columns,
    )
    splits, _summary, _exclusions = chronological_splits(frame)
    del frame
    return splits


def _load_embedding():
    model_path = config.MODELS_DIR / "embedding_network.pt"
    encoder_path = config.MODELS_DIR / "embedding_encoder.joblib"
    if not model_path.is_file() or not encoder_path.is_file():
        raise FileNotFoundError(
            "The approved embedding checkpoint or its training encoder is missing. "
            f"Expected {model_path} and {encoder_path}."
        )
    encoder = joblib.load(encoder_path)
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    cardinalities = [int(size) for size in payload["cardinalities"]]
    if [int(size) for size in encoder.cardinalities_] != cardinalities:
        raise RuntimeError(
            "The saved encoder cardinalities do not match the checkpoint. "
            "The locked test was not scored."
        )
    parameters = payload["parameters"]
    model = EmbeddingNet(
        cardinalities=cardinalities,
        embedding_dim=int(parameters["embedding_dim"]),
        hidden_size=int(parameters["hidden_size"]),
        dropout=float(parameters["dropout"]),
        n_numeric=1,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, encoder, int(parameters["batch_size"])


def _embedding_probabilities(model, encoder, features, batch_size: int):
    encoded = encoder.transform(features)
    _predicted, probabilities, _loss = predict_embedding(
        model,
        encoded["codes"],
        encoded["numeric"],
        None,
        batch_size,
    )
    return probabilities


def _require_validation_match(model_name: str, observed: float) -> None:
    expected = SAVED_VALIDATION_MACRO_F1[model_name]
    gap = abs(float(observed) - expected)
    if gap > VALIDATION_TOLERANCE:
        raise RuntimeError(
            f"{model_name} validation macro-F1 is {observed:.6f}. "
            f"The saved value is {expected:.6f}. The gap is {gap:.6f}, "
            "so its locked-test scores were not accepted."
        )


def score_embedding_on_locked_test(splits=None) -> dict:
    """Score the saved embedding checkpoint once. Do not fit it again."""
    own_splits = splits is None
    if own_splits:
        splits = _load_splits()
    model, encoder, batch_size = _load_embedding()
    validation_features = logging_feature_frame(splits["validation"])
    validation_labels = splits["validation"]["resolution_outcome"].to_numpy()
    validation_probability = _embedding_probabilities(
        model, encoder, validation_features, batch_size
    )
    validation_predicted = np.asarray(config.TARGET_LABELS)[np.argmax(validation_probability, axis=1)]
    validation_macro_f1 = classification_scores(validation_labels, validation_predicted)["macro_f1"]
    _require_validation_match(QUESTION62_COMPARATOR, validation_macro_f1)
    del validation_features, validation_probability

    test = splits["test"]
    test_features = logging_feature_frame(test)
    test_labels = test["resolution_outcome"].to_numpy()
    test_index = test.index.to_numpy()
    probabilities = _embedding_probabilities(model, encoder, test_features, batch_size)
    predicted = np.asarray(config.TARGET_LABELS, dtype=object)[np.argmax(probabilities, axis=1)]
    summary, per_class = summarise_predictions(
        test_labels,
        predicted,
        probabilities,
        model=QUESTION62_COMPARATOR,
        split="test",
        approval=APPROVED_APPROVAL,
        fit_seconds=float("nan"),
        locked_test_authorised=True,
    )
    champion = pd.read_parquet(config.TABLES_DIR / "06_test_predictions.parquet")
    embedding_frame = pd.DataFrame(
        {
            "source_index": test_index.astype(np.int64),
            "y_true": test_labels,
            "embedding_predicted": predicted,
            "p_resolved_fast": probabilities[:, 0],
            "p_resolved_slow": probabilities[:, 1],
            "p_unresolved": probabilities[:, 2],
        }
    )
    assert_matches_split(champion, test_index, test_labels)
    assert_matches_split(embedding_frame, champion["source_index"].to_numpy(), champion["y_true"].to_numpy())
    if not champion["y_true"].astype(str).reset_index(drop=True).equals(
        embedding_frame["y_true"].astype(str).reset_index(drop=True)
    ):
        raise RuntimeError("Champion and embedding ground-truth labels do not match.")
    counts = paired_correctness_counts(
        embedding_frame["y_true"],
        champion["champion_predicted"],
        embedding_frame["embedding_predicted"],
    )
    result = mcnemar_test(counts)
    frozen = json.loads((config.TABLES_DIR / "06_frozen_decision.json").read_text(encoding="utf-8"))
    record = {
        "question": "6.2",
        "champion": str(frozen.get("champion_family", CHAMPION)),
        "champion_selected": str(frozen.get("selected", "untuned")),
        "comparator": QUESTION62_COMPARATOR,
        "comparator_rule": (
            "Strongest approved model other than the champion by Phase 5 validation "
            "macro-F1. Ties break on Unresolved recall, then Resolved Slow recall, "
            "then model name. The embedding network is included. The locked test is "
            "not used to choose the champion or the comparator."
        ),
        "champion_validation_macro_f1": 0.4816599009477968,
        "comparator_validation_macro_f1": SAVED_VALIDATION_MACRO_F1[QUESTION62_COMPARATOR],
        "comparator_validation_macro_f1_recomputed": float(validation_macro_f1),
        "historical_comparator": HISTORICAL_COMPARATOR,
        "historical_result": "results/tables/07_mcnemar_test.csv",
        "split_id": chronological_split_id(),
        "test_rows": counts["rows"],
        "contingency": {
            "both_correct": counts["both_correct"],
            "champion_only": counts["champion_only"],
            "comparator_only": counts["comparator_only"],
            "both_wrong": counts["both_wrong"],
        },
        "discordant": result["discordant"],
        "statistic": result["statistic"],
        "p_value": result["p_value"],
        "alpha": result["alpha"],
        "method": result["method"],
        "significant": result["significant"],
        "conclusion": result["conclusion"],
        "interpretation": INTERPRETATION,
        "locked_test_macro_f1": float(summary["macro_f1"].iloc[0]),
        "locked_test_accuracy": float(summary["accuracy"].iloc[0]),
        "champion_locked_test_macro_f1": float(frozen["locked_test_macro_f1"]),
        "checkpoint": "models/embedding_network.pt",
        "encoder": "models/embedding_encoder.joblib",
        "encoder_fit_on": "training period",
        "retrained_on_test": False,
        "champion_changed": False,
    }
    prediction_path = config.TABLES_DIR / "13_embedding_locked_test_predictions.parquet"
    embedding_frame.to_parquet(prediction_path, index=False)
    summary.to_csv(config.TABLES_DIR / "13_embedding_locked_test_summary.csv", index=False)
    per_class.to_csv(config.TABLES_DIR / "13_embedding_locked_test_per_class.csv", index=False)
    paired_correctness_table(counts).to_csv(config.TABLES_DIR / "13_question62_counts.csv", index=False)
    pd.DataFrame([{**counts, **result, "comparator": QUESTION62_COMPARATOR}]).to_csv(
        config.TABLES_DIR / "13_question62_mcnemar.csv",
        index=False,
    )
    save_json(record, config.TABLES_DIR / "13_question62_record.json")
    if own_splits:
        del splits
    return record


def _baseline_matrices(splits):
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    preprocessor = build_baseline_preprocessor()
    x_train = logging_feature_frame(train)
    preprocessor.fit(x_train)
    return {
        "preprocessor": preprocessor,
        "y_train": train["resolution_outcome"].to_numpy(),
        "y_validation": validation["resolution_outcome"].to_numpy(),
        "train": preprocessor.transform(x_train),
        "validation": preprocessor.transform(logging_feature_frame(validation)),
        "test": preprocessor.transform(logging_feature_frame(test)),
    }


def _dummy_probabilities(prepared) -> np.ndarray:
    dummy = make_most_frequent_baseline()
    dummy.fit(prepared["train"], prepared["y_train"])
    validation_predicted = dummy.predict(prepared["validation"])
    _require_validation_match(
        "dummy_most_frequent",
        classification_scores(prepared["y_validation"], validation_predicted)["macro_f1"],
    )
    return probabilities_in_label_order(dummy.predict_proba(prepared["test"]), dummy.classes_)


def _logistic_probabilities(prepared) -> np.ndarray:
    logistic = make_logistic_baseline()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        logistic.fit(prepared["train"], prepared["y_train"])
    warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    if not warning:
        raise RuntimeError(
            "The logistic refit did not raise ConvergenceWarning. "
            "The reported baseline is the non-converged fit, so this run was not used."
        )
    validation_probability = probabilities_in_label_order(
        logistic.predict_proba(prepared["validation"]),
        logistic.classes_,
    )
    validation_labels = np.asarray(config.TARGET_LABELS)[np.argmax(validation_probability, axis=1)]
    _require_validation_match(
        "logistic_regression",
        classification_scores(prepared["y_validation"], validation_labels)["macro_f1"],
    )
    return probabilities_in_label_order(logistic.predict_proba(prepared["test"]), logistic.classes_)


def _perceptron_probabilities(splits) -> np.ndarray:
    """Refit the Phase 5 perceptron on the training period and score the locked test."""
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    x_train = logging_feature_frame(train)
    y_train = train["resolution_outcome"].to_numpy()
    x_validation = logging_feature_frame(validation)
    y_validation = validation["resolution_outcome"].to_numpy()
    x_test = logging_feature_frame(test)
    preprocessor = build_baseline_preprocessor()
    preprocessor.fit(x_train)
    trained = train_feedforward(
        preprocessor.transform(x_train),
        y_train,
        preprocessor.transform(x_validation),
        y_validation,
        verbose=False,
    )
    _require_validation_match(HISTORICAL_COMPARATOR, trained["best_validation_macro_f1"])
    batch_size = int(trained["parameters"]["batch_size"])
    _predicted, probabilities, _loss = predict_feedforward(
        trained["model"],
        preprocessor.transform(x_test),
        class_indices(test["resolution_outcome"]),
        batch_size,
    )
    return probabilities


def _plot_macro_curves(curves: list[dict], path_name: str) -> Path:
    import matplotlib.pyplot as plt

    use_project_style()
    figure, axis = plt.subplots(figsize=(8.4, 6.4), constrained_layout=True)
    colours = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
    axis.plot([0, 1], [0, 1], color="#666666", linewidth=1, linestyle="--", label="Chance")
    for index, row in enumerate(curves):
        axis.plot(
            row["fpr"],
            row["tpr"],
            color=colours[index % len(colours)],
            linewidth=2,
            label=f"{row['legend']} (macro AUC {row['macro_roc_auc']:.3f})",
        )
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("Locked test: macro one-vs-rest ROC")
    axis.legend(loc="lower right", fontsize=8)
    path = save_figure(figure, path_name, dpi=300)
    plt.close(figure)
    return path


def build_locked_test_roc(splits=None, extra_probabilities: dict[str, np.ndarray] | None = None) -> dict:
    """Draw one macro ROC curve per model that has locked-test probabilities."""
    champion = pd.read_parquet(config.TABLES_DIR / "06_test_predictions.parquet")
    embedding = pd.read_parquet(config.TABLES_DIR / "13_embedding_locked_test_predictions.parquet")
    assert_matches_split(embedding, champion["source_index"].to_numpy(), champion["y_true"].to_numpy())
    y_true = champion["y_true"].to_numpy()
    available = {
        "Histogram gradient boosting": np.column_stack(
            [
                champion["p_resolved_fast"].to_numpy(dtype=float),
                champion["p_resolved_slow"].to_numpy(dtype=float),
                champion["p_unresolved"].to_numpy(dtype=float),
            ]
        ),
        "Embedding network": np.column_stack(
            [
                embedding["p_resolved_fast"].to_numpy(dtype=float),
                embedding["p_resolved_slow"].to_numpy(dtype=float),
                embedding["p_unresolved"].to_numpy(dtype=float),
            ]
        ),
    }
    if extra_probabilities:
        names = {
            "dummy_most_frequent": "Most-frequent dummy",
            "logistic_regression": "Logistic regression",
            "multilayer_perceptron": "Multilayer perceptron",
        }
        for key, probabilities in extra_probabilities.items():
            if len(probabilities) != len(y_true):
                raise RuntimeError(f"{key} probabilities do not match the locked-test row count.")
            available[names[key]] = np.asarray(probabilities, dtype=float)
    plotted = []
    rows = []
    for name, probabilities in available.items():
        curve = macro_ovr_curve(y_true, probabilities)
        plotted.append(
            {
                "legend": name,
                "fpr": curve["fpr"],
                "tpr": curve["tpr"],
                "macro_roc_auc": curve["macro_roc_auc"],
            }
        )
        for class_row in curve["classes"]:
            rows.append(
                {
                    "model": name,
                    "label": class_row["label"],
                    "support": class_row["support"],
                    "roc_auc": class_row["roc_auc"],
                    "curve_defined": class_row["curve_defined"],
                    "macro_roc_auc": curve["macro_roc_auc"],
                    "micro_roc_auc": curve["micro_roc_auc"],
                    "evaluation_set": "locked test",
                }
            )
    figure_path = _plot_macro_curves(plotted, "13_roc_all_models_locked_test.png")
    table = pd.DataFrame(rows)
    table_path = config.TABLES_DIR / "13_roc_all_models_locked_test.csv"
    table.to_csv(table_path, index=False)
    payload = {
        "figure": str(figure_path.relative_to(config.PROJECT_ROOT)),
        "table": "results/tables/13_roc_all_models_locked_test.csv",
        "evaluation_set": "locked test",
        "split_id": chronological_split_id(),
        "models": [row["legend"] for row in plotted],
        "curve": "macro average of the three one-vs-rest ROC curves",
        "class_auc_in_table": True,
        "micro_auc_in_table": True,
        "champion_unchanged": True,
        "probabilities_fabricated": False,
    }
    save_json(payload, config.TABLES_DIR / "13_roc_all_models_config.json")
    return payload


def run() -> dict:
    """Score the embedding comparator, then the ROC models, on the locked test."""
    print("Scoring the saved embedding checkpoint.", flush=True)
    splits = _load_splits()
    record = score_embedding_on_locked_test(splits)
    print(json.dumps({key: record[key] for key in ("comparator", "test_rows", "statistic", "p_value")}, indent=2))
    print("Fitting the dummy and logistic baselines for the ROC figure.", flush=True)
    extra = {}
    skipped = {}
    prepared = _baseline_matrices(splits)
    for name, builder in (
        ("dummy_most_frequent", lambda: _dummy_probabilities(prepared)),
        ("logistic_regression", lambda: _logistic_probabilities(prepared)),
    ):
        try:
            extra[name] = builder()
        except RuntimeError as error:
            skipped[name] = str(error)
    del prepared
    print("Fitting the multilayer perceptron for the ROC figure.", flush=True)
    try:
        extra["multilayer_perceptron"] = _perceptron_probabilities(splits)
    except RuntimeError as error:
        skipped["multilayer_perceptron"] = str(error)
    roc_payload = build_locked_test_roc(splits, extra)
    roc_payload["skipped"] = skipped
    save_json(roc_payload, config.TABLES_DIR / "13_roc_all_models_config.json")
    print(json.dumps(roc_payload, indent=2))
    return {"question_6_2": record, "roc": roc_payload}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
