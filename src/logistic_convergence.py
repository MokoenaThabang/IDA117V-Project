"""Refit saga logistic regression with a higher iteration cap.

The Phase 4 fit used max_iter 200 and stopped with a convergence warning
after exactly 200 iterations. The one-hot columns are 0/1 and the weekend
indicator is standardised on the training rows, so the warning is the
iteration cap rather than an unscaled numeric feature. This script does not
read the locked test and does not replace the frozen champion.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src import config
from src.evaluation import classification_scores, ranking_scores
from src.preprocessing import (
    SOURCE_CATEGORICAL_COLUMNS,
    build_baseline_preprocessor,
    chronological_splits,
    logging_feature_frame,
    make_logistic_baseline,
)
from src.training import probabilities_in_label_order, save_json

MAX_ITER = 1000
PHASE5_LEADER = "hist_gradient_boosting"


def run() -> dict:
    columns = list(SOURCE_CATEGORICAL_COLUMNS) + ["Created_On_Date", "resolution_outcome"]
    frame = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "service_requests_labelled.parquet",
        columns=columns,
    )
    splits, _summary, _exclusions = chronological_splits(frame)
    del frame
    del splits["test"]
    train = logging_feature_frame(splits["train"])
    validation = logging_feature_frame(splits["validation"])
    y_train = splits["train"]["resolution_outcome"].to_numpy()
    y_validation = splits["validation"]["resolution_outcome"].to_numpy()
    del splits

    preprocessor = build_baseline_preprocessor()
    x_train = preprocessor.fit_transform(train)
    x_validation = preprocessor.transform(validation)
    del train
    model = make_logistic_baseline()
    model.set_params(max_iter=MAX_ITER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    iterations = int(np.max(model.n_iter_))
    rows = []
    per_class_rows = []
    for split_name, matrix, labels in (
        ("train", x_train, y_train),
        ("validation", x_validation, y_validation),
    ):
        predicted = model.predict(matrix)
        probabilities = probabilities_in_label_order(model.predict_proba(matrix), model.classes_)
        scores = classification_scores(labels, predicted)
        ranking = ranking_scores(labels, probabilities)
        row = {
            "model": "logistic_regression",
            "split": split_name,
            "accuracy": scores["accuracy"],
            "macro_f1": scores["macro_f1"],
            "weighted_f1": scores["weighted_f1"],
            "macro_roc_auc": ranking["macro_roc_auc"],
            "macro_average_precision": ranking["macro_average_precision"],
        }
        for item in scores["per_class"]:
            key = item["label"].lower().replace(" ", "_")
            row[f"precision_{key}"] = item["precision"]
            row[f"recall_{key}"] = item["recall"]
            row[f"f1_{key}"] = item["f1"]
            row[f"support_{key}"] = item["support"]
            per_class_rows.append({"model": "logistic_regression", "split": split_name, **item})
        rows.append(row)
    metrics = pd.DataFrame(rows)
    phase5 = pd.read_csv(config.TABLES_DIR / "05_split_metrics.csv")
    leader = phase5.loc[
        phase5["model"].eq(PHASE5_LEADER) & phase5["split"].eq("validation")
    ].iloc[0]
    validation_f1 = float(metrics.loc[metrics["split"].eq("validation"), "macro_f1"].iloc[0])
    leader_f1 = float(leader["macro_f1"])
    payload = {
        "solver": "saga",
        "C": 1.0,
        "penalty": "l2",
        "class_weight": "balanced",
        "tol": 0.001,
        "max_iter": MAX_ITER,
        "iterations_used": iterations,
        "convergence_warning": bool(convergence_warning),
        "converged": (not convergence_warning) and iterations < MAX_ITER,
        "likely_cause_of_the_original_warning": (
            "The recorded fit used max_iter 200 and stopped at iteration 200 with a "
            "ConvergenceWarning. Categorical inputs are one-hot 0/1 values and the "
            "weekend indicator is standardised on the training rows. The warning is "
            "the iteration cap on a 2.6 million row sparse problem, not a missing scaler."
        ),
        "historical_max_iter": 200,
        "historical_iterations_used": 200,
        "historical_convergence_warning": True,
        "locked_test_scored": False,
        "champion_rule": "Highest validation macro-F1 among approved models.",
        "phase5_hist_gradient_boosting_validation_macro_f1": leader_f1,
        "this_run_validation_macro_f1": validation_f1,
        "recorded_200_iteration_validation_macro_f1": 0.4602680703697125,
        "champion_changed": bool(validation_f1 > leader_f1),
        "delivered_pipeline_replaced": False,
        "adopted_as_the_reported_baseline": False,
        "adoption_reason": (
            "1000 iterations still raised ConvergenceWarning. Validation macro-F1 was "
            f"{validation_f1:.6f}, below the recorded 200-iteration validation macro-F1 "
            "of 0.460268. The Phase 4 configuration is unchanged."
        ),
    }
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.TABLES_DIR / "11_logistic_convergence.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(config.TABLES_DIR / "11_logistic_per_class.csv", index=False)
    save_json(payload, config.TABLES_DIR / "11_logistic_convergence.json")
    print(metrics.round(4).to_string(index=False))
    print(payload)
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
