"""Check why saga logistic regression stops with a convergence warning.

The reported baseline remains ``make_logistic_baseline``: saga, C = 1, L2,
balanced class weights, max_iter 200, tol 0.001, and an intercept. This
script fits that preprocessor on the training period, records the scale and
the one-hot dependence, and then fits two separate variants at the same
iteration cap. It does not read the locked test, does not silence warnings,
and does not replace the baseline.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from scipy import sparse

from src import config
from src.evaluation import classification_scores
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    SOURCE_CATEGORICAL_COLUMNS,
    build_baseline_preprocessor,
    chronological_splits,
    logging_feature_frame,
    make_logistic_baseline,
)
from src.training import save_json

REPORTED_VALIDATION_MACRO_F1 = 0.4602680703697125


def _load_matrices():
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
    return preprocessor, x_train, x_validation, y_train, y_validation


def _matrix_diagnostics(preprocessor, x_train) -> dict:
    """Record scaling and the exact one-hot dependence. Do not fit a model."""
    indices = preprocessor.output_indices_
    categorical = indices["categorical"]
    numeric = indices["numeric"]
    categorical_values = np.asarray(x_train[:, categorical].data)
    one_hot = preprocessor.named_transformers_["categorical"].named_steps["one_hot"]
    block_start = int(categorical.start)
    block_gaps = []
    for name, categories in zip(CATEGORICAL_FEATURES, one_hot.categories_, strict=True):
        width = len(categories)
        row_sums = np.asarray(x_train[:, block_start : block_start + width].sum(axis=1)).ravel()
        block_gaps.append(
            {
                "feature": name,
                "levels": int(width),
                "max_abs_gap_from_one": float(np.max(np.abs(row_sums - 1.0))),
            }
        )
        block_start += width
    numeric_values = np.asarray(x_train[:, numeric].todense()).ravel()
    squared = np.asarray(x_train.power(2).sum(axis=0)).ravel()
    norms = np.sqrt(squared)
    return {
        "rows": int(x_train.shape[0]),
        "columns": int(x_train.shape[1]),
        "sparse": bool(sparse.issparse(x_train)),
        "one_hot_stored_values": [float(value) for value in np.unique(categorical_values)],
        "one_hot_min": float(categorical_values.min()),
        "one_hot_max": float(categorical_values.max()),
        "weekend_training_mean": float(numeric_values.mean()),
        "weekend_training_std": float(numeric_values.std()),
        "column_norm_min": float(norms.min()),
        "column_norm_max": float(norms.max()),
        "column_norm_ratio": float(norms.max() / max(norms.min(), 1e-12)),
        "categorical_blocks_sum_to_one": block_gaps,
        "intercept_plus_full_one_hot_is_dependent": all(
            item["max_abs_gap_from_one"] < 1e-8 for item in block_gaps
        ),
    }


def _fit_variant(x_train, y_train, x_validation, y_validation, name: str, model) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    iterations = int(np.max(model.n_iter_))
    predicted = model.predict(x_validation)
    scores = classification_scores(y_validation, predicted)
    return {
        "name": name,
        "fit_intercept": bool(model.fit_intercept),
        "max_iter": int(model.max_iter),
        "tol": float(model.tol),
        "C": float(model.C),
        "solver": str(model.solver),
        "iterations_used": iterations,
        "convergence_warning": bool(warning),
        "converged": (not warning) and iterations < int(model.max_iter),
        "validation_macro_f1": float(scores["macro_f1"]),
        "validation_accuracy": float(scores["accuracy"]),
        "adopted_as_the_reported_baseline": False,
        "locked_test_scored": False,
    }


def _scale_one_hot(preprocessor, x_train, x_validation):
    """Variance-scale the one-hot block only. The weekend column is already standardised."""
    categorical = preprocessor.output_indices_["categorical"]
    numeric = preprocessor.output_indices_["numeric"]
    scaler = StandardScaler(with_mean=False)
    train_categorical = scaler.fit_transform(x_train[:, categorical])
    validation_categorical = scaler.transform(x_validation[:, categorical])
    train = sparse.hstack(
        [train_categorical, x_train[:, numeric]], format="csr", dtype=np.float64
    )
    validation = sparse.hstack(
        [validation_categorical, x_validation[:, numeric]], format="csr", dtype=np.float64
    )
    return train, validation


def run() -> dict:
    preprocessor, x_train, x_validation, y_train, y_validation = _load_matrices()
    diagnostics = _matrix_diagnostics(preprocessor, x_train)
    no_intercept = make_logistic_baseline()
    no_intercept.set_params(fit_intercept=False)
    scaled_train, scaled_validation = _scale_one_hot(preprocessor, x_train, x_validation)
    scaled_model = make_logistic_baseline()
    variants = [
        _fit_variant(
            x_train,
            y_train,
            x_validation,
            y_validation,
            "saga_no_intercept_max_iter_200",
            no_intercept,
        ),
        _fit_variant(
            scaled_train,
            y_train,
            scaled_validation,
            y_validation,
            "saga_variance_scaled_one_hot_max_iter_200",
            scaled_model,
        ),
    ]
    payload = {
        "reported_baseline": {
            "function": "src.preprocessing.make_logistic_baseline",
            "solver": "saga",
            "C": 1.0,
            "class_weight": "balanced",
            "fit_intercept": True,
            "max_iter": 200,
            "tol": 0.001,
            "validation_macro_f1": REPORTED_VALIDATION_MACRO_F1,
            "convergence_warning": True,
            "replaced": False,
        },
        "diagnostics": diagnostics,
        "variants": variants,
        "cause": (
            "The weekend indicator is already standardised on the training rows, and the "
            "one-hot values are 0/1. Each categorical block sums to 1, so an intercept "
            "makes the design dependent. Column norms still differ because frequent levels "
            "are much more common than rare levels. Saga is the sparse multinomial solver; "
            "a dense solver would materialise the one-hot matrix. L2 regularisation is "
            "present at C = 1. The recorded fit and the earlier max_iter 1000 fit both "
            "stopped on the iteration cap with a convergence warning."
        ),
        "locked_test_scored": False,
        "champion_changed": False,
        "warnings_suppressed": False,
    }
    save_json(payload, config.TABLES_DIR / "12_logistic_conditioning.json")
    pd.DataFrame(variants).to_csv(config.TABLES_DIR / "12_logistic_variants.csv", index=False)
    print(payload["cause"])
    print(pd.DataFrame(variants).to_string(index=False))
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
