"""Training utilities for the traditional and neural experiments.

Every learned transform is fit on the training period only. The locked test
split can be counted, and this module refuses to score it. Class imbalance
is handled with balanced class weights. These helpers do not resample any
split and they do not apply SMOTE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, TARGET_LABELS
from src.evaluation import classification_scores, metrics_tables, ranking_scores
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    MATURE_END,
    MISSING_CATEGORY_LABEL,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RARE_CATEGORY_LABEL,
    RARE_CATEGORY_MIN_COUNT,
    TRAIN_END,
    VALIDATION_END,
    _reject_blocked_features,
    make_logistic_baseline,
)

# Histogram gradient boosting keeps at most 255 bins. One bin is reserved
# for the grouped rare level and one for the missing level.
TREE_MAX_BINS = 255
TREE_MAX_KEPT_CATEGORIES = TREE_MAX_BINS - 2
APPROVED_APPROVAL = "approved"
PENDING_APPROVAL = "pending_lecturer_approval"
LOCKED_TEST_SPLIT = "test"

HIST_GRADIENT_BOOSTING_PARAMETERS = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "max_iter": 40,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "max_bins": TREE_MAX_BINS,
    "early_stopping": True,
    "scoring": "f1_macro",
    "n_iter_no_change": 6,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

MLP_PARAMETERS = {
    "hidden_sizes": (128, 64),
    "activation": "ReLU",
    "dropout": 0.3,
    "optimiser": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 8192,
    "max_epochs": 8,
    "patience": 3,
    "loss": "CrossEntropyLoss",
    "class_weight": "balanced",
    "seed": RANDOM_SEED,
}

EMBEDDING_PARAMETERS = {
    "approval": APPROVED_APPROVAL,
    "embedding_dim": 16,
    "hidden_size": 64,
    "activation": "ReLU",
    "dropout": 0.3,
    "optimiser": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 8192,
    "max_epochs": 8,
    "patience": 3,
    "loss": "CrossEntropyLoss",
    "class_weight": "balanced",
    "unseen_index": 0,
    "seed": RANDOM_SEED,
}


def assert_split_may_be_scored(split: str) -> None:
    """Phase 5 may score train and validation. The locked test stays locked."""
    if split == LOCKED_TEST_SPLIT:
        raise RuntimeError(
            "The locked test split is not scored in Phase 5. "
            "The provisional champion uses validation macro-F1 only."
        )


def balanced_class_weights(
    y,
    labels: tuple[str, ...] = TARGET_LABELS,
) -> np.ndarray:
    """Return n / (n_classes * n_class) in ``labels`` order.

    This is the same balanced weighting used by the logistic baseline.
    It is computed from the labels it is given, which must be the training
    labels when it is used to train a model.
    """
    observed = pd.Series(y)
    n_rows = int(len(observed))
    if n_rows == 0:
        raise ValueError("Class weights need at least one training label.")
    weights = []
    for label in labels:
        count = int(observed.eq(label).sum())
        if count == 0:
            raise ValueError(
                f"Cannot weight {label!r} because it has no training rows."
            )
        weights.append(n_rows / (len(labels) * count))
    return np.asarray(weights, dtype=np.float64)


def class_indices(y, labels: tuple[str, ...] = TARGET_LABELS) -> np.ndarray:
    """Map class names to 0, 1, 2 in ``TARGET_LABELS`` order."""
    lookup = {label: index for index, label in enumerate(labels)}
    observed = pd.Series(y, copy=False)
    unknown = sorted(set(observed.dropna().unique()) - set(lookup))
    if unknown or bool(observed.isna().any()):
        raise ValueError(
            "Labels must be exactly the three resolution classes. "
            f"Unexpected values: {unknown or ['<missing>']}."
        )
    return observed.map(lookup).to_numpy(dtype=np.int64)


def _clean_category(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text.eq(""), pd.NA)


def _assert_model_frame(frame: pd.DataFrame) -> None:
    _assert_predictor_frame(frame, list(MODEL_FEATURES))


def _assert_predictor_frame(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    allow_work_centre: bool = False,
) -> None:
    """Reject leakage columns and require the predictors this fit will use."""
    _reject_blocked_features(list(frame.columns), allow_work_centre=allow_work_centre)
    _reject_blocked_features(list(columns), allow_work_centre=allow_work_centre)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "The modelling frame is missing predictors: "
            + ", ".join(missing)
            + ". Actual columns: "
            + ", ".join(map(str, frame.columns))
            + "."
        )


def training_category_levels(
    series: pd.Series,
    min_count: int,
    max_kept: int | None,
) -> dict:
    """Choose training levels for one categorical column.

    Levels at or above ``min_count`` are eligible. When ``max_kept`` is set,
    only the most frequent eligible levels are kept. Frequency ties break on
    the category text so the result does not depend on row order.
    """
    if int(min_count) < 1:
        raise ValueError(f"min_count must be at least 1. Got {min_count}.")
    if max_kept is not None and int(max_kept) < 1:
        raise ValueError(f"max_kept must be at least 1. Got {max_kept}.")
    text = _clean_category(series)
    counts = text.dropna().value_counts()
    observed = [str(name) for name in counts.index]
    eligible = [
        (name, int(counts[name]))
        for name in observed
        if int(counts[name]) >= int(min_count)
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    if max_kept is not None:
        eligible = eligible[: int(max_kept)]
    kept = [name for name, _count in eligible]
    kept_set = set(kept)
    grouped = [name for name in observed if name not in kept_set]
    return {
        "kept": kept,
        "observed": observed,
        "grouped_levels": len(grouped),
        "distinct_levels": int(len(counts)),
        "missing_rows": int(text.isna().sum()),
        "training_rows": int(text.notna().sum()),
    }


class TreeCategoryEncoder:
    """Integer codes for histogram gradient boosting, fit on training rows.

    HistGradientBoosting rejects a categorical feature with more than 255
    levels. This encoder keeps at most 253 frequent training levels, plus
    ``(missing)`` and ``(rare)`` when those levels occur in training. A
    category that was too rare, or that was never seen in training, uses the
    trained rare code when that code exists. Otherwise it becomes missing,
    which the booster treats as a missing categorical value. The cap is the
    library limit, not a cutoff selected from the validation score.
    """

    def __init__(
        self,
        min_count: int = RARE_CATEGORY_MIN_COUNT,
        max_kept: int = TREE_MAX_KEPT_CATEGORIES,
        categorical_columns: tuple[str, ...] | list[str] | None = None,
        numeric_columns: tuple[str, ...] | list[str] | None = None,
        allow_work_centre: bool = False,
    ):
        self.min_count = min_count
        self.max_kept = max_kept
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.allow_work_centre = allow_work_centre

    def fit(self, frame: pd.DataFrame):
        categorical = list(
            CATEGORICAL_FEATURES if self.categorical_columns is None else self.categorical_columns
        )
        numeric = list(NUMERIC_FEATURES if self.numeric_columns is None else self.numeric_columns)
        if not categorical and not numeric:
            raise ValueError("The tree encoder needs at least one predictor.")
        if "Work_Center" in categorical and not getattr(self, "allow_work_centre", False):
            raise ValueError(
                "Work Center is not a champion predictor. "
                "The sensitivity comparison must set allow_work_centre=True."
            )
        _assert_predictor_frame(
            frame,
            categorical + numeric,
            allow_work_centre=getattr(self, "allow_work_centre", False),
        )
        self.categorical_columns_ = categorical
        self.numeric_columns_ = numeric
        self.code_maps_ = {}
        self.unseen_codes_ = {}
        self.summary_ = []
        self.numeric_fill_ = {}
        for column in categorical:
            levels = training_category_levels(
                frame[column],
                min_count=self.min_count,
                max_kept=self.max_kept,
            )
            ordered = []
            if levels["missing_rows"] > 0:
                ordered.append(MISSING_CATEGORY_LABEL)
            if levels["grouped_levels"] > 0:
                ordered.append(RARE_CATEGORY_LABEL)
            ordered.extend(levels["kept"])
            if len(ordered) > TREE_MAX_BINS:
                raise RuntimeError(
                    f"{column} still has {len(ordered)} encoded levels after the cap. "
                    f"The booster limit is {TREE_MAX_BINS}."
                )
            mapping = {name: float(index) for index, name in enumerate(ordered)}
            self.code_maps_[column] = mapping
            self.unseen_codes_[column] = mapping.get(RARE_CATEGORY_LABEL, np.nan)
            self.summary_.append(
                {
                    "column": column,
                    "distinct_training_levels": levels["distinct_levels"],
                    "kept_levels": len(levels["kept"]),
                    "grouped_levels": levels["grouped_levels"],
                    "missing_training_rows": levels["missing_rows"],
                    "encoded_levels": len(ordered),
                    "min_count": int(self.min_count),
                    "max_kept": int(self.max_kept),
                }
            )
        for column in numeric:
            observed = pd.to_numeric(frame[column], errors="coerce").dropna()
            self.numeric_fill_[column] = float(observed.median()) if len(observed) else 0.0
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "code_maps_"):
            raise AttributeError("TreeCategoryEncoder must be fit before transform.")
        _assert_predictor_frame(
            frame,
            list(self.categorical_columns_) + list(self.numeric_columns_),
            allow_work_centre=getattr(self, "allow_work_centre", False),
        )
        coded = {}
        for column in self.categorical_columns_:
            coded[column] = self._codes(frame[column], column)
        for column in self.numeric_columns_:
            values = pd.to_numeric(frame[column], errors="coerce")
            coded[column] = values.fillna(self.numeric_fill_[column]).to_numpy(dtype=np.float64)
        return pd.DataFrame(coded, index=frame.index)

    def level_summary(self) -> pd.DataFrame:
        if not hasattr(self, "summary_"):
            raise AttributeError("TreeCategoryEncoder must be fit before level_summary.")
        return pd.DataFrame(self.summary_)

    def _codes(self, series: pd.Series, column: str) -> np.ndarray:
        text = _clean_category(series)
        mapping = self.code_maps_[column]
        unseen = self.unseen_codes_[column]
        codes = np.full(len(text), np.nan, dtype=np.float64)
        missing = text.isna().to_numpy()
        if MISSING_CATEGORY_LABEL in mapping:
            codes[missing] = mapping[MISSING_CATEGORY_LABEL]
        present = np.flatnonzero(~missing)
        if len(present) == 0:
            return codes
        known_codes = np.array(text.iloc[present].map(mapping), dtype=np.float64, copy=True)
        unmatched = np.isnan(known_codes)
        known_codes[unmatched] = unseen
        codes[present] = known_codes
        return codes


class EmbeddingCategoryEncoder:
    """Training-only integer codes for a categorical embedding network.

    Index 0 is reserved for a category that was not in the training
    vocabulary. The embedding layer uses that index as padding, so its
    vector stays at zero. Missing values and rare training levels each
    receive their own positive index when they occur in training. This
    encoder does not cap cardinality at 255, because an embedding table
    does not have the histogram-boosting bin limit.
    """

    def __init__(self, min_count: int = RARE_CATEGORY_MIN_COUNT):
        self.min_count = min_count

    def fit(self, frame: pd.DataFrame):
        _assert_model_frame(frame)
        self.code_maps_ = {}
        self.training_levels_ = {}
        self.cardinalities_ = []
        self.summary_ = []
        for column in CATEGORICAL_FEATURES:
            levels = training_category_levels(
                frame[column],
                min_count=self.min_count,
                max_kept=None,
            )
            self.training_levels_[column] = set(levels["observed"])
            mapping = {}
            next_index = 1
            if levels["missing_rows"] > 0:
                mapping[MISSING_CATEGORY_LABEL] = next_index
                next_index += 1
            if levels["grouped_levels"] > 0:
                mapping[RARE_CATEGORY_LABEL] = next_index
                next_index += 1
            for name in levels["kept"]:
                mapping[name] = next_index
                next_index += 1
            self.code_maps_[column] = mapping
            self.cardinalities_.append(next_index)
            self.summary_.append(
                {
                    "column": column,
                    "distinct_training_levels": levels["distinct_levels"],
                    "kept_levels": len(levels["kept"]),
                    "grouped_levels": levels["grouped_levels"],
                    "missing_training_rows": levels["missing_rows"],
                    "embedding_rows": next_index,
                    "unseen_index": 0,
                    "min_count": int(self.min_count),
                }
            )
        weekend = pd.to_numeric(frame[NUMERIC_FEATURES[0]], errors="coerce")
        observed = weekend.dropna()
        self.numeric_fill_ = float(observed.median()) if len(observed) else 0.0
        filled = weekend.fillna(self.numeric_fill_).to_numpy(dtype=np.float64).reshape(-1, 1)
        self.numeric_mean_ = float(filled.mean())
        self.numeric_scale_ = float(filled.std())
        if self.numeric_scale_ == 0.0 or np.isnan(self.numeric_scale_):
            self.numeric_scale_ = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        if not hasattr(self, "code_maps_"):
            raise AttributeError("EmbeddingCategoryEncoder must be fit before transform.")
        _assert_model_frame(frame)
        codes = np.column_stack(
            [self._codes(frame[column], column) for column in CATEGORICAL_FEATURES]
        ).astype(np.int64, copy=False)
        weekend = pd.to_numeric(frame[NUMERIC_FEATURES[0]], errors="coerce")
        filled = weekend.fillna(self.numeric_fill_).to_numpy(dtype=np.float64)
        numeric = ((filled - self.numeric_mean_) / self.numeric_scale_).astype(np.float32)
        return {"codes": codes, "numeric": numeric.reshape(-1, 1)}

    def level_summary(self) -> pd.DataFrame:
        if not hasattr(self, "summary_"):
            raise AttributeError("EmbeddingCategoryEncoder must be fit before level_summary.")
        return pd.DataFrame(self.summary_)

    def _codes(self, series: pd.Series, column: str) -> np.ndarray:
        text = _clean_category(series)
        mapping = self.code_maps_[column]
        codes = np.zeros(len(text), dtype=np.int64)
        missing = text.isna().to_numpy()
        if MISSING_CATEGORY_LABEL in mapping:
            codes[missing] = mapping[MISSING_CATEGORY_LABEL]
        present = np.flatnonzero(~missing)
        if len(present) == 0:
            return codes
        values = text.iloc[present]
        known_codes = values.map(mapping).to_numpy(dtype=np.float64)
        resolved = np.zeros(len(present), dtype=np.int64)
        matched = ~pd.isna(known_codes)
        resolved[matched] = known_codes[matched].astype(np.int64)
        unmatched = ~matched
        rare_code = mapping.get(RARE_CATEGORY_LABEL)
        if unmatched.any() and rare_code is not None:
            grouped = values.isin(self.training_levels_[column]).to_numpy()
            resolved[unmatched & grouped] = rare_code
        codes[present] = resolved
        return codes


def describe_sparse_matrix(matrix, name: str) -> dict:
    """Describe a sparse predictor matrix, including the dense memory it would need."""
    rows, columns = matrix.shape
    nonzero = int(matrix.nnz)
    dense_gib = (rows * columns * 4) / (1024**3)
    return {
        "name": name,
        "rows": int(rows),
        "columns": int(columns),
        "nonzero": nonzero,
        "density": float(nonzero / (rows * columns)) if rows and columns else 0.0,
        "dense_float32_gib": float(dense_gib),
    }


def make_hist_gradient_boosting(**overrides):
    """Balanced histogram gradient boosting with validation early stopping.

    The learning rate, tree size, and iteration cap are pre-specified
    settings. They were not chosen by searching validation scores, and the
    locked test is not an input. Pass the chronological validation rows to
    ``fit`` as ``X_val`` and ``y_val``. A random ``validation_fraction``
    split is not used.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    parameters = dict(HIST_GRADIENT_BOOSTING_PARAMETERS)
    parameters.update(overrides)
    parameters["categorical_features"] = list(CATEGORICAL_FEATURES)
    parameters["random_state"] = int(parameters.get("random_state", RANDOM_SEED))
    if parameters.get("early_stopping") is not True:
        raise ValueError("Histogram gradient boosting in this project uses early stopping.")
    return HistGradientBoostingClassifier(**parameters)


def use_best_validation_iteration(model) -> int:
    """Keep the boosting iteration with the best validation score.

    The library stops after several rounds without improvement and still
    includes those extra trees. Predictions for this experiment use the
    prefix whose validation macro-F1 was highest. The recorded score curve
    is left intact.
    """
    scores = np.asarray(getattr(model, "validation_score_", []), dtype=float)
    if scores.size == 0:
        raise ValueError(
            "Boosting has no validation scores. Fit it with early_stopping and X_val."
        )
    best = int(np.argmax(scores))
    trained = len(model._predictors)
    model.n_iter_trained_ = trained
    model.best_iteration_ = best
    if best >= 1 and best < trained:
        model._predictors = model._predictors[:best]
    return best


def logistic_config() -> dict:
    """Parameters of the Phase 4 logistic baseline, read from the estimator."""
    model = make_logistic_baseline()
    return {
        "C": float(model.C),
        "solver": model.solver,
        "class_weight": model.class_weight,
        "max_iter": int(model.max_iter),
        "tol": float(model.tol),
        "l1_ratio": float(model.l1_ratio),
        "random_state": int(model.random_state),
    }


def base_experiment_config() -> dict:
    """Pre-specified experiment settings. Measured scores are added after fitting."""
    return {
        "seed": RANDOM_SEED,
        "split": {
            "method": "chronological",
            "train_end": str(pd.Timestamp(TRAIN_END).date()),
            "validation_end": str(pd.Timestamp(VALIDATION_END).date()),
            "mature_end": str(pd.Timestamp(MATURE_END).date()),
            "shuffled": False,
        },
        "imbalance": {
            "method": "class_weight_balanced",
            "smote": False,
            "resample_validation": False,
            "resample_locked_test": False,
        },
        "one_hot": {
            "rare_min_count": RARE_CATEGORY_MIN_COUNT,
            "missing_label": MISSING_CATEGORY_LABEL,
            "rare_label": RARE_CATEGORY_LABEL,
            "unknown_categories": "ignore",
            "fit_on": "train",
        },
        "tree_categories": {
            "rare_min_count": RARE_CATEGORY_MIN_COUNT,
            "max_kept_levels": TREE_MAX_KEPT_CATEGORIES,
            "max_bins": TREE_MAX_BINS,
            "unseen": "trained rare level, or missing when no rare level was learned",
            "fit_on": "train",
        },
        "embedding_categories": {
            "rare_min_count": RARE_CATEGORY_MIN_COUNT,
            "unseen_index": 0,
            "unseen_vector": "zero, not updated",
            "fit_on": "train",
            "approval": APPROVED_APPROVAL,
        },
        "logistic_regression": logistic_config(),
        "hist_gradient_boosting": {
            key: value
            for key, value in HIST_GRADIENT_BOOSTING_PARAMETERS.items()
        },
        "multilayer_perceptron": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in MLP_PARAMETERS.items()
        },
        "embedding_network": {
            key: value for key, value in EMBEDDING_PARAMETERS.items()
        },
        "locked_test_scored_in_phase_5": False,
        "champion_rule": (
            "Highest validation macro-F1 among approved models. "
            "Ties break on Unresolved recall, then Resolved Slow recall. "
            "The embedding network is included because lecturer approval was recorded. "
            "A model that is still marked pending is excluded. "
            "The locked test is not an input."
        ),
    }


def save_json(payload: dict, path: Path) -> Path:
    """Write a JSON config or result file with stable key order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def probabilities_in_label_order(probabilities, class_names, labels=TARGET_LABELS) -> np.ndarray:
    """Reorder probability columns to ``TARGET_LABELS``."""
    names = [str(name) for name in class_names]
    missing = [label for label in labels if label not in names]
    if missing:
        raise ValueError(
            "Probability columns are missing classes: " + ", ".join(missing) + "."
        )
    order = [names.index(label) for label in labels]
    aligned = np.asarray(probabilities, dtype=float)[:, order]
    if aligned.shape[0] == 0:
        return aligned
    return aligned


def summarise_predictions(
    y_true,
    y_pred,
    y_proba,
    model: str,
    split: str,
    approval: str,
    fit_seconds: float | None = None,
    locked_test_authorised: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one split. The locked test is refused unless a frozen model authorises it."""
    if not (split == LOCKED_TEST_SPLIT and locked_test_authorised):
        assert_split_may_be_scored(split)
    if approval not in {APPROVED_APPROVAL, PENDING_APPROVAL}:
        raise ValueError(
            f"approval must be {APPROVED_APPROVAL!r} or {PENDING_APPROVAL!r}. Got {approval!r}."
        )
    scores = classification_scores(y_true, y_pred)
    summary, per_class = metrics_tables(scores, model=model, split=split)
    ranking = ranking_scores(y_true, y_proba)
    summary = summary.copy()
    summary["macro_roc_auc"] = ranking["macro_roc_auc"]
    summary["macro_average_precision"] = ranking["macro_average_precision"]
    summary["approval"] = approval
    summary["fit_seconds"] = np.nan if fit_seconds is None else float(fit_seconds)
    return summary, per_class


def provisional_champion(metrics: pd.DataFrame) -> pd.Series:
    """Choose the approved model with the highest validation macro-F1.

    Pending-approval models are ignored even when their macro-F1 is higher.
    The locked test is not a column this function reads. Equal macro-F1
    breaks on Unresolved recall, then Resolved Slow recall, then the model
    name.
    """
    required = {"model", "split", "approval", "macro_f1", "recall_unresolved", "recall_resolved_slow"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError("Champion metrics are missing columns: " + ", ".join(sorted(missing)) + ".")
    rows = metrics.loc[
        metrics["split"].eq("validation") & metrics["approval"].eq(APPROVED_APPROVAL)
    ].copy()
    if rows.empty:
        raise ValueError("No approved validation metrics are available for a champion.")
    ranked = rows.sort_values(
        ["macro_f1", "recall_unresolved", "recall_resolved_slow", "model"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def validation_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Validation rows used in the model-comparison table, champion row first."""
    rows = metrics.loc[metrics["split"].eq("validation")].copy()
    if rows.empty:
        raise ValueError("The metrics table has no validation rows.")
    rows["_approved"] = rows["approval"].eq(APPROVED_APPROVAL).astype(int)
    rows = rows.sort_values(
        ["_approved", "macro_f1"],
        ascending=[False, False],
        kind="mergesort",
    ).drop(columns="_approved")
    columns = [
        "model",
        "approval",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "recall_resolved_fast",
        "recall_resolved_slow",
        "recall_unresolved",
        "macro_roc_auc",
        "macro_average_precision",
        "fit_seconds",
    ]
    return rows.loc[:, columns].reset_index(drop=True)


def _metric_value(metrics: pd.DataFrame, model: str, split: str, column: str) -> float:
    rows = metrics.loc[metrics["model"].eq(model) & metrics["split"].eq(split)]
    if rows.empty:
        raise ValueError(f"No {split} metrics for {model}.")
    return float(rows.iloc[0][column])


def render_phase5_report(
    split_summary: pd.DataFrame,
    one_hot: dict,
    tree_levels: pd.DataFrame,
    metrics: pd.DataFrame,
    run_notes: dict,
) -> str:
    """Write the Phase 5 report from measured tables. It does not invent scores."""
    champion = provisional_champion(metrics)
    comparison = validation_comparison(metrics)

    def split_line(name: str) -> str:
        row = split_summary.loc[split_summary["split"].eq(name)].iloc[0]
        return (
            f"{row['start_date']} to {row['end_date']}, {int(row['rows']):,} labelled requests "
            f"({row['Resolved Fast percent']:.1f}% Resolved Fast, "
            f"{row['Resolved Slow percent']:.1f}% Resolved Slow, "
            f"{row['Unresolved percent']:.1f}% Unresolved)"
        )

    def score_line(model: str, split: str) -> str:
        accuracy = _metric_value(metrics, model, split, "accuracy")
        macro = _metric_value(metrics, model, split, "macro_f1")
        weighted = _metric_value(metrics, model, split, "weighted_f1")
        slow = _metric_value(metrics, model, split, "recall_resolved_slow")
        unresolved = _metric_value(metrics, model, split, "recall_unresolved")
        return (
            f"accuracy {accuracy:.3f}, macro-F1 {macro:.3f}, weighted-F1 {weighted:.3f}, "
            f"Resolved Slow recall {slow:.3f}, Unresolved recall {unresolved:.3f}"
        )

    tree_lines = []
    for row in tree_levels.itertuples(index=False):
        tree_lines.append(
            f"- {row.column}: {int(row.kept_levels):,} of {int(row.distinct_training_levels):,} "
            f"training levels kept, {int(row.encoded_levels):,} encoded levels."
        )

    approved = comparison.loc[comparison["approval"].eq(APPROVED_APPROVAL)]
    recall_notes = []
    for column, label in (
        ("recall_resolved_slow", "Resolved Slow"),
        ("recall_unresolved", "Unresolved"),
    ):
        leader = approved.sort_values(column, ascending=False, kind="mergesort").iloc[0]
        recall_notes.append(
            f"The highest approved validation recall for {label} is "
            f"{float(leader[column]):.3f} ({leader['model']}). "
            f"The macro-F1 leader's {label} recall is "
            f"{float(champion[column]):.3f}."
        )

    pending = comparison.loc[comparison["approval"].eq(PENDING_APPROVAL)]
    if pending.empty:
        pending_text = "The pending embedding network was not scored."
    else:
        pending_row = pending.iloc[0]
        pending_text = (
            f"The categorical embedding network has validation macro-F1 "
            f"{float(pending_row['macro_f1']):.3f}, Resolved Slow recall "
            f"{float(pending_row['recall_resolved_slow']):.3f}, and Unresolved recall "
            f"{float(pending_row['recall_unresolved']):.3f}. "
            "Those figures describe an architecture that is still pending lecturer approval. "
            "They are not an approved experiment and they do not choose the provisional champion."
        )

    logistic = run_notes["logistic_regression"]
    boosting = run_notes["hist_gradient_boosting"]
    mlp = run_notes["multilayer_perceptron"]
    embedding = run_notes["embedding_network"]
    convergence = (
        "The logistic solver stopped with a convergence warning before the tolerance was met."
        if logistic["convergence_warning"]
        else "The logistic solver met its tolerance."
    )
    boosting_stop = (
        f"Boosting trained {int(boosting['n_iter_trained'])} iterations and predictions use "
        f"iteration {int(boosting['best_iteration'])}, where validation macro-F1 on the "
        f"early-stopping curve was {float(boosting['validation_macro_f1_at_best']):.3f}."
    )
    if boosting["hit_max_iter"]:
        boosting_stop += " Training reached the pre-set iteration cap."
    else:
        boosting_stop += " Early stopping halted training before the iteration cap."

    gap_lines = []
    ordered = pd.concat(
        [
            approved,
            comparison.loc[comparison["approval"].eq(PENDING_APPROVAL)],
        ],
        ignore_index=True,
    )
    for row in ordered.itertuples(index=False):
        train_f1 = _metric_value(metrics, row.model, "train", "macro_f1")
        val_f1 = _metric_value(metrics, row.model, "validation", "macro_f1")
        status = "" if row.approval == APPROVED_APPROVAL else " (pending lecturer approval)"
        gap_lines.append(
            f"- {row.model}{status}: training macro-F1 {train_f1:.3f}, validation macro-F1 {val_f1:.3f}, "
            f"gap {train_f1 - val_f1:+.3f}."
        )
    loss_curves = run_notes.get("neural_loss_curves", {})
    if loss_curves and all(bool(value) for value in loss_curves.values()):
        loss_sentence = (
            "On both neural networks, training loss fell while validation loss rose across the epochs that were run. "
            "Early stopping restored the epoch with the highest validation macro-F1."
        )
    else:
        loss_sentence = (
            "Early stopping restored the epoch with the highest validation macro-F1."
        )

    champion_name = str(champion["model"])
    lines = [
        "# Phase 5 model experiments",
        "",
        "Notebooks 04 and 05 fit the traditional models and the neural models on the same chronological partitions. "
        "The provisional champion is chosen from validation macro-F1. The locked test is not scored in this phase.",
        "",
        "## Split",
        "",
        f"- Train: {split_line('train')}.",
        f"- Validation: {split_line('validation')}.",
        f"- Locked test, not scored: {split_line('test')}.",
        "",
        "The split dates are the Phase 4 dates. They were not moved to improve a model score.",
        "",
        "## Encoding",
        "",
        f"The one-hot training matrix has {int(one_hot['rows']):,} rows and {int(one_hot['columns']):,} columns, "
        f"with {int(one_hot['nonzero']):,} stored values. A dense float32 copy would be "
        f"{float(one_hot['dense_float32_gib']):.1f} GiB. "
        "The sparse matrix is practical for logistic regression and the multilayer perceptron. "
        "A dense random forest on that matrix is not practical in this environment, so the tree model is "
        "histogram gradient boosting on native categorical codes.",
        "",
        "Tree codes are fit on the training period. A level needs at least 200 training rows and the encoder keeps "
        f"at most {TREE_MAX_KEPT_CATEGORIES} levels, because the booster accepts at most {TREE_MAX_BINS} categories "
        "including the missing and rare levels.",
        "",
        *tree_lines,
        "",
        "The embedding encoder keeps the same 200-row rare rule and does not "
        "apply the 255-level cap. Unseen categories use index 0, whose embedding stays at zero. "
        "Work centre is not one of its inputs.",
        "",
        "## Imbalance",
        "",
        "All approved models use balanced class weights computed from the training counts. "
        "Validation rows are not resampled. SMOTE is not used. The predictors are categorical codes or one-hot "
        "columns, and SMOTE on raw category strings would invent levels that were not logged.",
        "",
        "## Models",
        "",
        "Logistic regression is the interpretable baseline from Phase 4: saga, C = 1, L2, balanced class weights, "
        f"max_iter = {int(logistic['max_iter'])}, tol = {float(logistic['tol'])}. "
        f"Fit time was {float(logistic['fit_seconds']):.1f} seconds. {convergence}",
        "",
        "Histogram gradient boosting uses log loss, learning rate 0.1, at most 31 leaves, "
        "minimum 50 samples per leaf, L2 regularisation 1, balanced class weights, and early stopping on the "
        f"chronological validation macro-F1. Fit time was {float(boosting['fit_seconds']):.1f} seconds. {boosting_stop}",
        "",
        "The multilayer perceptron reads the training-only one-hot pipeline. It has three output logits, "
        f"hidden sizes {mlp['hidden_sizes']}, ReLU, dropout {mlp['dropout']}, AdamW with learning rate "
        f"{mlp['learning_rate']} and weight decay {mlp['weight_decay']}, batch size {mlp['batch_size']}, "
        f"and class-weighted cross-entropy. It has {int(mlp['n_parameters']):,} parameters and "
        f"{int(mlp['n_features']):,} inputs. "
        f"It ran {int(mlp['epochs_ran'])} epochs, restored epoch {int(mlp['best_epoch'])}, "
        f"and {'stopped early' if mlp['stopped_early'] else 'reached the epoch cap'}. "
        f"Fit time was {float(mlp['fit_seconds']):.1f} seconds.",
        "",
        "The second neural architecture is a categorical embedding network. "
        + (
            "Lecturer approval has been recorded for this architecture. No approval date was supplied, "
            "and the architecture was not changed. "
            if not comparison.loc[comparison["model"].eq("embedding_network") & comparison["approval"].eq(PENDING_APPROVAL)].shape[0]
            else "It is pending lecturer approval and is not an approved final experiment. "
        )
        + f"Embedding dimension {embedding['embedding_dim']}, hidden size {embedding['hidden_size']}, "
        f"dropout {embedding['dropout']}, AdamW learning rate {embedding['learning_rate']}, "
        f"weight decay {embedding['weight_decay']}, batch size {embedding['batch_size']}. "
        f"It ran {int(embedding['epochs_ran'])} epochs and restored epoch {int(embedding['best_epoch'])}. "
        f"Fit time was {float(embedding['fit_seconds']):.1f} seconds.",
        "",
        "## Validation",
        "",
    ]
    for row in comparison.itertuples(index=False):
        status = "approved" if row.approval == APPROVED_APPROVAL else "pending lecturer approval"
        lines.append(
            f"- {row.model} ({status}): accuracy {row.accuracy:.3f}, macro-F1 {row.macro_f1:.3f}, "
            f"weighted-F1 {row.weighted_f1:.3f}, Resolved Slow recall {row.recall_resolved_slow:.3f}, "
            f"Unresolved recall {row.recall_unresolved:.3f}, macro ROC-AUC {row.macro_roc_auc:.3f}, "
            f"macro PR-AUC {row.macro_average_precision:.3f}, fit {row.fit_seconds:.1f} seconds."
        )
    lines.extend(
        [
            "",
            "## Training and validation",
            "",
            *gap_lines,
            "",
            "Learning curves for the neural models show training and validation loss and macro-F1. "
            "Those training curves are computed with dropout turned off, on the full training period, "
            "so they can be compared with the validation curves. "
            f"{loss_sentence} "
            "The boosting curve shows validation macro-F1 at each iteration. "
            "The per-iteration boosting training score is the library's 10,000-row subset, so the "
            "training macro-F1 in the table above is the one computed on the full training period.",
            "",
            "## Provisional champion",
            "",
            f"The provisional champion is {champion_name}. "
            f"Its validation macro-F1 is {float(champion['macro_f1']):.3f}. "
            f"Validation {score_line(champion_name, 'validation')}. "
            "This choice uses validation macro-F1 among the approved models only. "
            "The locked test was not scored and did not enter the decision.",
            "",
            *recall_notes,
            "",
            pending_text,
            "",
            "These settings were not grid-searched. A later notebook can tune the champion. "
            "This phase does not claim a locked-test result.",
            "",
        ]
    )
    return "\n".join(lines)
