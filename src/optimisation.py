"""Training-period tuning for the provisional champion.

The search uses chronological folds inside 2023–2024. It does not read the
2025 validation period or the locked test. sklearn's random K-fold would
place later training requests into earlier folds, so the sampler is
``ParameterSampler`` and the folds are written out explicitly.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.evaluation import classification_scores
from src.preprocessing import CATEGORICAL_FEATURES, MODEL_FEATURES
from src.training import (
    HIST_GRADIENT_BOOSTING_PARAMETERS,
    TREE_MAX_BINS,
    TreeCategoryEncoder,
    use_best_validation_iteration,
)

PROVISIONAL_CHAMPION = "hist_gradient_boosting"
COMPARATOR_EXCLUSIONS = ("embedding_network",)
TUNING_TRIALS = 8
TUNING_SEED = RANDOM_SEED
SEARCH_MAX_ITER = 40
SEARCH_N_ITER_NO_CHANGE = 6
SIGNIFICANCE_ALPHA = 0.05

# Expanding windows inside the training period. Each validation slice starts
# the day after train_end and ends on validate_end, inclusive.
TRAINING_FOLDS = (
    {"fold": 1, "train_end": "2023-06-30", "validate_end": "2023-12-31"},
    {"fold": 2, "train_end": "2023-12-31", "validate_end": "2024-06-30"},
    {"fold": 3, "train_end": "2024-06-30", "validate_end": "2024-12-31"},
)

SEARCH_SPACE = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [50, 200, 500],
    "l2_regularization": [0.1, 1.0, 10.0],
}

GEOGRAPHIC_FEATURES = ("Suburb", "Ward", "Sub_Council")
WORK_CENTRE_FEATURE = "Work_Center"


def untuned_parameters() -> dict:
    """Phase 5 boosting settings that the search is allowed to change."""
    return {
        "learning_rate": float(HIST_GRADIENT_BOOSTING_PARAMETERS["learning_rate"]),
        "max_leaf_nodes": int(HIST_GRADIENT_BOOSTING_PARAMETERS["max_leaf_nodes"]),
        "min_samples_leaf": int(HIST_GRADIENT_BOOSTING_PARAMETERS["min_samples_leaf"]),
        "l2_regularization": float(HIST_GRADIENT_BOOSTING_PARAMETERS["l2_regularization"]),
    }


def _plain(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def parameter_trials(n_trials: int = TUNING_TRIALS, seed: int = TUNING_SEED) -> list[dict]:
    """Baseline settings plus ``n_trials`` draws from ``ParameterSampler``.

    The baseline is always evaluated. Random draws are not a shuffled K-fold.
    """
    from sklearn.model_selection import ParameterSampler

    if int(n_trials) < 1:
        raise ValueError(f"n_trials must be at least 1. Got {n_trials}.")
    trials = [
        {
            "trial": 0,
            "origin": "untuned_baseline",
            "parameters": untuned_parameters(),
        }
    ]
    sampler = ParameterSampler(SEARCH_SPACE, n_iter=int(n_trials), random_state=int(seed))
    for index, draw in enumerate(sampler, start=1):
        trials.append(
            {
                "trial": index,
                "origin": "parameter_sampler",
                "parameters": {key: _plain(value) for key, value in draw.items()},
            }
        )
    return trials


def training_fold_table(created: pd.Series) -> pd.DataFrame:
    """Row counts for the chronological training folds. Dates must already be training rows."""
    stamps = pd.to_datetime(created)
    rows = []
    for fold in TRAINING_FOLDS:
        train_end = pd.Timestamp(fold["train_end"])
        validate_end = pd.Timestamp(fold["validate_end"])
        train_mask = stamps.le(train_end)
        valid_mask = stamps.gt(train_end) & stamps.le(validate_end)
        if int(train_mask.sum()) == 0 or int(valid_mask.sum()) == 0:
            raise ValueError(
                f"Fold {fold['fold']} has an empty side. "
                f"Train rows {int(train_mask.sum())}, validation rows {int(valid_mask.sum())}."
            )
        rows.append(
            {
                "fold": int(fold["fold"]),
                "train_end": fold["train_end"],
                "validate_end": fold["validate_end"],
                "train_rows": int(train_mask.sum()),
                "validate_rows": int(valid_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def _fold_masks(created: pd.Series, fold: dict) -> tuple[np.ndarray, np.ndarray]:
    stamps = pd.to_datetime(created)
    train_end = pd.Timestamp(fold["train_end"])
    validate_end = pd.Timestamp(fold["validate_end"])
    train_mask = stamps.le(train_end).to_numpy()
    valid_mask = (stamps.gt(train_end) & stamps.le(validate_end)).to_numpy()
    return train_mask, valid_mask


def _new_booster(parameters: dict, categorical_columns: list[str], max_iter: int, early_stopping: bool, class_weight):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(parameters["learning_rate"]),
        max_iter=int(max_iter),
        max_leaf_nodes=int(parameters["max_leaf_nodes"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        l2_regularization=float(parameters["l2_regularization"]),
        max_bins=TREE_MAX_BINS,
        early_stopping=bool(early_stopping),
        scoring="f1_macro",
        n_iter_no_change=SEARCH_N_ITER_NO_CHANGE,
        class_weight=class_weight,
        random_state=TUNING_SEED,
        categorical_features=list(categorical_columns),
    )


def score_parameters_on_folds(
    features: pd.DataFrame,
    labels,
    created: pd.Series,
    parameters: dict,
    categorical_columns: list[str] | None = None,
    class_weight="balanced",
) -> pd.DataFrame:
    """Fit one setting on each training fold and score macro-F1 on that fold's later slice.

    The encoder is fit on the fold's training rows only. Early stopping uses
    that fold's later slice, not the 2025 validation period.
    """
    categorical = list(CATEGORICAL_FEATURES if categorical_columns is None else categorical_columns)
    features = features.reset_index(drop=True)
    created = pd.Series(pd.to_datetime(created)).reset_index(drop=True)
    labels = np.asarray(labels)
    if not (len(features) == len(created) == len(labels)):
        raise ValueError(
            "Features, dates, and labels must have the same number of rows. "
            f"Got {len(features)}, {len(created)}, {len(labels)}."
        )
    rows = []
    for fold in TRAINING_FOLDS:
        train_mask, valid_mask = _fold_masks(created, fold)
        train_positions = np.flatnonzero(train_mask)
        valid_positions = np.flatnonzero(valid_mask)
        encoder = TreeCategoryEncoder(categorical_columns=categorical)
        encoder.fit(features.iloc[train_positions])
        coded_train = encoder.transform(features.iloc[train_positions])
        coded_valid = encoder.transform(features.iloc[valid_positions])
        model = _new_booster(
            parameters,
            categorical,
            max_iter=SEARCH_MAX_ITER,
            early_stopping=True,
            class_weight=class_weight,
        )
        started = time.perf_counter()
        model.fit(
            coded_train,
            labels[train_positions],
            X_val=coded_valid,
            y_val=labels[valid_positions],
        )
        best_iteration = use_best_validation_iteration(model)
        elapsed = time.perf_counter() - started
        predicted = model.predict(coded_valid)
        macro_f1 = classification_scores(labels[valid_positions], predicted)["macro_f1"]
        rows.append(
            {
                "fold": int(fold["fold"]),
                "train_end": fold["train_end"],
                "validate_end": fold["validate_end"],
                "macro_f1": float(macro_f1),
                "best_iteration": int(best_iteration),
                "fit_seconds": float(elapsed),
            }
        )
    return pd.DataFrame(rows)


def run_champion_search(
    features: pd.DataFrame,
    labels,
    created: pd.Series,
    n_trials: int = TUNING_TRIALS,
    seed: int = TUNING_SEED,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the untuned settings and the random trials on the training folds."""
    detail_frames = []
    summary_rows = []
    for trial in parameter_trials(n_trials=n_trials, seed=seed):
        if verbose:
            print(
                f"Trial {trial['trial']} ({trial['origin']}): {trial['parameters']}",
                flush=True,
            )
        detail = score_parameters_on_folds(features, labels, created, trial["parameters"])
        detail.insert(0, "origin", trial["origin"])
        detail.insert(0, "trial", trial["trial"])
        for key, value in trial["parameters"].items():
            detail[key] = value
        detail_frames.append(detail)
        summary_rows.append(
            {
                "trial": trial["trial"],
                "origin": trial["origin"],
                **trial["parameters"],
                "mean_macro_f1": float(detail["macro_f1"].mean()),
                "median_best_iteration": int(np.median(detail["best_iteration"])),
                "fit_seconds": float(detail["fit_seconds"].sum()),
            }
        )
        if verbose:
            print(
                f"  mean macro-F1 {summary_rows[-1]['mean_macro_f1']:.4f}, "
                f"median iteration {summary_rows[-1]['median_best_iteration']}",
                flush=True,
            )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_macro_f1", "trial"],
        ascending=[False, True],
        kind="mergesort",
    )
    return summary.reset_index(drop=True), pd.concat(detail_frames, ignore_index=True)


def frozen_iteration(median_best_iteration: int) -> int:
    """Iteration cap for a refit that no longer looks at a validation slice."""
    chosen = int(median_best_iteration)
    if chosen < 1:
        return 1
    return chosen


def select_search_winner(summary: pd.DataFrame) -> pd.Series:
    """Highest mean training-fold macro-F1. Ties keep the earlier trial number."""
    ranked = summary.sort_values(
        ["mean_macro_f1", "trial"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def choose_final_candidate(untuned_validation_macro_f1: float, tuned_validation_macro_f1: float) -> str:
    """Freeze the tuned refit only when its validation macro-F1 is higher.

    A tie keeps the untuned settings. The locked test is not an input.
    """
    if float(tuned_validation_macro_f1) > float(untuned_validation_macro_f1):
        return "tuned"
    return "untuned"


def strongest_approved_comparator(comparison: pd.DataFrame, champion: str = PROVISIONAL_CHAMPION) -> pd.Series:
    """Highest validation macro-F1 among approved models other than the champion.

    Pending-approval models are excluded. This uses the Phase 5 validation
    table and does not read the locked test.
    """
    required = {"model", "approval", "macro_f1"}
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError("Comparator table is missing columns: " + ", ".join(sorted(missing)) + ".")
    rows = comparison.loc[
        comparison["approval"].eq("approved") & ~comparison["model"].eq(champion)
    ].copy()
    if rows.empty:
        raise ValueError("No approved comparator is available.")
    ranked = rows.sort_values(
        ["macro_f1", "model"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def fit_frozen_boosting(
    features: pd.DataFrame,
    labels,
    parameters: dict,
    max_iter: int,
    categorical_columns: list[str] | None = None,
    class_weight="balanced",
):
    """Refit on the rows it is given, with a frozen iteration cap and no early stopping."""
    categorical = list(CATEGORICAL_FEATURES if categorical_columns is None else categorical_columns)
    encoder = TreeCategoryEncoder(categorical_columns=categorical)
    encoder.fit(features)
    coded = encoder.transform(features)
    model = _new_booster(
        parameters,
        categorical,
        max_iter=frozen_iteration(max_iter),
        early_stopping=False,
        class_weight=class_weight,
    )
    started = time.perf_counter()
    model.fit(coded, np.asarray(labels))
    elapsed = time.perf_counter() - started
    return encoder, model, float(elapsed)


def ablation_plan(full_features: list[str]) -> pd.DataFrame:
    """Three pre-specified ablations. Work Center is already absent from the full model."""
    if WORK_CENTRE_FEATURE in full_features:
        raise ValueError(
            "The frozen feature list includes Work Center. Phase 6 expected it to be excluded."
        )
    missing_geography = [column for column in GEOGRAPHIC_FEATURES if column not in full_features]
    if missing_geography:
        raise ValueError(
            "The full model is missing geographic features: " + ", ".join(missing_geography) + "."
        )
    without_geography = [
        column
        for column in full_features
        if column not in GEOGRAPHIC_FEATURES
    ]
    return pd.DataFrame(
        [
            {
                "experiment": "remove_geographic_features",
                "component_changed": "Suburb, Ward, and Sub_Council",
                "executed_fit": True,
                "features": ", ".join(without_geography),
                "note": "Retrained without suburb, ward, and subcouncil. Complaint type and calendar fields stay.",
            },
            {
                "experiment": "remove_work_centre",
                "component_changed": "Work Center",
                "executed_fit": False,
                "features": ", ".join(full_features),
                "note": (
                    "Work Center is already excluded from the frozen pipeline because the extract "
                    "has no assignment time. Removing it does not change the inputs. No second model was fit."
                ),
            },
            {
                "experiment": "no_class_weight",
                "component_changed": "class_weight",
                "executed_fit": True,
                "features": ", ".join(full_features),
                "note": "Same features and frozen hyperparameters with class_weight left unset. SMOTE is not used.",
            },
        ]
    )


def geographic_categorical_columns() -> list[str]:
    """Categorical predictors that remain after the geographic ablation."""
    return [column for column in CATEGORICAL_FEATURES if column not in GEOGRAPHIC_FEATURES]


def full_feature_list() -> list[str]:
    return list(MODEL_FEATURES)


def render_phase6_report(
    search_summary: pd.DataFrame,
    validation_comparison: pd.DataFrame,
    frozen: dict,
    test_metrics: pd.DataFrame | None,
    ablation: pd.DataFrame | None,
    mcnemar: dict | None,
) -> str:
    """Write the Phase 6 report from measured tables."""
    winner = select_search_winner(search_summary)
    chosen_name = str(frozen["selected"])
    lines = [
        "# Phase 6 evaluation",
        "",
        "The provisional champion family is histogram gradient boosting, from the Phase 5 validation macro-F1 ranking. "
        "Pending embedding results did not enter that choice. Tuning used chronological folds inside the training period only.",
        "",
        "## Search",
        "",
        f"Random draws: {TUNING_TRIALS}. Seed: {TUNING_SEED}. "
        "The untuned Phase 5 settings were scored on the same folds. "
        "The score is mean macro-F1. Early stopping inside a fold uses that fold's later slice.",
        "",
        f"The highest training-fold mean is trial {int(winner['trial'])} "
        f"({winner['origin']}), mean macro-F1 {float(winner['mean_macro_f1']):.3f}, "
        f"median selected iteration {int(winner['median_best_iteration'])}.",
        "",
        "## Validation comparison",
        "",
    ]
    for row in validation_comparison.itertuples(index=False):
        lines.append(
            f"- {row.model}: validation macro-F1 {float(row.macro_f1):.3f}, "
            f"accuracy {float(row.accuracy):.3f}, "
            f"Resolved Slow recall {float(row.recall_resolved_slow):.3f}, "
            f"Unresolved recall {float(row.recall_unresolved):.3f}."
        )
    lines.extend(
        [
            "",
            f"The frozen candidate is `{chosen_name}`. "
            "The rule keeps the tuned refit only when its validation macro-F1 is higher than the untuned refit. "
            "A tie keeps the untuned settings. This choice was saved before the locked test was scored.",
            "",
        ]
    )
    if test_metrics is None:
        lines.append("The locked test has not been scored.")
    else:
        lines.append("## Locked test")
        lines.append("")
        for row in test_metrics.itertuples(index=False):
            lines.append(
                f"- {row.model}: accuracy {float(row.accuracy):.3f}, macro-F1 {float(row.macro_f1):.3f}, "
                f"weighted-F1 {float(row.weighted_f1):.3f}, "
                f"Resolved Slow recall {float(row.recall_resolved_slow):.3f}, "
                f"Unresolved recall {float(row.recall_unresolved):.3f}, "
                f"macro ROC-AUC {float(row.macro_roc_auc):.3f}, "
                f"macro PR-AUC {float(row.macro_average_precision):.3f}."
            )
        lines.append("")
        lines.append("The frozen parameters were not changed after these test metrics were computed.")
        lines.append("")
    if ablation is not None:
        lines.extend(["## Ablations", ""])
        for row in ablation.itertuples(index=False):
            lines.append(
                f"- {row.experiment}: validation macro-F1 {float(row.macro_f1):.3f} "
                f"(change {float(row.delta_macro_f1):+.3f}), "
                f"Resolved Slow recall {float(row.recall_resolved_slow):.3f} "
                f"(change {float(row.delta_recall_resolved_slow):+.3f}), "
                f"Unresolved recall {float(row.recall_unresolved):.3f} "
                f"(change {float(row.delta_recall_unresolved):+.3f}). "
                f"{row.note}"
            )
        lines.append("")
    if mcnemar is not None:
        lines.extend(
            [
                "## McNemar",
                "",
                f"Comparator: {mcnemar['comparator']}. "
                f"Both correct {mcnemar['both_correct']:,}. "
                f"Champion only {mcnemar['champion_only']:,}. "
                f"Comparator only {mcnemar['comparator_only']:,}. "
                f"Both wrong {mcnemar['both_wrong']:,}.",
                "",
                f"Method: {mcnemar['method']}. Statistic: {mcnemar['statistic']}. "
                f"p-value: {float(mcnemar['p_value']):.4g}. Threshold: {SIGNIFICANCE_ALPHA}. "
                f"Conclusion: {mcnemar['conclusion']}",
                "",
                "The test compares paired error rates on the same cases. "
                "It does not show that one model is better on every class.",
                "",
                f"Accuracy difference (champion minus comparator): {float(mcnemar['accuracy_difference']):+.4f}. "
                f"Macro-F1 difference: {float(mcnemar['macro_f1_difference']):+.4f}. "
                "Those gaps are the practical comparison. A small gap can still have a small p-value when the test set is large.",
                "",
            ]
        )
    return "\n".join(lines)


def recall_focus(scores: dict) -> dict[str, float]:
    """Pull Resolved Slow and Unresolved recall out of a classification_scores result."""
    found = {row["label"]: float(row["recall"]) for row in scores["per_class"]}
    missing = [label for label in ("Resolved Slow", "Unresolved") if label not in found]
    if missing:
        raise ValueError("Recall rows are missing: " + ", ".join(missing) + ".")
    return {
        "recall_resolved_slow": found["Resolved Slow"],
        "recall_unresolved": found["Unresolved"],
    }

