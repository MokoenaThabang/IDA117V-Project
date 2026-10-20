"""Classification metrics shared by the model notebooks.

Macro-F1 is the primary score. These functions only score arrays they are
given. They do not load data and they do not choose a champion model.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.config import TARGET_LABELS


def classification_scores(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = TARGET_LABELS,
) -> dict:
    """Score predicted class labels.

    ``labels`` fixes the class order. A class that is missing from ``y_true``
    still appears in the macro average, with precision, recall, and F1 of 0
    when ``zero_division=0``. That keeps a model from looking better because
    a rare class was absent.
    """
    label_list = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_list,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_list)
    per_class = []
    for index, label in enumerate(label_list):
        per_class.append(
            {
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=label_list, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=label_list, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "labels": label_list,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def ranking_scores(
    y_true: Sequence[str],
    y_proba: np.ndarray,
    labels: Sequence[str] = TARGET_LABELS,
) -> dict:
    """Macro one-vs-rest ROC-AUC and macro average precision.

    ``y_proba`` must have one column per label, in the same order as ``labels``.
    Average precision is computed from a one-vs-rest indicator matrix. These
    scores need probabilities; class labels alone are not enough.
    """
    label_list = list(labels)
    probabilities = np.asarray(y_proba, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(label_list):
        raise ValueError(
            "y_proba must have shape (n_samples, n_labels) in TARGET_LABELS order. "
            f"Got {probabilities.shape} for {len(label_list)} labels."
        )
    y_true_list = list(y_true)
    if len(y_true_list) != len(probabilities):
        raise ValueError("y_true and y_proba must have the same number of rows.")
    indicator = label_binarize(y_true_list, classes=label_list)
    # label_binarize returns one column when only two classes are requested.
    if indicator.ndim == 1 or indicator.shape[1] == 1:
        raise ValueError("ranking_scores expects three or more classes.")
    return {
        "macro_roc_auc": float(
            roc_auc_score(
                y_true_list,
                probabilities,
                labels=label_list,
                multi_class="ovr",
                average="macro",
            )
        ),
        "macro_average_precision": float(
            average_precision_score(indicator, probabilities, average="macro")
        ),
    }


def _recall_column(label: str) -> str:
    return "recall_" + label.lower().replace(" ", "_")


def metrics_tables(scores: dict, model: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn one classification_scores result into an overall row and class rows."""
    summary = {
        "model": model,
        "split": split,
        "accuracy": scores["accuracy"],
        "macro_f1": scores["macro_f1"],
        "weighted_f1": scores["weighted_f1"],
    }
    per_class_rows = []
    for row in scores["per_class"]:
        summary[_recall_column(row["label"])] = row["recall"]
        per_class_rows.append(
            {
                "model": model,
                "split": split,
                "label": row["label"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "support": row["support"],
            }
        )
    return pd.DataFrame([summary]), pd.DataFrame(per_class_rows)


def probability_summary(estimator, features, labels: Sequence[str] = TARGET_LABELS) -> dict:
    """Check that predict_proba covers the classes and that rows sum to one."""
    if not hasattr(estimator, "predict_proba"):
        raise AttributeError(f"{type(estimator).__name__} does not implement predict_proba.")
    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    class_names = [str(name) for name in estimator.classes_]
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError(
            "predict_proba must have one column per class. "
            f"Got shape {probabilities.shape} for classes {class_names}."
        )
    missing = [label for label in labels if label not in class_names]
    if missing:
        raise ValueError(
            "predict_proba is missing classes: "
            + ", ".join(missing)
            + f". Estimator classes: {', '.join(class_names)}."
        )
    row_sums = probabilities.sum(axis=1)
    return {
        "rows": int(probabilities.shape[0]),
        "columns": int(probabilities.shape[1]),
        "classes": class_names,
        "row_sum_min": float(row_sums.min()) if len(row_sums) else float("nan"),
        "row_sum_max": float(row_sums.max()) if len(row_sums) else float("nan"),
        "min_probability": float(probabilities.min()) if probabilities.size else float("nan"),
        "max_probability": float(probabilities.max()) if probabilities.size else float("nan"),
    }


def save_metric_tables(
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    directory: Path,
    prefix: str = "04_baseline",
) -> dict[str, Path]:
    """Write the overall and per-class metric tables."""
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / f"{prefix}_summary.csv"
    per_class_path = directory / f"{prefix}_per_class.csv"
    summary.to_csv(summary_path, index=False)
    per_class.to_csv(per_class_path, index=False)
    return {"summary": summary_path, "per_class": per_class_path}
