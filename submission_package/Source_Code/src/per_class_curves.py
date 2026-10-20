"""One-vs-rest ROC and precision-recall curves for the frozen champion.

The curves use the locked-test probabilities saved by notebook 06. They are
not computed from hard class labels. Existing macro scores are left in place
and checked against these probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from src import config
from src.evaluation import ranking_scores
from src.training import save_json
from src.visualization import save_figure, use_project_style

PROBABILITY_COLUMNS = {
    "Resolved Fast": "p_resolved_fast",
    "Resolved Slow": "p_resolved_slow",
    "Unresolved": "p_unresolved",
}


def _curve_table(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    curves = []
    for index, label in enumerate(config.TARGET_LABELS):
        actual = y_true == label
        support = int(actual.sum())
        scores = probabilities[:, index]
        if support == 0 or support == len(actual):
            rows.append(
                {
                    "label": label,
                    "support": support,
                    "roc_auc": np.nan,
                    "average_precision": np.nan,
                    "curve_defined": False,
                    "reason": "The class is absent or is the only class, so the curve is undefined.",
                }
            )
            continue
        false_positive, true_positive, _thresholds = roc_curve(actual, scores)
        precision, recall, _thresholds = precision_recall_curve(actual, scores)
        roc_auc = float(auc(false_positive, true_positive))
        average_precision = float(average_precision_score(actual, scores))
        rows.append(
            {
                "label": label,
                "support": support,
                "roc_auc": roc_auc,
                "average_precision": average_precision,
                "curve_defined": True,
                "reason": "",
            }
        )
        curves.append(
            {
                "label": label,
                "support": support,
                "roc_auc": roc_auc,
                "average_precision": average_precision,
                "fpr": false_positive,
                "tpr": true_positive,
                "precision": precision,
                "recall": recall,
            }
        )
    return pd.DataFrame(rows), curves


def _plot(curves: list[dict], kind: str, filename: str) -> None:
    import matplotlib.pyplot as plt

    use_project_style()
    figure, axis = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    colours = {"Resolved Fast": "#0072B2", "Resolved Slow": "#E69F00", "Unresolved": "#CC79A7"}
    for row in curves:
        if kind == "roc":
            axis.plot(
                row["fpr"],
                row["tpr"],
                color=colours[row["label"]],
                linewidth=2,
                label=f"{row['label']} (AUC {row['roc_auc']:.3f}, support {row['support']:,})",
            )
        else:
            axis.plot(
                row["recall"],
                row["precision"],
                color=colours[row["label"]],
                linewidth=2,
                label=f"{row['label']} (AP {row['average_precision']:.3f}, support {row['support']:,})",
            )
    if kind == "roc":
        axis.plot([0, 1], [0, 1], color="#666666", linewidth=1, linestyle="--", label="Chance")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_title("Locked test: one-vs-rest ROC")
    else:
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title("Locked test: one-vs-rest precision-recall")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="best", fontsize=8)
    save_figure(figure, filename, dpi=300)
    plt.close(figure)


def run_curves() -> dict:
    predictions = pd.read_parquet(config.TABLES_DIR / "06_test_predictions.parquet")
    required = ["y_true", *PROBABILITY_COLUMNS.values()]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise RuntimeError("Saved predictions are missing columns: " + ", ".join(missing))
    y_true = predictions["y_true"].to_numpy()
    probabilities = np.column_stack(
        [predictions[PROBABILITY_COLUMNS[label]].to_numpy(dtype=float) for label in config.TARGET_LABELS]
    )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Saved probabilities contain a non-finite value.")
    table, curves = _curve_table(y_true, probabilities)
    macro = ranking_scores(y_true, probabilities)
    saved = pd.read_csv(config.TABLES_DIR / "06_locked_test_summary.csv").iloc[0]
    roc_gap = abs(float(macro["macro_roc_auc"]) - float(saved["macro_roc_auc"]))
    pr_gap = abs(float(macro["macro_average_precision"]) - float(saved["macro_average_precision"]))
    if roc_gap > 1e-6 or pr_gap > 1e-6:
        raise RuntimeError(
            "Recomputed macro ranking scores do not match the saved locked-test summary. "
            f"ROC gap {roc_gap}, PR gap {pr_gap}."
        )
    _plot(curves, "roc", "09_roc_per_class_locked_test.png")
    _plot(curves, "pr", "09_pr_per_class_locked_test.png")
    export = table
    export["macro_roc_auc_retained"] = float(saved["macro_roc_auc"])
    export["macro_average_precision_retained"] = float(saved["macro_average_precision"])
    path = config.TABLES_DIR / "09_per_class_ranking_locked_test.csv"
    export.to_csv(path, index=False)
    payload = {
        "source_predictions": "results/tables/06_test_predictions.parquet",
        "prior_summary_left_unchanged": "results/tables/06_locked_test_summary.csv",
        "macro_roc_auc": float(saved["macro_roc_auc"]),
        "macro_average_precision": float(saved["macro_average_precision"]),
        "recomputed_gap_roc": roc_gap,
        "recomputed_gap_average_precision": pr_gap,
        "curves_use_probabilities": True,
    }
    save_json(payload, config.TABLES_DIR / "09_per_class_curves_config.json")
    print(export.round(4).to_string(index=False))
    print(f"Wrote {path}")
    return payload


def main() -> None:
    run_curves()


if __name__ == "__main__":
    main()
