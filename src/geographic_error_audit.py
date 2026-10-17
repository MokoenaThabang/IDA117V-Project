"""Group the frozen champion's locked-test errors by suburb and by ward.

The predictions are the ones already saved by notebook 06. This module does
not fit a model. Small groups are counted and flagged. They are not treated
as evidence of geographic bias. No notification id or row-level record is written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.training import save_json
from src.visualization import save_figure, use_project_style

MIN_GROUP_SUPPORT = 200
MIN_CLASS_SUPPORT = 30
PLACEHOLDERS = {"#", "Not assigned", "GENERIC TOWN 0000", "000", "0", "MUL"}


def _clean_group(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.mask(text.isna() | text.eq(""), "(blank)")
    return text.astype("object")


def _class_counts(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Precision, recall, and F1 pieces for every group and class."""
    rows = []
    groups = frame.groupby(group_column, dropna=False).size().index
    for label in config.TARGET_LABELS:
        actual = frame["y_true"].eq(label)
        predicted = frame["predicted"].eq(label)
        true_positive = frame.loc[actual & predicted].groupby(group_column, dropna=False).size()
        predicted_positive = frame.loc[predicted].groupby(group_column, dropna=False).size()
        actual_positive = frame.loc[actual].groupby(group_column, dropna=False).size()
        rows.append(
            pd.DataFrame(
                {
                    group_column: groups,
                    "label": label,
                    "true_positive": true_positive.reindex(groups, fill_value=0).to_numpy(),
                    "predicted_positive": predicted_positive.reindex(groups, fill_value=0).to_numpy(),
                    "actual_positive": actual_positive.reindex(groups, fill_value=0).to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _audit(frame: pd.DataFrame, group_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped_size = frame.groupby(group_column, dropna=False).size().rename("n")
    correct = (
        frame.assign(_correct=frame["y_true"].eq(frame["predicted"]))
        .groupby(group_column, dropna=False)["_correct"]
        .sum()
        .astype(int)
        .rename("n_correct")
    )
    summary = pd.concat([grouped_size, correct], axis=1).reset_index()
    summary["n_errors"] = summary["n"] - summary["n_correct"]
    summary["accuracy"] = summary["n_correct"] / summary["n"]
    summary["error_rate"] = summary["n_errors"] / summary["n"]
    summary["low_support"] = summary["n"] < MIN_GROUP_SUPPORT
    summary["placeholder_or_blank"] = summary[group_column].isin(PLACEHOLDERS | {"(blank)"})

    errors = frame.loc[frame["y_true"].ne(frame["predicted"]), [group_column, "y_true", "predicted"]]
    if errors.empty:
        summary["most_common_error"] = pd.NA
        summary["most_common_error_count"] = 0
    else:
        pattern_counts = (
            errors.groupby([group_column, "y_true", "predicted"], dropna=False)
            .size()
            .reset_index(name="pattern_count")
        )
        pattern_counts["most_common_error"] = (
            pattern_counts["y_true"].astype(str) + " predicted as " + pattern_counts["predicted"].astype(str)
        )
        top = (
            pattern_counts.sort_values(["pattern_count", "most_common_error"], ascending=[False, True])
            .groupby(group_column, dropna=False)
            .head(1)
        )
        summary = summary.merge(
            top[[group_column, "most_common_error", "pattern_count"]],
            on=group_column,
            how="left",
        )
        summary = summary.rename(columns={"pattern_count": "most_common_error_count"})
        summary["most_common_error_count"] = summary["most_common_error_count"].fillna(0).astype(int)

    pieces = _class_counts(frame, group_column)
    pieces = pieces.merge(summary[[group_column, "n", "low_support"]], on=group_column, how="left")
    report_class = (~pieces["low_support"]) & (pieces["actual_positive"] >= MIN_CLASS_SUPPORT)
    precision = np.full(len(pieces), np.nan)
    recall = np.full(len(pieces), np.nan)
    f1 = np.full(len(pieces), np.nan)
    predicted_ok = pieces["predicted_positive"].to_numpy() > 0
    actual_ok = pieces["actual_positive"].to_numpy() > 0
    usable = report_class.to_numpy()
    tp = pieces["true_positive"].to_numpy(dtype=float)
    pp = pieces["predicted_positive"].to_numpy(dtype=float)
    ap = pieces["actual_positive"].to_numpy(dtype=float)
    precision[usable & predicted_ok] = tp[usable & predicted_ok] / pp[usable & predicted_ok]
    recall[usable & actual_ok] = tp[usable & actual_ok] / ap[usable & actual_ok]
    finite = np.isfinite(precision) & np.isfinite(recall) & ((precision + recall) > 0)
    f1[finite] = 2 * precision[finite] * recall[finite] / (precision[finite] + recall[finite])
    pieces["precision"] = precision
    pieces["recall"] = recall
    pieces["f1"] = f1
    pieces["class_metrics_reported"] = usable & (np.isfinite(precision) | np.isfinite(recall))
    pieces["precision_undefined"] = usable & ~predicted_ok
    pieces["recall_undefined"] = ~actual_ok

    confusion = (
        frame.groupby([group_column, "y_true", "predicted"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    confusion = confusion.merge(summary[[group_column, "n", "low_support"]], on=group_column, how="left")
    confusion = confusion.loc[~confusion["low_support"]].drop(columns=["low_support"])
    summary = summary.sort_values(["n", group_column], ascending=[False, True], kind="mergesort")
    return summary, pieces, confusion


def _plot_largest(summary: pd.DataFrame, group_column: str, filename: str) -> None:
    import matplotlib.pyplot as plt

    use_project_style()
    eligible = summary.loc[~summary["low_support"]].head(15).iloc[::-1]
    figure, axis = plt.subplots(figsize=(10, 6.2), constrained_layout=True)
    labels = [str(value) if len(str(value)) <= 32 else str(value)[:29] + "..." for value in eligible[group_column]]
    axis.barh(labels, eligible["error_rate"], color="#0072B2")
    axis.set_xlabel("Error rate")
    axis.set_xlim(0, 1)
    axis.set_title(
        f"Largest {group_column.lower()} groups on the locked test (n ≥ {MIN_GROUP_SUPPORT})\n"
        "Smaller groups are omitted. A higher bar is not, by itself, evidence of geographic bias."
    )
    save_figure(figure, filename, dpi=300)
    plt.close(figure)


def run_audit() -> dict:
    predictions = pd.read_parquet(
        config.TABLES_DIR / "06_test_predictions.parquet",
        columns=["source_index", "y_true", "champion_predicted"],
    )
    import pyarrow.parquet as pq

    table = pq.read_table(
        config.PROCESSED_DATA_DIR / "service_requests_labelled.parquet",
        columns=["Suburb", "Ward"],
    )
    taken = table.take(predictions["source_index"].to_numpy())
    frame = taken.to_pandas()
    frame["y_true"] = predictions["y_true"].to_numpy()
    frame["predicted"] = predictions["champion_predicted"].to_numpy()
    if not frame["y_true"].isin(config.TARGET_LABELS).all():
        raise RuntimeError("Locked-test labels include a value outside the three classes.")
    frame["suburb"] = _clean_group(frame["Suburb"])
    frame["ward"] = _clean_group(frame["Ward"])

    suburb_summary, suburb_class, suburb_confusion = _audit(frame, "suburb")
    ward_summary, ward_class, ward_confusion = _audit(frame, "ward")
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    suburb_summary.to_csv(config.TABLES_DIR / "09_suburb_error_audit.csv", index=False)
    ward_summary.to_csv(config.TABLES_DIR / "09_ward_error_audit.csv", index=False)
    suburb_class.to_csv(config.TABLES_DIR / "09_suburb_class_metrics.csv", index=False)
    ward_class.to_csv(config.TABLES_DIR / "09_ward_class_metrics.csv", index=False)
    suburb_confusion.to_csv(config.TABLES_DIR / "09_suburb_confusion_counts.csv", index=False)
    ward_confusion.to_csv(config.TABLES_DIR / "09_ward_confusion_counts.csv", index=False)
    _plot_largest(suburb_summary, "suburb", "09_suburb_error_rates.png")
    _plot_largest(ward_summary, "ward", "09_ward_error_rates.png")

    def coverage(column: str, summary: pd.DataFrame) -> dict:
        return {
            "groups": int(len(summary)),
            "rows": int(summary["n"].sum()),
            "groups_below_support": int(summary["low_support"].sum()),
            "rows_in_groups_below_support": int(summary.loc[summary["low_support"], "n"].sum()),
            "blank_rows": int(summary.loc[summary[column].eq("(blank)"), "n"].sum())
            if (summary[column] == "(blank)").any()
            else 0,
            "placeholder_rows": int(summary.loc[summary["placeholder_or_blank"], "n"].sum()),
        }

    payload = {
        "prediction_file": "results/tables/06_test_predictions.parquet",
        "model": "frozen histogram gradient boosting champion",
        "split": "locked test",
        "min_group_support_for_class_metrics": MIN_GROUP_SUPPORT,
        "min_class_support": MIN_CLASS_SUPPORT,
        "suburb": coverage("suburb", suburb_summary),
        "ward": coverage("ward", ward_summary),
        "row_level_records_saved": False,
        "interpretation": (
            "Groups below the support threshold are not evidence of geographic bias. "
            "Placeholder and blank codes are stored values, not measured neighbourhood effects."
        ),
    }
    save_json(payload, config.TABLES_DIR / "09_geographic_audit_config.json")
    print(payload)
    return payload


def main() -> None:
    run_audit()


if __name__ == "__main__":
    main()
