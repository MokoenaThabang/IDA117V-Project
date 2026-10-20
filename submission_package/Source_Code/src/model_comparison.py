"""Check the recorded champion-comparator comparison.

Notebook 06 writes the comparator into the frozen decision before the locked
test is opened. Notebook 07 must use that recorded name. This module checks
the saved prediction files and writes the McNemar record. It does not fit a
model and it does not replace either prediction file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config
from src.optimisation import assert_recorded_comparator, comparison_record
from src.statistics import (
    assert_aligned_observations,
    assert_matches_split,
    mcnemar_test,
    paired_correctness_counts,
)

INTERPRETATION = (
    "The test compares paired error outcomes on the same locked-test observations. "
    "It does not establish that one model is better on every class, and it does not "
    "by itself establish general superiority."
)


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing comparison file: {path}")
    return path


def saved_comparison_record(
    comparison: pd.DataFrame,
    frozen: dict,
    champion_predictions: pd.DataFrame,
    paired_predictions: pd.DataFrame,
    test_identifiers=None,
    test_labels=None,
) -> dict:
    """Rebuild the McNemar record from saved tables. Do not invent predictions."""
    selection = assert_recorded_comparator(frozen, comparison)
    for column in ("champion_predicted", "comparator_predicted", "source_index", "y_true"):
        if column not in paired_predictions.columns:
            raise ValueError(f"The paired predictions are missing {column}.")
    if "champion_predicted" not in champion_predictions.columns:
        raise ValueError("The champion predictions are missing champion_predicted.")
    assert_aligned_observations(champion_predictions, paired_predictions)
    if not champion_predictions["champion_predicted"].reset_index(drop=True).equals(
        paired_predictions["champion_predicted"].reset_index(drop=True)
    ):
        raise ValueError("The champion labels in the paired file do not match notebook 06.")
    if test_identifiers is not None or test_labels is not None:
        if test_identifiers is None or test_labels is None:
            raise ValueError("Both the test identifiers and the test labels are required.")
        assert_matches_split(champion_predictions, test_identifiers, test_labels)
    counts = paired_correctness_counts(
        paired_predictions["y_true"],
        paired_predictions["champion_predicted"],
        paired_predictions["comparator_predicted"],
    )
    result = mcnemar_test(counts)
    return {
        **selection,
        "comparator_model": selection["comparator"],
        "champion_model": selection["champion"],
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
        "predictions_regenerated": False,
    }


def write_saved_comparison_record() -> dict:
    """Verify the files notebook 07 already saved and write the record."""
    comparison_path = _require(config.TABLES_DIR / "05_model_comparison.csv")
    frozen_path = _require(config.TABLES_DIR / "06_frozen_decision.json")
    champion_path = _require(config.TABLES_DIR / "06_test_predictions.parquet")
    paired_path = _require(config.TABLES_DIR / "07_paired_predictions.parquet")
    from src.preprocessing import SOURCE_CATEGORICAL_COLUMNS, chronological_splits

    columns = list(SOURCE_CATEGORICAL_COLUMNS) + ["Created_On_Date", "resolution_outcome"]
    frame = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "service_requests_labelled.parquet",
        columns=columns,
    )
    splits, _summary, _exclusions = chronological_splits(frame)
    test = splits["test"]
    del frame
    del splits
    record = saved_comparison_record(
        pd.read_csv(comparison_path),
        json.loads(frozen_path.read_text(encoding="utf-8")),
        pd.read_parquet(champion_path),
        pd.read_parquet(paired_path),
        test.index.to_numpy(),
        test["resolution_outcome"].to_numpy(),
    )
    destination = config.TABLES_DIR / "07_comparison_record.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    record = write_saved_comparison_record()
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
