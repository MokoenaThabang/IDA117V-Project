"""Checks for the exploratory tables. The rows are synthetic."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import eda, visualization  # noqa: E402


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sub_Council": ["A", None, "Not assigned", "A", "A"],
            "Ward": ["1", "1", None, "2", "2"],
            "Suburb": ["#", "SEA POINT", "GENERIC TOWN 0000", "SEA POINT", "SEA POINT"],
            "C3_Complaint_Type": ["No Power", "No Power", "Leak", "Leak", "Leak"],
            "Work_Center": ["Energy", "Energy", "Depot", "Depot", "Depot"],
            "Created_On_Date": pd.to_datetime(
                ["2024-01-15", "2024-01-20", "2024-03-02", "2024-03-18", "2024-03-19"]
            ),
            "Notifications_Created": [1.0, None, 1.0, 1.0, 1.0],
            "resolution_outcome": [
                "Resolved Fast",
                "Resolved Slow",
                "Unresolved",
                None,
                "Resolved Fast",
            ],
            "resolution_days": [0.0, 12.0, None, None, 3.0],
            "label_status": [
                "labelled",
                "labelled",
                "labelled",
                "insufficient_follow_up",
                "labelled",
            ],
            "duration_flag": [None, None, None, None, None],
            "coordinate_quality": [
                "missing",
                "plausible_local_grid",
                "outside_local_grid",
                "missing",
                "plausible_local_grid",
            ],
            "notification_id_repeated": [False, True, True, False, False],
        }
    )


class ExploratoryTableTest(unittest.TestCase):
    def test_target_distribution_keeps_unlabelled_rows_out_of_the_percentages(self) -> None:
        table = eda.target_distribution(_frame())
        labelled = table.loc[table["outcome"] != "(no label)"]
        self.assertEqual(list(labelled["outcome"]), list(eda.TARGET_LABELS))
        self.assertAlmostEqual(float(labelled["percent_of_labelled"].sum()), 100.0)
        self.assertEqual(int(labelled.loc[labelled["outcome"] == "Resolved Fast", "requests"].iloc[0]), 2)
        self.assertEqual(int(table.loc[table["outcome"] == "(no label)", "requests"].iloc[0]), 1)
        self.assertTrue(pd.isna(table.loc[table["outcome"] == "(no label)", "percent_of_labelled"].iloc[0]))

    def test_blank_completion_is_not_counted_as_predictor_missingness(self) -> None:
        frame = _frame()
        missing = eda.predictor_missingness(frame)
        self.assertEqual(list(missing["column"]), list(eda.PREDICTOR_COLUMNS))
        self.assertNotIn("resolution_days", set(missing["column"]))
        self.assertNotIn("resolution_outcome", set(missing["column"]))
        subcouncil = missing.loc[missing["column"] == "Sub_Council"].iloc[0]
        self.assertEqual(int(subcouncil["missing_count"]), 1)
        completion = eda.completion_status(frame)
        by_status = dict(zip(completion["status"], completion["requests"], strict=True))
        self.assertEqual(by_status["completion_recorded"], 3)
        self.assertEqual(by_status["blank_unresolved"], 1)
        self.assertEqual(by_status["blank_insufficient_follow_up"], 1)
        self.assertEqual(by_status["completion_before_creation"], 0)

    def test_invalid_completion_stays_separate_from_an_open_request(self) -> None:
        frame = _frame()
        frame.loc[4, "resolution_outcome"] = None
        frame.loc[4, "resolution_days"] = None
        frame.loc[4, "label_status"] = "completion_before_creation"
        completion = eda.completion_status(frame)
        by_status = dict(zip(completion["status"], completion["requests"], strict=True))
        self.assertEqual(by_status["completion_before_creation"], 1)
        self.assertEqual(by_status["blank_unresolved"], 1)

    def test_largest_groups_follow_the_count_rule(self) -> None:
        outcomes = eda.group_outcomes(_frame(), "Work_Center")
        self.assertEqual(list(outcomes["Work_Center"]), ["Depot", "Energy"])
        selected = eda.largest_groups(outcomes, top_n=1)
        self.assertEqual(list(selected["Work_Center"]), ["Depot"])
        self.assertEqual(int(selected["requests"].iloc[0]), 2)
        self.assertIn("largest 1 groups", selected["selection_rule"].iloc[0])
        small = eda.small_group_summary(outcomes, minimum_requests=3)
        self.assertEqual(small["small_groups"], 2)

    def test_monthly_activity_inserts_a_month_with_no_requests(self) -> None:
        monthly = eda.monthly_activity(_frame())
        self.assertEqual(list(monthly["month"]), ["2024-01", "2024-02", "2024-03"])
        february = monthly.loc[monthly["month"] == "2024-02"].iloc[0]
        self.assertEqual(int(february["requests"]), 0)
        self.assertTrue(bool(february["gap"]))
        self.assertTrue(pd.isna(february["resolved_fast_percent"]))
        march = monthly.loc[monthly["month"] == "2024-03"].iloc[0]
        self.assertEqual(int(march["requests"]), 3)
        self.assertEqual(int(march["labelled_requests"]), 2)
        self.assertEqual(int(march["insufficient_follow_up"]), 1)

    def test_duration_bins_keep_long_completions(self) -> None:
        frame = _frame()
        frame.loc[1, "resolution_days"] = 400
        bins = eda.duration_distribution(frame)
        self.assertEqual(int(bins["requests"].sum()), 3)
        long_bin = bins.loc[bins["bin"] == "More than 365 days", "requests"].iloc[0]
        self.assertEqual(int(long_bin), 1)
        extremes = eda.extreme_duration_tables(frame)
        metrics = dict(zip(extremes["summary"]["metric"], extremes["summary"]["value"], strict=True))
        self.assertEqual(metrics["longer_than_365_days"], 1)
        self.assertEqual(metrics["maximum_days"], 400)

    def test_duplicate_count_is_read_from_the_cleaning_log(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "02_cleaning_decisions.csv"
            pd.DataFrame(
                {
                    "decision": ["Remove exact duplicate business rows, ignoring ObjectId."],
                    "rows_affected": [12],
                }
            ).to_csv(path, index=False)
            self.assertEqual(eda.phase2_duplicate_removals(path), 12)
            quality = eda.data_quality_table(_frame(), exact_duplicates_removed=12)
            removed = quality.loc[quality["item"] == "exact_business_duplicates_removed"].iloc[0]
            self.assertEqual(int(removed["requests"]), 12)
            self.assertEqual(removed["source"], "phase2_cleaning_log")
            self.assertTrue(pd.isna(removed["percent_of_analysis_file"]))

    def test_a_missing_group_column_names_the_actual_columns(self) -> None:
        frame = _frame().drop(columns=["Work_Center"])
        with self.assertRaises(ValueError) as caught:
            eda.group_outcomes(frame, "Work_Center")
        message = str(caught.exception)
        self.assertIn("Work_Center", message)
        self.assertIn("Actual columns", message)

    def test_plots_save_without_using_coordinates(self) -> None:
        frame = _frame()
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            target = eda.target_distribution(frame)
            _figure, path = visualization.plot_target_distribution(target, directory=directory)
            self.assertTrue(path.is_file())
            missing = eda.predictor_missingness(frame)
            completion = eda.completion_status(frame)
            _figure, path = visualization.plot_missingness_and_completion(
                missing, completion, directory=directory
            )
            self.assertTrue(path.is_file())
            centres = eda.largest_groups(eda.group_outcomes(frame, "Work_Center"), top_n=2)
            _figure, path = visualization.plot_outcome_by_group(
                centres,
                "Work_Center",
                "Outcome by work centre",
                "work.png",
                directory=directory,
            )
            self.assertTrue(path.is_file())
            monthly = eda.monthly_activity(frame)
            _figure, path = visualization.plot_monthly_patterns(monthly, directory=directory)
            self.assertTrue(path.is_file())
            _figure, path = visualization.plot_duration_bins(
                eda.duration_distribution(frame), 7, directory=directory
            )
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
