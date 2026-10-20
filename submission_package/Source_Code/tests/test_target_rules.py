"""Target, date, and observation rules on toy rows.

These rows are not City of Cape Town records.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_table  # noqa: E402
from src.preprocessing import (  # noqa: E402
    build_resolution_target,
    parse_portal_datetime,
    require_date_columns,
    select_model_features,
)


class TargetRuleTests(unittest.TestCase):
    def test_epoch_milliseconds_use_cape_town_calendar_day(self) -> None:
        parsed = parse_portal_datetime(pd.Series([1672524000000]))
        self.assertEqual(parsed.iloc[0].date().isoformat(), "2023-01-01")

    def test_seven_day_boundary_and_long_duration_kept(self) -> None:
        frame = pd.DataFrame(
            {
                "Created On Date": ["2026-09-01", "2026-09-01", "2026-01-01"],
                "Completed Date": ["2026-09-08", "2026-09-09", "2027-02-15"],
            }
        )
        labelled = build_resolution_target(
            frame,
            "Created On Date",
            "Completed Date",
            fast_slow_threshold_days=7,
            observation_window_days=7,
            reference_date="2027-03-01",
        )
        self.assertEqual(
            list(labelled["resolution_outcome"]),
            ["Resolved Fast", "Resolved Slow", "Resolved Slow"],
        )
        self.assertEqual(list(labelled["duration_flag"]), [None, None, "unusual_long"])
        self.assertEqual(labelled.loc[0, "resolution_days"], 7.0)
        self.assertEqual(labelled.loc[2, "resolution_days"], 410.0)

    def test_missing_malformed_and_insufficient_observation(self) -> None:
        frame = pd.DataFrame(
            {
                "Created On Date": [None, "not-a-date", "2026-10-01", "2026-09-01", "2026-10-02"],
                "Completed Date": [None, None, "still-open-text", None, None],
            }
        )
        labelled = build_resolution_target(
            frame,
            "Created On Date",
            "Completed Date",
            fast_slow_threshold_days=7,
            observation_window_days=7,
            reference_date="2026-10-06",
        )
        self.assertEqual(
            list(labelled["label_status"]),
            [
                "missing_created_date",
                "malformed_created_date",
                "malformed_completed_date",
                "labelled",
                "insufficient_follow_up",
            ],
        )
        self.assertEqual(labelled.loc[3, "resolution_outcome"], "Unresolved")
        self.assertIsNone(labelled.loc[4, "resolution_outcome"])
        self.assertNotEqual(labelled.loc[2, "resolution_outcome"], "Unresolved")

    def test_completion_before_creation_is_excluded(self) -> None:
        frame = pd.DataFrame(
            {
                "Created On Date": ["2026-09-10"],
                "Completed Date": ["2026-09-01"],
            }
        )
        labelled = build_resolution_target(
            frame,
            "Created On Date",
            "Completed Date",
            fast_slow_threshold_days=7,
            observation_window_days=7,
            reference_date="2026-10-06",
        )
        self.assertEqual(labelled.loc[0, "label_status"], "completion_before_creation")
        self.assertIsNone(labelled.loc[0, "resolution_outcome"])

    def test_missing_columns_stop_with_the_actual_names(self) -> None:
        frame = pd.DataFrame({"Ward": ["1"], "Suburb": ["Sea Point"]})
        with self.assertRaises(KeyError) as caught:
            require_date_columns(frame)
        message = str(caught.exception)
        self.assertIn("Ward", message)
        self.assertIn("Suburb", message)
        self.assertIn("Created_On_Date", message)

    def test_outcome_fields_are_not_model_inputs(self) -> None:
        columns = [
            "Created_On_Date",
            "Completed_Date",
            "Changed_on",
            "resolution_days",
            "resolution_outcome",
            "X_Y_Co_ordinate_1",
            "Notification",
            "C3_Complaint_Type",
        ]
        self.assertEqual(select_model_features(columns), ["Created_On_Date", "C3_Complaint_Type"])

    def test_excel_file_can_be_loaded(self) -> None:
        frame = pd.DataFrame({"Created On Date": ["2026-09-01"], "Ward": ["12"]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.xlsx"
            frame.to_excel(path, index=False)
            loaded = load_raw_table(path)
        self.assertEqual(list(loaded.columns), ["Created On Date", "Ward"])
        self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
