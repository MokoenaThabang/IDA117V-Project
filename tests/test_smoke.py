"""Smoke tests for imports, project paths, and label rules.

The label example is a four-row toy table. It is not the City of Cape Town
extract and it is not a measured model result.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, data_loader, evaluation, preprocessing, visualization  # noqa: E402


class ProjectSmokeTest(unittest.TestCase):
    def test_source_modules_import(self) -> None:
        self.assertEqual(config.TARGET_LABELS[0], "Resolved Fast")
        self.assertEqual(
            config.SOURCE_PAGE_URL,
            "https://odp-cctegis.opendata.arcgis.com/datasets/"
            "90711027d33940b5a06e96ad8a8f7ede_0/explore",
        )
        self.assertEqual(config.FAST_SLOW_THRESHOLD_DAYS, 7)
        self.assertEqual(config.OBSERVATION_WINDOW_DAYS, 7)
        self.assertTrue(callable(data_loader.list_raw_files))
        self.assertTrue(callable(preprocessing.build_resolution_target))
        self.assertTrue(callable(evaluation.classification_scores))
        self.assertTrue(callable(visualization.save_figure))

    def test_project_directories_resolve(self) -> None:
        expected = [
            PROJECT_ROOT / "data" / "raw",
            PROJECT_ROOT / "data" / "interim",
            PROJECT_ROOT / "data" / "processed",
            PROJECT_ROOT / "notebooks",
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "models",
            PROJECT_ROOT / "app",
            PROJECT_ROOT / "results" / "figures",
            PROJECT_ROOT / "results" / "tables",
            PROJECT_ROOT / "reports",
            PROJECT_ROOT / "tests",
        ]
        for path in expected:
            self.assertTrue(path.is_dir(), msg=f"Missing directory: {path}")
        self.assertEqual(config.PROJECT_ROOT, PROJECT_ROOT)
        self.assertEqual(config.RAW_DATA_DIR, PROJECT_ROOT / "data" / "raw")
        self.assertEqual(config.FIGURES_DIR, PROJECT_ROOT / "results" / "figures")
        self.assertEqual(config.TABLES_DIR, PROJECT_ROOT / "results" / "tables")

    def test_empty_raw_directory_does_not_invent_a_file(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(data_loader.list_raw_files(Path(folder)), [])
        instructions = data_loader.manual_download_instructions()
        self.assertIn(config.SOURCE_PAGE_URL, instructions)
        self.assertIn("18 September 2026", instructions)
        self.assertIn("data/raw/", instructions)

    def test_leakage_columns_are_removed_and_created_date_is_kept(self) -> None:
        columns = [
            "Created On Date",
            "Completed Date",
            "Changed on",
            "resolution_outcome",
            "Notification",
        ]
        kept = preprocessing.select_model_features(columns)
        self.assertEqual(kept, ["Created On Date"])
        report = preprocessing.planned_column_report(columns)
        found = dict(zip(report["role"], report["found"], strict=True))
        self.assertTrue(found["created_on_date"])
        self.assertTrue(found["completed_date"])
        self.assertTrue(found["changed_on"])
        service_columns = ["Created_On_Date", "Completed_Date", "Changed_on", "Ward"]
        self.assertEqual(
            preprocessing.select_model_features(service_columns),
            ["Created_On_Date", "Ward"],
        )
        service_found = dict(
            zip(
                preprocessing.planned_column_report(service_columns)["role"],
                preprocessing.planned_column_report(service_columns)["found"],
                strict=True,
            )
        )
        self.assertTrue(all(service_found.values()))

    def test_target_rules_on_a_toy_table(self) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "Created On Date": [
                    "2026-09-01",
                    "2026-09-01",
                    "2026-09-01",
                    "2026-09-15",
                    "2026-09-01",
                    "2026-09-01",
                ],
                "Completed Date": [
                    "2026-09-02",
                    "2026-09-05",
                    None,
                    None,
                    "2026-08-01",
                    "2026-09-20",
                ],
            }
        )
        labelled = preprocessing.build_resolution_target(
            frame,
            created_column="Created On Date",
            completed_column="Completed Date",
            fast_slow_threshold_days=2,
            observation_window_days=10,
            reference_date="2026-09-18",
        )
        self.assertEqual(
            list(labelled["resolution_outcome"]),
            [
                "Resolved Fast",
                "Resolved Slow",
                "Unresolved",
                None,
                None,
                "Unresolved",
            ],
        )
        self.assertEqual(
            list(labelled["label_status"]),
            [
                "labelled",
                "labelled",
                "labelled",
                "insufficient_follow_up",
                "completion_before_creation",
                "labelled",
            ],
        )
        self.assertNotIn("Completed Date", labelled.columns)
        self.assertEqual(labelled.loc[0, "resolution_days"], 1.0)
        self.assertEqual(labelled.loc[1, "resolution_days"], 4.0)

    def test_macro_f1_on_perfect_toy_predictions(self) -> None:
        labels = ["Resolved Fast", "Resolved Slow", "Unresolved"]
        scores = evaluation.classification_scores(labels, labels, labels)
        self.assertEqual(scores["macro_f1"], 1.0)
        self.assertEqual(scores["accuracy"], 1.0)
        self.assertEqual(scores["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
