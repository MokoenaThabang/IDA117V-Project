"""Leakage-safe preprocessing checks. The rows are synthetic."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    MISSING_CATEGORY_LABEL,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    GroupRareCategories,
    build_baseline_preprocessor,
    chronological_splits,
    feature_decision_table,
    logging_feature_frame,
    make_logistic_baseline,
    make_most_frequent_baseline,
)


def _requests() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Created_On_Date": pd.to_datetime(
                [
                    "2023-01-02",
                    "2023-06-03",
                    "2024-12-31",
                    "2025-03-04",
                    "2025-11-08",
                    "2026-02-09",
                    "2026-04-30",
                    "2026-06-15",
                    "2024-01-06",
                ]
            ),
            "C3_Complaint_Type": [
                "No Power",
                "No Power",
                "Leak",
                "Leak",
                "No Power",
                "Leak",
                "Other",
                "No Power",
                "Rare Type",
            ],
            "Suburb": ["A", "A", "B", "A", "C", "B", "A", "Z", None],
            "Ward": ["1", "1", "2", "1", "2", "2", "1", "9", "1"],
            "Sub_Council": ["North", "North", "South", "North", "South", "South", "North", "East", "North"],
            "Work_Center": ["Depot", "Depot", "Depot", "Depot", "Depot", "Depot", "Depot", "Depot", "Depot"],
            "resolution_outcome": [
                "Resolved Fast",
                "Resolved Slow",
                "Unresolved",
                "Resolved Fast",
                "Resolved Slow",
                "Unresolved",
                "Resolved Fast",
                "Resolved Slow",
                None,
            ],
            "resolution_days": [1, 12, None, 2, 20, None, 0, 30, None],
            "Completed_Date": pd.to_datetime(
                [
                    "2023-01-03",
                    "2023-06-15",
                    None,
                    "2025-03-06",
                    "2025-11-28",
                    None,
                    "2026-04-30",
                    "2026-07-15",
                    None,
                ]
            ),
        }
    )


class PreprocessingRuleTest(unittest.TestCase):
    def test_feature_decisions_exclude_leakage_and_work_centre(self) -> None:
        decisions = feature_decision_table().set_index("column")["decision"]
        self.assertEqual(decisions["C3_Complaint_Type"], "include")
        self.assertEqual(decisions["created_is_weekend"], "include")
        for column in (
            "Work_Center",
            "Completed_Date",
            "Changed_on",
            "resolution_days",
            "resolution_outcome",
            "Notification",
            "Created_On_Date",
        ):
            self.assertEqual(decisions[column], "exclude", msg=column)

    def test_logging_features_drop_the_target_and_future_fields(self) -> None:
        features = logging_feature_frame(_requests())
        self.assertEqual(list(features.columns), list(MODEL_FEATURES))
        self.assertNotIn("resolution_outcome", features.columns)
        self.assertNotIn("resolution_days", features.columns)
        self.assertNotIn("Completed_Date", features.columns)
        self.assertNotIn("Work_Center", features.columns)
        self.assertNotIn("Created_On_Date", features.columns)
        monday = features.loc[0]
        self.assertEqual(monday["created_day_of_week"], "Monday")
        self.assertEqual(monday["created_month"], "January")
        self.assertEqual(monday["created_is_weekend"], 0.0)
        self.assertEqual(features.loc[4, "created_is_weekend"], 1.0)

    def test_missing_source_column_names_the_actual_columns(self) -> None:
        frame = _requests().drop(columns=["Suburb"])
        with self.assertRaises(ValueError) as caught:
            logging_feature_frame(frame)
        message = str(caught.exception)
        self.assertIn("Suburb", message)
        self.assertIn("Actual columns", message)

    def test_preprocessor_rejects_a_leakage_column(self) -> None:
        with self.assertRaises(ValueError) as caught:
            build_baseline_preprocessor(
                categorical_columns=["C3_Complaint_Type", "resolution_days"],
                numeric_columns=[],
            )
        self.assertIn("resolution_days", str(caught.exception))

    def test_chronological_split_keeps_every_mature_labelled_row(self) -> None:
        frame = _requests()
        splits, summary, exclusions = chronological_splits(frame)
        self.assertEqual(list(splits), ["train", "validation", "test"])
        self.assertEqual(len(splits["train"]), 3)
        self.assertEqual(len(splits["validation"]), 2)
        self.assertEqual(len(splits["test"]), 2)
        self.assertEqual(int(exclusions.loc[exclusions["reason"] == "labelled_after_mature_end", "rows"].iloc[0]), 1)
        self.assertEqual(
            int(exclusions.loc[exclusions["reason"] == "unlabelled_or_missing_created_date", "rows"].iloc[0]),
            1,
        )
        self.assertEqual(int(summary["rows"].sum()) + 2, len(frame))
        self.assertLessEqual(pd.Timestamp(summary.loc[summary["split"] == "train", "end_date"].iloc[0]), pd.Timestamp("2024-12-31"))
        self.assertGreaterEqual(
            pd.Timestamp(summary.loc[summary["split"] == "test", "start_date"].iloc[0]),
            pd.Timestamp("2026-01-01"),
        )

    def test_rare_grouper_learns_counts_from_fit_rows_only(self) -> None:
        train = pd.DataFrame({"Suburb": ["A", "A", "A", "B", None, ""]})
        later = pd.DataFrame({"Suburb": ["C", "B", "A", None]})
        grouper = GroupRareCategories(min_count=2)
        grouper.fit(train)
        self.assertEqual(grouper.kept_categories_["Suburb"], {"A"})
        transformed = grouper.transform(later)
        self.assertEqual(transformed[0, 0], "(rare)")
        self.assertEqual(transformed[1, 0], "(rare)")
        self.assertEqual(transformed[2, 0], "A")
        self.assertTrue(pd.isna(transformed[3, 0]))
        summary = grouper.rare_summary()
        self.assertEqual(int(summary.loc[0, "kept_categories"]), 1)
        self.assertEqual(int(summary.loc[0, "grouped_categories"]), 1)

    def test_pipeline_accepts_an_unseen_category_and_does_not_refit_on_it(self) -> None:
        train = pd.DataFrame(
            {
                "C3_Complaint_Type": ["No Power", "No Power", "Leak", "Leak"],
                "Suburb": ["Sea Point", "Sea Point", "Athlone", "Sea Point"],
                "Ward": ["54", "54", "48", "54"],
                "Sub_Council": ["16", "16", "16", "16"],
                "created_day_of_week": ["Monday", "Tuesday", "Monday", "Saturday"],
                "created_month": ["January", "January", "March", "March"],
                "created_is_weekend": [0.0, 0.0, 0.0, 1.0],
            }
        )
        later = pd.DataFrame(
            {
                "C3_Complaint_Type": ["Brand New"],
                "Suburb": ["Nowhere"],
                "Ward": ["54"],
                "Sub_Council": ["16"],
                "created_day_of_week": ["Monday"],
                "created_month": ["January"],
                "created_is_weekend": [1.0],
            }
        )
        preprocessor = build_baseline_preprocessor(min_count=2)
        preprocessor.fit(train)
        transformed = preprocessor.transform(later)
        self.assertEqual(transformed.shape[0], 1)
        rare = GroupRareCategories(min_count=2)
        rare.fit(train[list(CATEGORICAL_FEATURES)])
        grouped = rare.transform(later[list(CATEGORICAL_FEATURES)])
        self.assertEqual(grouped[0, 0], "(rare)")
        self.assertEqual(grouped[0, 1], "(rare)")
        encoder = preprocessor.named_transformers_["categorical"].named_steps["one_hot"]
        self.assertEqual(encoder.handle_unknown, "ignore")
        encoded = encoder.transform([["(not in training)"] * len(CATEGORICAL_FEATURES)])
        self.assertEqual(float(encoded.sum()), 0.0)
        scaler = preprocessor.named_transformers_["numeric"]
        self.assertAlmostEqual(float(scaler.mean_[0]), 0.25)
        self.assertIn("created_is_weekend", NUMERIC_FEATURES)

    def test_missing_categorical_level_is_distinct_from_rare(self) -> None:
        train = pd.DataFrame({"Suburb": ["A", "A", "B", None]})
        grouper = GroupRareCategories(min_count=2)
        grouped = grouper.fit_transform(train)
        from sklearn.impute import SimpleImputer

        imputed = SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY_LABEL).fit_transform(grouped)
        self.assertEqual(imputed[2, 0], "(rare)")
        self.assertEqual(imputed[3, 0], "(missing)")

    def test_baselines_are_configured_without_resampling(self) -> None:
        dummy = make_most_frequent_baseline()
        logistic = make_logistic_baseline()
        self.assertEqual(dummy.strategy, "most_frequent")
        self.assertEqual(logistic.class_weight, "balanced")
        self.assertEqual(logistic.C, 1.0)
        self.assertEqual(logistic.random_state, 42)


if __name__ == "__main__":
    unittest.main()
