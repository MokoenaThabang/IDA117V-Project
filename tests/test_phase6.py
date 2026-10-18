"""Phase 6 tuning, ablation, and McNemar checks. The rows are synthetic."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TARGET_LABELS  # noqa: E402
from src.optimisation import (  # noqa: E402
    GEOGRAPHIC_FEATURES,
    ablation_plan,
    assert_recorded_comparator,
    choose_final_candidate,
    comparison_record,
    frozen_iteration,
    full_feature_list,
    geographic_categorical_columns,
    parameter_trials,
    render_phase6_report,
    score_parameters_on_folds,
    select_search_winner,
    strongest_approved_comparator,
    training_fold_table,
)
from src.preprocessing import CATEGORICAL_FEATURES, MODEL_FEATURES  # noqa: E402
from src.model_comparison import saved_comparison_record  # noqa: E402
from src.statistics import assert_aligned_observations, mcnemar_test, paired_correctness_counts  # noqa: E402
from src.training import APPROVED_APPROVAL, PENDING_APPROVAL, TreeCategoryEncoder, summarise_predictions  # noqa: E402


def _frame(dates, complaints):
    rows = []
    for date, complaint in zip(dates, complaints):
        rows.append(
            {
                "Created_On_Date": pd.Timestamp(date),
                "C3_Complaint_Type": complaint,
                "Suburb": "north" if complaint == "water" else "south",
                "Ward": "1",
                "Sub_Council": "A",
                "created_day_of_week": "Monday",
                "created_month": "January",
                "created_is_weekend": 0.0,
            }
        )
    return pd.DataFrame(rows)


class SearchProtocolTest(unittest.TestCase):
    def test_sampler_is_seeded_and_keeps_the_baseline(self) -> None:
        first = parameter_trials(n_trials=8, seed=42)
        second = parameter_trials(n_trials=8, seed=42)
        self.assertEqual(len(first), 9)
        self.assertEqual(first[0]["origin"], "untuned_baseline")
        self.assertEqual(first[0]["parameters"]["learning_rate"], 0.1)
        self.assertEqual(
            [trial["parameters"] for trial in first],
            [trial["parameters"] for trial in second],
        )
        self.assertEqual(len({tuple(sorted(trial["parameters"].items())) for trial in first[1:]}), 8)

    def test_tie_keeps_the_untuned_validation_model(self) -> None:
        self.assertEqual(choose_final_candidate(0.48, 0.49), "tuned")
        self.assertEqual(choose_final_candidate(0.48, 0.48), "untuned")
        self.assertEqual(choose_final_candidate(0.48, 0.47), "untuned")
        self.assertEqual(frozen_iteration(0), 1)
        self.assertEqual(frozen_iteration(2), 2)

    def test_comparator_ignores_the_pending_network_and_the_champion(self) -> None:
        comparison = pd.DataFrame(
            [
                {"model": "hist_gradient_boosting", "approval": "approved", "macro_f1": 0.48},
                {"model": "multilayer_perceptron", "approval": "approved", "macro_f1": 0.476},
                {"model": "logistic_regression", "approval": "approved", "macro_f1": 0.46},
                {"model": "embedding_network", "approval": "pending_lecturer_approval", "macro_f1": 0.90},
            ]
        )
        chosen = strongest_approved_comparator(comparison)
        self.assertEqual(chosen["model"], "multilayer_perceptron")

    def test_approved_embedding_is_barred_from_the_comparator(self) -> None:
        comparison = pd.DataFrame(
            [
                {
                    "model": "hist_gradient_boosting",
                    "approval": "approved",
                    "macro_f1": 0.482,
                    "recall_unresolved": 0.49,
                    "recall_resolved_slow": 0.42,
                },
                {
                    "model": "embedding_network",
                    "approval": "approved",
                    "macro_f1": 0.481,
                    "recall_unresolved": 0.46,
                    "recall_resolved_slow": 0.44,
                },
                {
                    "model": "multilayer_perceptron",
                    "approval": "approved",
                    "macro_f1": 0.476,
                    "recall_unresolved": 0.39,
                    "recall_resolved_slow": 0.45,
                },
                {
                    "model": "logistic_regression",
                    "approval": "approved",
                    "macro_f1": 0.460,
                    "recall_unresolved": 0.37,
                    "recall_resolved_slow": 0.46,
                },
            ]
        )
        self.assertEqual(strongest_approved_comparator(comparison)["model"], "multilayer_perceptron")
        unbarred = strongest_approved_comparator(comparison, excluded=())
        self.assertEqual(unbarred["model"], "embedding_network")
        record = comparison_record(comparison)
        self.assertEqual(record["comparator"], "multilayer_perceptron")
        self.assertFalse(record["locked_test_used_for_selection"])
        frozen = {
            "champion_family": "hist_gradient_boosting",
            "comparator": "multilayer_perceptron",
            "comparator_phase5_validation_macro_f1": 0.476,
            "split_id": record["split_id"],
        }
        self.assertEqual(assert_recorded_comparator(frozen, comparison)["comparator"], "multilayer_perceptron")
        frozen["comparator"] = "embedding_network"
        with self.assertRaises(ValueError):
            assert_recorded_comparator(frozen, comparison)

    def test_tie_break_uses_unresolved_recall_before_the_model_name(self) -> None:
        comparison = pd.DataFrame(
            [
                {
                    "model": "hist_gradient_boosting",
                    "approval": "approved",
                    "macro_f1": 0.5,
                    "recall_unresolved": 0.1,
                    "recall_resolved_slow": 0.1,
                },
                {
                    "model": "aaa_model",
                    "approval": "approved",
                    "macro_f1": 0.4,
                    "recall_unresolved": 0.2,
                    "recall_resolved_slow": 0.9,
                },
                {
                    "model": "zzz_model",
                    "approval": "approved",
                    "macro_f1": 0.4,
                    "recall_unresolved": 0.3,
                    "recall_resolved_slow": 0.1,
                },
            ]
        )
        self.assertEqual(strongest_approved_comparator(comparison, excluded=())["model"], "zzz_model")

    def test_work_centre_ablation_is_already_excluded(self) -> None:
        plan = ablation_plan(full_feature_list())
        work = plan.loc[plan["experiment"].eq("remove_work_centre")].iloc[0]
        self.assertFalse(bool(work["executed_fit"]))
        self.assertIn("already excluded", work["note"])
        self.assertNotIn("Work_Center", full_feature_list())
        self.assertNotIn("Suburb", geographic_categorical_columns())
        with self.assertRaises(ValueError):
            ablation_plan(full_feature_list() + ["Work_Center"])

    def test_encoder_can_drop_geographic_features(self) -> None:
        frame = _frame(["2023-01-02", "2023-01-03"], ["water", "power"])
        features = frame.loc[:, list(MODEL_FEATURES)]
        encoder = TreeCategoryEncoder(categorical_columns=geographic_categorical_columns(), min_count=1)
        coded = encoder.fit(features).transform(features)
        self.assertNotIn("Suburb", coded.columns)
        self.assertIn("C3_Complaint_Type", coded.columns)
        self.assertIn("created_is_weekend", coded.columns)

    def test_training_folds_cover_the_synthetic_calendar(self) -> None:
        dates = pd.date_range("2023-01-15", "2024-12-15", freq="MS")
        table = training_fold_table(pd.Series(dates))
        self.assertEqual(list(table["fold"]), [1, 2, 3])
        self.assertTrue((table["train_rows"] > 0).all())
        self.assertTrue((table["validate_rows"] > 0).all())

    def test_one_setting_can_be_scored_on_the_training_folds(self) -> None:
        dates = []
        complaints = []
        labels = []
        cycle = ["water", "power", "sewer"]
        label_map = {
            "water": TARGET_LABELS[0],
            "power": TARGET_LABELS[1],
            "sewer": TARGET_LABELS[2],
        }
        for date in pd.date_range("2023-01-15", "2024-12-15", freq="15D"):
            for index, complaint in enumerate(cycle):
                dates.append(date)
                complaints.append(complaint)
                labels.append(label_map[complaint])
        frame = _frame(dates, complaints)
        detail = score_parameters_on_folds(
            frame.loc[:, list(MODEL_FEATURES)],
            np.array(labels, dtype=object),
            frame["Created_On_Date"],
            {
                "learning_rate": 0.2,
                "max_leaf_nodes": 4,
                "min_samples_leaf": 5,
                "l2_regularization": 0.1,
            },
        )
        self.assertEqual(len(detail), 3)
        self.assertTrue(detail["macro_f1"].between(0, 1).all())
        self.assertTrue((detail["best_iteration"] >= 0).all())

    def test_report_does_not_invent_a_locked_test_score(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "trial": 0,
                    "origin": "untuned_baseline",
                    "mean_macro_f1": 0.4,
                    "median_best_iteration": 2,
                }
            ]
        )
        validation = pd.DataFrame(
            [
                {
                    "model": "untuned",
                    "accuracy": 0.5,
                    "macro_f1": 0.4,
                    "recall_resolved_slow": 0.2,
                    "recall_unresolved": 0.3,
                }
            ]
        )
        text = render_phase6_report(
            summary,
            validation,
            {"selected": "untuned"},
            test_metrics=None,
            ablation=None,
            mcnemar=None,
        )
        self.assertIn("has not been scored", text)
        self.assertIn("untuned", text)
        winner = select_search_winner(summary)
        self.assertEqual(int(winner["trial"]), 0)


class McNemarTest(unittest.TestCase):
    def test_counts_and_exact_binomial_for_a_small_discordant_set(self) -> None:
        actual = np.array(["a"] * 10)
        champion = np.array(["a"] * 8 + ["b", "b"])
        comparator = np.array(["a"] * 6 + ["b"] * 4)
        counts = paired_correctness_counts(actual, champion, comparator)
        self.assertEqual(counts["rows"], 10)
        self.assertEqual(
            counts["both_correct"]
            + counts["champion_only"]
            + counts["comparator_only"]
            + counts["both_wrong"],
            10,
        )
        result = mcnemar_test(counts)
        self.assertEqual(result["method"], "exact binomial")
        expected = stats.binomtest(counts["champion_only"], result["discordant"], 0.5, alternative="two-sided")
        self.assertAlmostEqual(result["p_value"], float(expected.pvalue))
        self.assertLess(result["discordant"], 25)

    def test_large_discordant_set_uses_the_corrected_chi_square(self) -> None:
        champion_only = 30
        comparator_only = 10
        counts = {
            "rows": 80,
            "both_correct": 20,
            "champion_only": champion_only,
            "comparator_only": comparator_only,
            "both_wrong": 20,
        }
        result = mcnemar_test(counts)
        statistic = (abs(champion_only - comparator_only) - 1) ** 2 / (champion_only + comparator_only)
        self.assertEqual(result["method"], "chi-square with continuity correction")
        self.assertAlmostEqual(result["statistic"], statistic)
        self.assertAlmostEqual(result["p_value"], float(stats.chi2.sf(statistic, df=1)))
        self.assertTrue(result["significant"])
        self.assertIn("every class", result["conclusion"])

    def test_alignment_rejects_a_reordered_or_relabelled_file(self) -> None:
        reference = pd.DataFrame(
            {
                "source_index": [3, 1, 2],
                "y_true": ["Resolved Fast", "Resolved Slow", "Unresolved"],
                "champion_predicted": ["Resolved Fast", "Resolved Fast", "Unresolved"],
            }
        )
        swapped = reference.iloc[[1, 0, 2]].reset_index(drop=True)
        with self.assertRaises(ValueError):
            assert_aligned_observations(reference, swapped)
        relabelled = reference.copy()
        relabelled.loc[0, "y_true"] = "Unresolved"
        with self.assertRaises(ValueError):
            assert_aligned_observations(reference, relabelled)
        missing = reference.drop(columns=["source_index"])
        with self.assertRaises(ValueError):
            assert_aligned_observations(reference, missing)

    def test_saved_mcnemar_files_agree_on_the_recorded_comparator(self) -> None:
        comparison_path = PROJECT_ROOT / "results" / "tables" / "05_model_comparison.csv"
        frozen_path = PROJECT_ROOT / "results" / "tables" / "06_frozen_decision.json"
        champion_path = PROJECT_ROOT / "results" / "tables" / "06_test_predictions.parquet"
        paired_path = PROJECT_ROOT / "results" / "tables" / "07_paired_predictions.parquet"
        for path in (comparison_path, frozen_path, champion_path, paired_path):
            self.assertTrue(path.is_file(), path)
        import json

        record = saved_comparison_record(
            pd.read_csv(comparison_path),
            json.loads(frozen_path.read_text(encoding="utf-8")),
            pd.read_parquet(champion_path),
            pd.read_parquet(paired_path),
        )
        self.assertEqual(record["champion_model"], "hist_gradient_boosting")
        self.assertEqual(record["comparator_model"], "multilayer_perceptron")
        self.assertEqual(record["test_rows"], 422033)
        self.assertEqual(record["contingency"]["champion_only"], 19282)
        self.assertEqual(record["contingency"]["comparator_only"], 21854)
        self.assertAlmostEqual(record["statistic"], 160.6875)
        self.assertIn("paired error", record["interpretation"])
        self.assertNotIn("embedding_network", record["comparator_model"])

    def test_locked_test_still_requires_authorisation(self) -> None:
        with self.assertRaises(RuntimeError):
            summarise_predictions(
                list(TARGET_LABELS),
                list(TARGET_LABELS),
                np.eye(3),
                model="hist_gradient_boosting",
                split="test",
                approval=APPROVED_APPROVAL,
            )
        summary, _per_class = summarise_predictions(
            list(TARGET_LABELS),
            list(TARGET_LABELS),
            np.eye(3),
            model="hist_gradient_boosting",
            split="test",
            approval=APPROVED_APPROVAL,
            locked_test_authorised=True,
        )
        self.assertAlmostEqual(float(summary["macro_f1"].iloc[0]), 1.0)
        self.assertIn("Suburb", GEOGRAPHIC_FEATURES)
        self.assertIn("C3_Complaint_Type", CATEGORICAL_FEATURES)
        self.assertNotEqual(PENDING_APPROVAL, APPROVED_APPROVAL)


if __name__ == "__main__":
    unittest.main()
