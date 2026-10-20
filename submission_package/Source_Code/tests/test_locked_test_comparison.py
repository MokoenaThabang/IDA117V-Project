"""Checks for the locked-test ROC helper and the saved Question 6.2 record."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.locked_test_comparison import macro_ovr_curve  # noqa: E402


class MacroCurveTests(unittest.TestCase):
    def test_separated_scores_have_a_high_macro_auc(self) -> None:
        labels = np.array(["Resolved Fast", "Resolved Slow", "Unresolved"] * 20)
        probabilities = np.zeros((len(labels), 3))
        lookup = {"Resolved Fast": 0, "Resolved Slow": 1, "Unresolved": 2}
        for row, label in enumerate(labels):
            probabilities[row, lookup[label]] = 0.9
            probabilities[row, (lookup[label] + 1) % 3] = 0.1
        curve = macro_ovr_curve(labels, probabilities)
        self.assertGreater(curve["macro_roc_auc"], 0.95)
        self.assertEqual(len(curve["fpr"]), 201)
        self.assertTrue(all(row["curve_defined"] for row in curve["classes"]))

    def test_a_single_class_column_is_rejected(self) -> None:
        labels = np.array(["Resolved Fast", "Resolved Slow"])
        with self.assertRaises(ValueError):
            macro_ovr_curve(labels, np.ones((2, 1)))


class Question62RecordTests(unittest.TestCase):
    def test_saved_record_compares_the_champion_with_the_embedding_network(self) -> None:
        record_path = PROJECT_ROOT / "results" / "tables" / "13_question62_record.json"
        champion_path = PROJECT_ROOT / "results" / "tables" / "06_test_predictions.parquet"
        embedding_path = PROJECT_ROOT / "results" / "tables" / "13_embedding_locked_test_predictions.parquet"
        historical_path = PROJECT_ROOT / "results" / "tables" / "07_mcnemar_test.csv"
        self.assertTrue(record_path.is_file())
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(record["question"], "6.2")
        self.assertEqual(record["champion"], "hist_gradient_boosting")
        self.assertEqual(record["comparator"], "embedding_network")
        self.assertEqual(record["historical_comparator"], "multilayer_perceptron")
        self.assertFalse(record["champion_changed"])
        self.assertFalse(record["retrained_on_test"])
        self.assertEqual(record["test_rows"], 422033)
        self.assertEqual(record["contingency"]["both_correct"], 229320)
        self.assertEqual(record["contingency"]["champion_only"], 23738)
        self.assertEqual(record["contingency"]["comparator_only"], 18625)
        self.assertEqual(record["contingency"]["both_wrong"], 150350)
        self.assertAlmostEqual(record["statistic"], 616.8718929254302)
        self.assertIn("paired error", record["interpretation"])
        champion = pd.read_parquet(champion_path, columns=["source_index", "y_true"])
        embedding = pd.read_parquet(embedding_path, columns=["source_index", "y_true"])
        self.assertTrue(champion["source_index"].equals(embedding["source_index"]))
        self.assertTrue(champion["y_true"].astype(str).equals(embedding["y_true"].astype(str)))
        historical = pd.read_csv(historical_path)
        self.assertEqual(historical["comparator"].iloc[0], "multilayer_perceptron")
        self.assertEqual(int(historical["champion_only"].iloc[0]), 19282)

    def test_common_roc_table_covers_the_required_models(self) -> None:
        table_path = PROJECT_ROOT / "results" / "tables" / "13_roc_all_models_locked_test.csv"
        figure_path = PROJECT_ROOT / "results" / "figures" / "13_roc_all_models_locked_test.png"
        config_path = PROJECT_ROOT / "results" / "tables" / "13_roc_all_models_config.json"
        self.assertTrue(figure_path.is_file())
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["skipped"], {})
        self.assertTrue(payload["champion_unchanged"])
        self.assertFalse(payload["probabilities_fabricated"])
        table = pd.read_csv(table_path)
        expected = {
            "Histogram gradient boosting",
            "Embedding network",
            "Most-frequent dummy",
            "Logistic regression",
            "Multilayer perceptron",
        }
        self.assertEqual(set(table["model"]), expected)
        self.assertTrue(table["curve_defined"].all())
        champion = table.loc[table["model"].eq("Histogram gradient boosting")]
        self.assertAlmostEqual(float(champion["macro_roc_auc"].iloc[0]), 0.7241433691367188)
        dummy = table.loc[table["model"].eq("Most-frequent dummy")]
        self.assertTrue(np.allclose(dummy["roc_auc"].to_numpy(dtype=float), 0.5))
