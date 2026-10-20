"""Checks for Phase 5 training utilities. The rows are synthetic."""

from __future__ import annotations

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

from src.config import TARGET_LABELS  # noqa: E402
from src.neural import EmbeddingNet, FeedForwardNet, train_embedding_network, train_feedforward  # noqa: E402
from src.preprocessing import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES  # noqa: E402
from src.training import (  # noqa: E402
    APPROVED_APPROVAL,
    PENDING_APPROVAL,
    EmbeddingCategoryEncoder,
    TreeCategoryEncoder,
    assert_split_may_be_scored,
    balanced_class_weights,
    base_experiment_config,
    make_hist_gradient_boosting,
    provisional_champion,
    render_phase5_report,
    summarise_predictions,
    use_best_validation_iteration,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in MODEL_FEATURES:
        if column not in frame.columns:
            frame[column] = "kept"
    return frame.loc[:, list(MODEL_FEATURES)]


def _labels(n_rows: int) -> np.ndarray:
    cycle = np.array(TARGET_LABELS, dtype=object)
    return cycle[np.arange(n_rows) % len(cycle)]


class CategoryEncoderTest(unittest.TestCase):
    def test_tree_encoder_caps_cardinality_and_handles_unseen(self) -> None:
        suburbs = [f"suburb-{index}" for index in range(300)]
        rows = []
        for suburb in suburbs:
            rows.append({"Suburb": suburb, "C3_Complaint_Type": "water"})
            rows.append({"Suburb": suburb, "C3_Complaint_Type": "water"})
        rows.append({"Suburb": pd.NA, "C3_Complaint_Type": "water"})
        rows.append({"Suburb": "suburb-0", "C3_Complaint_Type": "once"})
        frame = _frame(rows)
        encoder = TreeCategoryEncoder(min_count=2, max_kept=10)
        encoded = encoder.fit(frame).transform(frame)
        summary = encoder.level_summary().set_index("column")
        self.assertLessEqual(int(summary.loc["Suburb", "encoded_levels"]), 12)
        self.assertLessEqual(int(summary.loc["Suburb", "encoded_levels"]), 255)
        self.assertEqual(int(summary.loc["Suburb", "kept_levels"]), 10)

        later = _frame(
            [
                {"Suburb": "never-seen", "C3_Complaint_Type": "water"},
                {"Suburb": pd.NA, "C3_Complaint_Type": "water"},
                {"Suburb": "suburb-0", "C3_Complaint_Type": "once"},
            ]
        )
        transformed = encoder.transform(later)
        rare_code = encoder.code_maps_["Suburb"]["(rare)"]
        missing_code = encoder.code_maps_["Suburb"]["(missing)"]
        kept_code = encoder.code_maps_["Suburb"]["suburb-0"]
        self.assertEqual(transformed.loc[0, "Suburb"], rare_code)
        self.assertEqual(transformed.loc[1, "Suburb"], missing_code)
        self.assertEqual(transformed.loc[2, "Suburb"], kept_code)
        self.assertNotEqual(rare_code, missing_code)
        self.assertNotEqual(kept_code, rare_code)
        self.assertTrue(encoded["Suburb"].notna().all())

    def test_embedding_unseen_is_zero_and_rare_is_trained(self) -> None:
        rows = [{"C3_Complaint_Type": "water", "Suburb": "alpha"} for _ in range(6)]
        rows.append({"C3_Complaint_Type": "once", "Suburb": pd.NA})
        frame = _frame(rows)
        encoder = EmbeddingCategoryEncoder(min_count=2)
        trained = encoder.fit(frame).transform(frame)
        self.assertTrue(np.all(trained["codes"][:, 0] > 0))
        later = _frame(
            [
                {"C3_Complaint_Type": "brand-new", "Suburb": "alpha"},
                {"C3_Complaint_Type": "once", "Suburb": pd.NA},
                {"C3_Complaint_Type": "water", "Suburb": "alpha"},
            ]
        )
        transformed = encoder.transform(later)
        complaint = CATEGORICAL_FEATURES.index("C3_Complaint_Type")
        suburb = CATEGORICAL_FEATURES.index("Suburb")
        self.assertEqual(int(transformed["codes"][0, complaint]), 0)
        self.assertGreater(int(transformed["codes"][1, complaint]), 0)
        self.assertGreater(int(transformed["codes"][1, suburb]), 0)
        self.assertGreater(int(transformed["codes"][2, complaint]), 0)
        self.assertNotEqual(
            int(transformed["codes"][1, complaint]),
            int(transformed["codes"][2, complaint]),
        )

    def test_leakage_columns_are_rejected(self) -> None:
        frame = _frame([{"C3_Complaint_Type": "water"}])
        frame["Completed_Date"] = pd.Timestamp("2024-01-02")
        with self.assertRaises(ValueError):
            TreeCategoryEncoder().fit(frame)
        with self.assertRaises(ValueError):
            EmbeddingCategoryEncoder().fit(frame)

    def test_locked_test_cannot_be_scored(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_split_may_be_scored("test")
        with self.assertRaises(RuntimeError):
            summarise_predictions(
                list(TARGET_LABELS),
                list(TARGET_LABELS),
                np.eye(3),
                model="logistic_regression",
                split="test",
                approval=APPROVED_APPROVAL,
            )


class ChampionAndConfigTest(unittest.TestCase):
    def test_pending_model_cannot_be_champion(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "model": "logistic_regression",
                    "split": "validation",
                    "approval": APPROVED_APPROVAL,
                    "macro_f1": 0.40,
                    "recall_unresolved": 0.20,
                    "recall_resolved_slow": 0.30,
                },
                {
                    "model": "multilayer_perceptron",
                    "split": "validation",
                    "approval": APPROVED_APPROVAL,
                    "macro_f1": 0.50,
                    "recall_unresolved": 0.25,
                    "recall_resolved_slow": 0.22,
                },
                {
                    "model": "embedding_network",
                    "split": "validation",
                    "approval": PENDING_APPROVAL,
                    "macro_f1": 0.99,
                    "recall_unresolved": 0.99,
                    "recall_resolved_slow": 0.99,
                },
                {
                    "model": "multilayer_perceptron",
                    "split": "train",
                    "approval": APPROVED_APPROVAL,
                    "macro_f1": 0.80,
                    "recall_unresolved": 0.70,
                    "recall_resolved_slow": 0.70,
                },
            ]
        )
        chosen = provisional_champion(metrics)
        self.assertEqual(chosen["model"], "multilayer_perceptron")
        self.assertAlmostEqual(float(chosen["macro_f1"]), 0.50)

    def test_config_uses_class_weights_and_locks_the_test(self) -> None:
        config = base_experiment_config()
        self.assertFalse(config["imbalance"]["smote"])
        self.assertFalse(config["locked_test_scored_in_phase_5"])
        self.assertEqual(config["logistic_regression"]["class_weight"], "balanced")
        self.assertEqual(config["embedding_network"]["approval"], APPROVED_APPROVAL)
        weights = balanced_class_weights(
            ["Resolved Fast", "Resolved Fast", "Resolved Slow", "Unresolved"]
        )
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(weights[0], 4 / (3 * 2))
        self.assertAlmostEqual(weights[2], 4 / (3 * 1))

    def test_report_names_the_approved_champion(self) -> None:
        split_summary = pd.DataFrame(
            [
                {
                    "split": "train",
                    "rows": 30,
                    "start_date": "2023-01-01",
                    "end_date": "2024-12-31",
                    "Resolved Fast percent": 50.0,
                    "Resolved Slow percent": 30.0,
                    "Unresolved percent": 20.0,
                },
                {
                    "split": "validation",
                    "rows": 12,
                    "start_date": "2025-01-01",
                    "end_date": "2025-12-31",
                    "Resolved Fast percent": 40.0,
                    "Resolved Slow percent": 30.0,
                    "Unresolved percent": 30.0,
                },
                {
                    "split": "test",
                    "rows": 9,
                    "start_date": "2026-01-01",
                    "end_date": "2026-04-30",
                    "Resolved Fast percent": 34.0,
                    "Resolved Slow percent": 33.0,
                    "Unresolved percent": 33.0,
                },
            ]
        )
        metrics = pd.DataFrame(
            [
                _metric_row("logistic_regression", "train", APPROVED_APPROVAL, 0.61),
                _metric_row("logistic_regression", "validation", APPROVED_APPROVAL, 0.42),
                _metric_row("hist_gradient_boosting", "train", APPROVED_APPROVAL, 0.70),
                _metric_row("hist_gradient_boosting", "validation", APPROVED_APPROVAL, 0.55),
                _metric_row("multilayer_perceptron", "train", APPROVED_APPROVAL, 0.66),
                _metric_row("multilayer_perceptron", "validation", APPROVED_APPROVAL, 0.48),
                _metric_row("embedding_network", "train", PENDING_APPROVAL, 0.90),
                _metric_row("embedding_network", "validation", PENDING_APPROVAL, 0.80),
            ]
        )
        report = render_phase5_report(
            split_summary,
            {"rows": 30, "columns": 12, "nonzero": 90, "dense_float32_gib": 0.01},
            pd.DataFrame(
                [
                    {
                        "column": "Suburb",
                        "kept_levels": 2,
                        "distinct_training_levels": 4,
                        "encoded_levels": 4,
                    }
                ]
            ),
            metrics,
            {
                "logistic_regression": {
                    "max_iter": 200,
                    "tol": 1e-3,
                    "fit_seconds": 1.0,
                    "convergence_warning": True,
                },
                "hist_gradient_boosting": {
                    "fit_seconds": 2.0,
                    "n_iter_trained": 10,
                    "best_iteration": 8,
                    "validation_macro_f1_at_best": 0.55,
                    "hit_max_iter": False,
                },
                "multilayer_perceptron": {
                    "hidden_sizes": [8],
                    "dropout": 0.3,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "batch_size": 16,
                    "n_parameters": 20,
                    "n_features": 12,
                    "epochs_ran": 3,
                    "best_epoch": 2,
                    "stopped_early": True,
                    "fit_seconds": 3.0,
                },
                "embedding_network": {
                    "embedding_dim": 4,
                    "hidden_size": 8,
                    "dropout": 0.3,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "batch_size": 16,
                    "epochs_ran": 2,
                    "best_epoch": 2,
                    "fit_seconds": 2.5,
                },
            },
        )
        self.assertIn("provisional champion is hist_gradient_boosting", report)
        self.assertIn("pending lecturer approval", report)
        self.assertNotIn("provisional champion is embedding_network", report)
        self.assertIn("locked test", report.lower())

    def test_approved_embedding_wins_only_on_validation_macro_f1(self) -> None:
        from src.embedding_experiment import champion_decision_from_metrics

        recorded = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "05_split_metrics.csv")
        decision = champion_decision_from_metrics(recorded, 0.4765377211093145)
        self.assertEqual(decision["champion_model"], "hist_gradient_boosting")
        self.assertFalse(decision["champion_changed"])
        self.assertFalse(decision["locked_test_used"])
        self.assertIsNone(decision["approval_date"])
        self.assertGreater(
            decision["champion_validation_macro_f1"],
            decision["embedding_validation_macro_f1"],
        )
        self.assertGreater(
            decision["embedding_validation_macro_f1"],
            decision["frozen_boosting_validation_macro_f1"],
        )
        higher = recorded.copy()
        mask = higher["model"].eq("embedding_network") & higher["split"].eq("validation")
        higher.loc[mask, "macro_f1"] = 0.99
        higher.loc[mask, "recall_unresolved"] = 0.99
        higher.loc[mask, "recall_resolved_slow"] = 0.99
        promoted = champion_decision_from_metrics(higher, 0.4765377211093145)
        self.assertEqual(promoted["champion_model"], "embedding_network")
        self.assertTrue(promoted["champion_changed"])


def _metric_row(model: str, split: str, approval: str, macro_f1: float) -> dict:
    return {
        "model": model,
        "split": split,
        "approval": approval,
        "accuracy": macro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": macro_f1,
        "recall_resolved_fast": macro_f1,
        "recall_resolved_slow": macro_f1 / 2,
        "recall_unresolved": macro_f1 / 3,
        "macro_roc_auc": 0.7,
        "macro_average_precision": 0.6,
        "fit_seconds": 1.0,
    }


class ModelFitTest(unittest.TestCase):
    def test_boosting_uses_the_validation_split_and_three_class_probabilities(self) -> None:
        rows = []
        complaints = ["water", "power", "sewer"]
        for index in range(90):
            rows.append(
                {
                    "C3_Complaint_Type": complaints[index % 3],
                    "Suburb": "north" if index % 2 == 0 else "south",
                    "created_is_weekend": float(index % 5 == 0),
                }
            )
        frame = _frame(rows)
        labels = np.array([complaints[index % 3] for index in range(90)], dtype=object)
        label_map = {"water": TARGET_LABELS[0], "power": TARGET_LABELS[1], "sewer": TARGET_LABELS[2]}
        y = np.array([label_map[value] for value in labels], dtype=object)
        encoder = TreeCategoryEncoder(min_count=1, max_kept=10)
        encoded = encoder.fit(frame.iloc[:60]).transform(frame)
        model = make_hist_gradient_boosting(
            max_iter=6,
            min_samples_leaf=2,
            max_leaf_nodes=4,
            n_iter_no_change=2,
            learning_rate=0.2,
        )
        model.fit(encoded.iloc[:60], y[:60], X_val=encoded.iloc[60:], y_val=y[60:])
        self.assertEqual(model.is_categorical_.sum(), len(CATEGORICAL_FEATURES))
        self.assertFalse(bool(model.is_categorical_[list(MODEL_FEATURES).index(NUMERIC_FEATURES[0])]))
        self.assertGreater(len(model.validation_score_), 1)
        self.assertTrue(np.isfinite(model.validation_score_).all())
        best = use_best_validation_iteration(model)
        self.assertGreaterEqual(best, 0)
        probabilities = model.predict_proba(encoded.iloc[60:])
        self.assertEqual(probabilities.shape, (30, 3))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5))

    def test_mlp_outputs_three_probabilities_and_stops_within_the_epoch_cap(self) -> None:
        generator = np.random.default_rng(0)
        features = generator.normal(size=(48, 6))
        labels = _labels(48)
        result = train_feedforward(
            features[:36],
            labels[:36],
            features[36:],
            labels[36:],
            parameters={
                "hidden_sizes": (8,),
                "dropout": 0.0,
                "batch_size": 12,
                "max_epochs": 3,
                "patience": 2,
                "learning_rate": 1e-2,
            },
        )
        self.assertEqual(result["approval"], APPROVED_APPROVAL)
        self.assertLessEqual(result["epochs_ran"], 3)
        self.assertGreaterEqual(result["best_epoch"], 1)
        self.assertIn("validation_macro_f1", result["history"].columns)
        repeated = train_feedforward(
            features[:36],
            labels[:36],
            features[36:],
            labels[36:],
            parameters={
                "hidden_sizes": (8,),
                "dropout": 0.0,
                "batch_size": 12,
                "max_epochs": 3,
                "patience": 2,
                "learning_rate": 1e-2,
            },
        )
        from src.neural import predict_feedforward

        y_index = np.arange(12) % 3
        first, proba, _loss = predict_feedforward(
            result["model"], features[36:], y_index, batch_size=12
        )
        second, _proba, _loss = predict_feedforward(
            repeated["model"], features[36:], y_index, batch_size=12
        )
        self.assertEqual(proba.shape, (12, 3))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0, atol=1e-5))
        self.assertTrue(np.array_equal(first, second))
        module = FeedForwardNet(4, hidden_sizes=(5,), dropout=0.1)
        logits = module(torch_tensor(np.zeros((2, 4), dtype=np.float32)))
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_pending_embedding_network_keeps_a_zero_unseen_vector(self) -> None:
        rows = []
        for index in range(30):
            rows.append(
                {
                    "C3_Complaint_Type": "water" if index % 2 == 0 else "power",
                    "Suburb": "north" if index % 3 else "south",
                    "created_is_weekend": float(index % 2),
                }
            )
        frame = _frame(rows)
        labels = _labels(30)
        encoder = EmbeddingCategoryEncoder(min_count=1)
        encoder.fit(frame.iloc[:20])
        train = encoder.transform(frame.iloc[:20])
        validation = encoder.transform(frame.iloc[20:])
        unseen = encoder.transform(_frame([{"C3_Complaint_Type": "brand-new"}]))
        complaint = CATEGORICAL_FEATURES.index("C3_Complaint_Type")
        self.assertEqual(int(unseen["codes"][0, complaint]), 0)
        result = train_embedding_network(
            train["codes"],
            train["numeric"],
            labels[:20],
            validation["codes"],
            validation["numeric"],
            labels[20:],
            encoder.cardinalities_,
            parameters={
                "embedding_dim": 4,
                "hidden_size": 8,
                "dropout": 0.0,
                "batch_size": 10,
                "max_epochs": 2,
                "patience": 2,
                "learning_rate": 1e-2,
            },
        )
        self.assertEqual(result["approval"], APPROVED_APPROVAL)
        self.assertIn("No approval date was supplied", result["approval_note"])
        unseen_vector = result["model"].embeddings[complaint].weight.detach().numpy()[0]
        self.assertTrue(np.allclose(unseen_vector, 0.0))
        module = EmbeddingNet([3, 3], embedding_dim=4, hidden_size=6)
        import torch

        logits = module(torch.zeros((2, 2), dtype=torch.long), torch.zeros((2, 1)))
        self.assertEqual(tuple(logits.shape), (2, 3))


def torch_tensor(array):
    import torch

    return torch.from_numpy(array)


if __name__ == "__main__":
    unittest.main()
