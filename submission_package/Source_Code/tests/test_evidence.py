"""Checks for the submission-evidence helpers. They do not fit the full dataset."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.geographic_error_audit import MIN_GROUP_SUPPORT, _audit
from src.per_class_curves import _curve_table
from src.preprocessing import _reject_blocked_features
from src.training import TreeCategoryEncoder


def test_work_centre_stays_blocked_unless_the_sensitivity_flag_is_set():
    with pytest.raises(ValueError, match="Work Center"):
        _reject_blocked_features(["Work_Center"])
    _reject_blocked_features(["Work_Center"], allow_work_centre=True)


def test_work_centre_encoder_requires_the_flag():
    frame = pd.DataFrame(
        {
            "C3_Complaint_Type": ["Water"] * 20,
            "Suburb": ["PHILIPPI"] * 20,
            "Ward": ["000"] * 20,
            "Sub_Council": ["Sub-Council 16"] * 20,
            "created_day_of_week": ["Wednesday"] * 20,
            "created_month": ["June"] * 20,
            "created_is_weekend": [0] * 20,
            "Work_Center": ["Depot A"] * 20,
        }
    )
    with pytest.raises(ValueError, match="Work Center"):
        TreeCategoryEncoder(
            categorical_columns=["C3_Complaint_Type", "Work_Center"],
            numeric_columns=["created_is_weekend"],
        ).fit(frame)
    encoder = TreeCategoryEncoder(
        categorical_columns=["C3_Complaint_Type", "Work_Center"],
        numeric_columns=["created_is_weekend"],
        allow_work_centre=True,
        min_count=1,
    )
    encoder.fit(frame)
    coded = encoder.transform(frame)
    assert "Work_Center" in coded.columns
    assert len(coded) == 20


def test_per_class_curves_use_probabilities():
    labels = np.array(config.TARGET_LABELS * 10)
    probabilities = np.tile(np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]), (10, 1))
    table, curves = _curve_table(labels, probabilities)
    assert list(table["label"]) == list(config.TARGET_LABELS)
    assert table["curve_defined"].all()
    assert len(curves) == 3
    assert set(curves[0]) >= {"fpr", "tpr", "precision", "recall", "roc_auc"}


def test_small_geographic_groups_do_not_get_class_metrics():
    frame = pd.DataFrame(
        {
            "suburb": ["A"] * 5 + ["B"] * (MIN_GROUP_SUPPORT + 40),
            "y_true": (["Resolved Fast"] * 3 + ["Resolved Slow"] * 2)
            + (["Resolved Fast"] * (MIN_GROUP_SUPPORT + 10) + ["Resolved Slow"] * 30),
            "predicted": (["Resolved Slow"] * 5)
            + (["Resolved Fast"] * (MIN_GROUP_SUPPORT + 10) + ["Resolved Slow"] * 30),
        }
    )
    summary, pieces, confusion = _audit(frame, "suburb")
    small = summary.loc[summary["suburb"].eq("A")].iloc[0]
    assert bool(small["low_support"]) is True
    small_metrics = pieces.loc[pieces["suburb"].eq("A")]
    assert small_metrics["class_metrics_reported"].eq(False).all()
    assert small_metrics["precision"].isna().all()
    large = pieces.loc[pieces["suburb"].eq("B") & pieces["label"].eq("Resolved Slow")].iloc[0]
    assert bool(large["class_metrics_reported"]) is True
    assert "A" not in set(confusion["suburb"])
