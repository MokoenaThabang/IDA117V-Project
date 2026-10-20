"""Guards for the embedding reproduction path. These tests do not fit a model."""

import pandas as pd

from src import config
from src.embedding_experiment import artifact_directory, metric_differences


def test_existing_checkpoint_is_not_the_write_target() -> None:
    side = artifact_directory(checkpoint_exists=True, overwrite_checkpoint=False)
    official = artifact_directory(checkpoint_exists=False, overwrite_checkpoint=False)
    forced = artifact_directory(checkpoint_exists=True, overwrite_checkpoint=True)
    assert side.name == "embedding_retrain"
    assert official == config.PROJECT_ROOT
    assert forced == official
    assert side != official


def test_metric_differences_report_the_absolute_gap() -> None:
    saved = pd.DataFrame(
        [
            {"split": "validation", "macro_f1": 0.480687, "accuracy": 0.592},
            {"split": "train", "macro_f1": 0.586, "accuracy": 0.691},
        ]
    )
    reproduced = saved.copy()
    reproduced.loc[reproduced["split"].eq("validation"), "macro_f1"] = 0.480688
    gaps = metric_differences(reproduced, saved, ["macro_f1", "accuracy"])
    assert gaps["validation"]["macro_f1"] == abs(0.480688 - 0.480687)
    assert gaps["train"]["accuracy"] == 0.0
