"""Paired error comparison for the frozen champion and one comparator.

McNemar's test uses the discordant cases only. The exact binomial test is
used when there are fewer than 25 discordant cases. Otherwise the
continuity-corrected chi-square approximation is used. SciPy supplies both
calculations. The test does not score a model by itself and it does not
read a validation table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EXACT_BINOMIAL_MAX_DISCORDANT = 25
SIGNIFICANCE_ALPHA = 0.05


def paired_correctness_counts(y_true, champion_pred, comparator_pred) -> dict[str, int]:
    """Count the four paired outcomes on the same cases."""
    actual = np.asarray(y_true)
    champion = np.asarray(champion_pred)
    comparator = np.asarray(comparator_pred)
    if not (len(actual) == len(champion) == len(comparator)):
        raise ValueError(
            "Paired predictions must have the same number of rows. "
            f"Got {len(actual)}, {len(champion)}, {len(comparator)}."
        )
    if len(actual) == 0:
        raise ValueError("Paired predictions are empty.")
    champion_correct = champion == actual
    comparator_correct = comparator == actual
    return {
        "rows": int(len(actual)),
        "both_correct": int(np.sum(champion_correct & comparator_correct)),
        "champion_only": int(np.sum(champion_correct & ~comparator_correct)),
        "comparator_only": int(np.sum(~champion_correct & comparator_correct)),
        "both_wrong": int(np.sum(~champion_correct & ~comparator_correct)),
    }


def paired_correctness_table(counts: dict[str, int]) -> pd.DataFrame:
    """One-row table of the paired outcomes."""
    return pd.DataFrame([counts])


def mcnemar_test(counts: dict[str, int], alpha: float = SIGNIFICANCE_ALPHA) -> dict:
    """Test whether the two models have the same paired error rate.

    ``champion_only`` and ``comparator_only`` are the discordant cells.
    """
    champion_only = int(counts["champion_only"])
    comparator_only = int(counts["comparator_only"])
    discordant = champion_only + comparator_only
    if discordant < 0:
        raise ValueError("Discordant counts cannot be negative.")
    if discordant == 0:
        statistic = 0.0
        p_value = 1.0
        method = "no discordant pairs"
    elif discordant < EXACT_BINOMIAL_MAX_DISCORDANT:
        result = stats.binomtest(champion_only, discordant, 0.5, alternative="two-sided")
        statistic = float(champion_only)
        p_value = float(result.pvalue)
        method = "exact binomial"
    else:
        statistic = (abs(champion_only - comparator_only) - 1) ** 2 / discordant
        p_value = float(stats.chi2.sf(statistic, df=1))
        method = "chi-square with continuity correction"
    significant = bool(p_value < float(alpha))
    if discordant == 0:
        conclusion = (
            "There are no discordant cases, so the test does not show a difference "
            "in paired error rates."
        )
    elif significant:
        conclusion = (
            "The paired error rates differ at the 0.05 threshold. "
            "This does not show that one model is better on every class."
        )
    else:
        conclusion = (
            "The test does not show a difference in paired error rates at the 0.05 threshold. "
            "This does not show that the models are equal on every class."
        )
    return {
        "discordant": int(discordant),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "method": method,
        "significant": significant,
        "conclusion": conclusion,
    }
