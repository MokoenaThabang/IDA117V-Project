"""Save figures into results/figures.

Plotting functions are not called during Phase 1. The audit notebook does
not draw charts until a raw extract is loaded in a later run.
"""

from __future__ import annotations

from pathlib import Path

from src.config import FIGURES_DIR


def use_project_style() -> None:
    """Apply a readable Matplotlib and seaborn style for later notebooks."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (8, 4.5)
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titlesize"] = 12


def save_figure(figure, filename: str, directory: Path | None = None) -> Path:
    """Save a Matplotlib figure as a PNG and return the path.

    ``filename`` may omit the ``.png`` suffix. The figure is not closed,
    so the caller can still display it in a notebook.
    """
    output_dir = FIGURES_DIR if directory is None else directory
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = filename if filename.lower().endswith(".png") else f"{filename}.png"
    path = output_dir / safe_name
    figure.savefig(path, dpi=150)
    return path
