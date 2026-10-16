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


def save_duration_histogram(durations, threshold_days: int, clip_days: int = 90):
    """Save a histogram of completed durations.

    Durations above ``clip_days`` stay in the data and are left off the
    chart so the body of the distribution stays readable. No location
    fields are drawn.
    """
    import matplotlib.pyplot as plt

    use_project_style()
    values = durations.dropna()
    shown = values[values <= clip_days]
    figure, axis = plt.subplots()
    axis.hist(shown, bins=clip_days + 1, color="#2c6e8a")
    axis.axvline(threshold_days, color="#b85c38", linewidth=1.5)
    axis.set_xlabel("Resolution duration (calendar days)")
    axis.set_ylabel("Completed requests")
    axis.set_title(f"Completed durations up to {clip_days} days")
    hidden = int((values > clip_days).sum())
    axis.text(
        0.98,
        0.95,
        f"Hidden above {clip_days} days: {hidden}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    return figure, save_figure(figure, "01_resolution_duration.png")


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
