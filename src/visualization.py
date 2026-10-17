"""Save report figures into results/figures.

The exploratory plots take aggregate tables, not raw coordinates. They do
not draw maps and they do not print notification identifiers.
"""

from __future__ import annotations

from pathlib import Path

from src.config import FIGURES_DIR, TARGET_LABELS

# Okabe-Ito colours that remain distinct in print and for colour-blind readers.
OUTCOME_COLOURS = {
    "Resolved Fast": "#0072B2",
    "Resolved Slow": "#E69F00",
    "Unresolved": "#CC79A7",
}


def use_project_style() -> None:
    """Apply a readable Matplotlib and seaborn style for report figures."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (8, 4.5)
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["font.size"] = 10


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


def save_figure(figure, filename: str, directory: Path | None = None, dpi: int = 200) -> Path:
    """Save a Matplotlib figure as a high-resolution PNG and return the path.

    ``filename`` may omit the ``.png`` suffix. The figure is not closed,
    so the caller can still display it in a notebook.
    """
    output_dir = FIGURES_DIR if directory is None else directory
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = filename if filename.lower().endswith(".png") else f"{filename}.png"
    path = output_dir / safe_name
    figure.savefig(path, dpi=dpi)
    return path


def _finish(figure, filename: str, directory: Path | None, apply_layout: bool = True):
    import matplotlib.pyplot as plt

    if apply_layout:
        figure.tight_layout()
    path = save_figure(figure, filename, directory=directory)
    plt.close(figure)
    return figure, path


def plot_target_distribution(table, filename: str = "03_01_target_distribution.png", directory: Path | None = None):
    """Bar chart of the three resolution classes, with counts and percentages."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    use_project_style()
    plotted = table.loc[table["outcome"].isin(TARGET_LABELS)].copy()
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(
        data=plotted,
        x="outcome",
        y="requests",
        hue="outcome",
        palette=OUTCOME_COLOURS,
        legend=False,
        ax=axis,
        order=list(TARGET_LABELS),
    )
    labelled = int(plotted["requests"].sum())
    for patch, outcome in zip(axis.patches, TARGET_LABELS, strict=True):
        row = plotted.loc[plotted["outcome"] == outcome].iloc[0]
        count = int(row["requests"])
        percent = float(row["percent_of_labelled"])
        axis.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            f"{count:,}\n({percent:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_xlabel("Resolution outcome")
    axis.set_ylabel("Labelled requests")
    axis.set_title("Resolution outcome of labelled service requests")
    axis.set_ylim(0, plotted["requests"].max() * 1.22)
    axis.text(
        0.0,
        -0.2,
        f"Labelled requests: {labelled:,}. Requests with no label are omitted from this chart.",
        transform=axis.transAxes,
        fontsize=8,
        color="#333333",
    )
    return _finish(figure, filename, directory)


def plot_missingness_and_completion(
    missing,
    completion,
    filename: str = "03_02_missingness_and_quality.png",
    directory: Path | None = None,
):
    """Predictor missingness beside the meaning of a blank completion date."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    use_project_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    sns.barplot(
        data=missing,
        y="display_name",
        x="missing_percent",
        color="#0072B2",
        ax=axes[0],
        orient="h",
    )
    axes[0].set_xlabel("Missing values (% of analysis rows)")
    axes[0].set_ylabel("")
    axes[0].set_title("Genuine missing values by column")
    axes[0].set_xlim(0, max(8, float(missing["missing_percent"].max()) * 1.25))
    for index, row in missing.reset_index(drop=True).iterrows():
        axes[0].text(
            row["missing_percent"] + 0.1,
            index,
            f"{int(row['missing_count']):,}",
            va="center",
            fontsize=8,
        )

    completion_colours = {
        "completion_recorded": "#0072B2",
        "blank_unresolved": "#CC79A7",
        "blank_insufficient_follow_up": "#7F7F7F",
        "completion_before_creation": "#D55E00",
    }
    colours = [completion_colours.get(status, "#999999") for status in completion["status"]]
    axes[1].barh(completion["display_name"], completion["requests"], color=colours)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Requests")
    axes[1].set_ylabel("")
    axes[1].set_title("Completion date: recorded, blank, or invalid")
    largest = float(completion["requests"].max())
    axes[1].set_xlim(0, largest * 1.28)
    for index, row in completion.reset_index(drop=True).iterrows():
        axes[1].text(
            row["requests"] + largest * 0.02,
            index,
            f"{int(row['requests']):,}",
            va="center",
            fontsize=8,
        )
    figure.suptitle("Missingness and completion-date status", fontsize=13)
    return _finish(figure, filename, directory)


def plot_outcome_by_group(
    table,
    group_column: str,
    title: str,
    filename: str,
    directory: Path | None = None,
):
    """Normalised stacked bars for the largest groups, with group size labelled."""
    import matplotlib.pyplot as plt
    import numpy as np

    use_project_style()
    plotted = table.sort_values("requests", ascending=True)
    labels = plotted[group_column].astype(str).tolist()
    figure, axis = plt.subplots(figsize=(11.5, 7.2))
    left = np.zeros(len(plotted))
    for outcome in TARGET_LABELS:
        percent = plotted[f"{outcome} percent"].to_numpy()
        axis.barh(labels, percent, left=left, color=OUTCOME_COLOURS[outcome], label=outcome, height=0.72)
        left = left + percent
    for index, count in enumerate(plotted["requests"].tolist()):
        axis.text(101.2, index, f"n={int(count):,}", va="center", ha="left", fontsize=8)
    axis.set_xlim(0, 124)
    axis.set_xlabel("Share of labelled requests in the group (%)")
    axis.set_ylabel("")
    axis.set_title(title)
    axis.legend(
        title="Outcome",
        loc="upper center",
        bbox_to_anchor=(0.42, -0.14),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.2)
    return _finish(figure, filename, directory, apply_layout=False)


def plot_monthly_patterns(
    monthly,
    filename: str = "03_05_monthly_request_volume.png",
    directory: Path | None = None,
):
    """Monthly request volume and the labelled outcome mix."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd

    use_project_style()
    months = pd.to_datetime(monthly["month"])
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True)
    axes[0].bar(months, monthly["requests"], width=20, color="#0072B2", label="Requests created")
    gap_months = months.loc[monthly["gap"].to_numpy()]
    if len(gap_months):
        axes[0].scatter(
            gap_months,
            [0] * len(gap_months),
            marker="v",
            s=40,
            color="#D55E00",
            zorder=3,
            label="Month with no created requests",
        )
    axes[0].set_ylabel("Requests created")
    axes[0].set_title("Service requests created each month")
    axes[0].legend(loc="upper right")

    for outcome, column, colour in (
        ("Resolved Fast", "resolved_fast_percent", OUTCOME_COLOURS["Resolved Fast"]),
        ("Resolved Slow", "resolved_slow_percent", OUTCOME_COLOURS["Resolved Slow"]),
        ("Unresolved", "unresolved_percent", OUTCOME_COLOURS["Unresolved"]),
    ):
        axes[1].plot(months, monthly[column], color=colour, linewidth=1.8, label=outcome)
    axes[1].set_ylabel("Share of labelled requests (%)")
    axes[1].set_xlabel("Month the request was created")
    axes[1].set_title("Outcome mix among requests that received a label")
    axes[1].set_ylim(0, 105)
    axes[1].legend(title="Outcome", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    axes[1].set_xlim(pd.Timestamp("2022-12-15"), months.max() + pd.Timedelta(days=18))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    figure.autofmt_xdate(rotation=45, ha="right")
    figure.tight_layout()
    figure.subplots_adjust(right=0.78, hspace=0.42)
    return _finish(figure, filename, directory, apply_layout=False)


def plot_duration_bins(
    table,
    threshold_days: int,
    filename: str = "03_06_resolution_duration.png",
    directory: Path | None = None,
):
    """Completed-duration bins. Long durations stay in the last bin."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    use_project_style()
    figure, axis = plt.subplots(figsize=(10.5, 5.0))
    sns.barplot(data=table, y="bin", x="requests", color="#0072B2", ax=axis, orient="h")
    axis.set_xlabel("Completed requests")
    axis.set_ylabel("Resolution duration")
    axis.set_title(f"Completed resolution time (fast/slow threshold = {threshold_days} days)")
    largest = float(table["requests"].max())
    axis.set_xlim(0, largest * 1.22)
    for index, row in table.reset_index(drop=True).iterrows():
        axis.text(
            row["requests"] + largest * 0.015,
            index,
            f"{int(row['requests']):,} ({row['percent_of_completed']:.1f}%)",
            va="center",
            fontsize=8,
        )
    return _finish(figure, filename, directory)


def plot_confusion_matrix(matrix, labels, title: str, filename: str, directory: Path | None = None):
    """Save a count confusion matrix. Rows are the actual class."""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    use_project_style()
    counts = np.asarray(matrix, dtype=int)
    annotations = np.array([[f"{int(value):,}" for value in row] for row in counts])
    figure, axis = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    sns.heatmap(
        counts,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=list(labels),
        yticklabels=list(labels),
        cbar_kws={"label": "Requests"},
        ax=axis,
    )
    axis.set_xlabel("Predicted outcome")
    axis.set_ylabel("Actual outcome", labelpad=16)
    axis.set_title(title)
    axis.tick_params(axis="y", pad=4)
    return _finish(figure, filename, directory, apply_layout=False)


def plot_learning_curves(history, title: str, filename: str, directory: Path | None = None):
    """Plot training and validation loss and macro-F1 against epoch."""
    import matplotlib.pyplot as plt

    use_project_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    epochs = history["epoch"]
    axes[0].plot(epochs, history["train_loss"], color="#0072B2", marker="o", label="Training")
    axes[0].plot(epochs, history["validation_loss"], color="#D55E00", marker="o", label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean weighted cross-entropy")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_macro_f1"], color="#0072B2", marker="o", label="Training")
    axes[1].plot(epochs, history["validation_macro_f1"], color="#D55E00", marker="o", label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Macro-F1")
    axes[1].legend()
    figure.suptitle(title)
    return _finish(figure, filename, directory, apply_layout=False)


def plot_boosting_curve(curve, chosen_iteration: int, title: str, filename: str, directory: Path | None = None):
    """Plot validation macro-F1 at each boosting iteration and mark the selected one."""
    import matplotlib.pyplot as plt

    use_project_style()
    figure, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    axis.plot(curve["iteration"], curve["validation_macro_f1"], color="#0072B2", marker="o")
    axis.axvline(chosen_iteration, color="#D55E00", linestyle="--", label=f"Selected iteration {chosen_iteration}")
    axis.set_xlabel("Boosting iteration")
    axis.set_ylabel("Validation macro-F1")
    axis.set_title(title)
    axis.legend()
    return _finish(figure, filename, directory, apply_layout=False)


def plot_validation_comparison(comparison, filename: str = "05_validation_comparison.png", directory: Path | None = None):
    """Grouped bars of validation macro-F1 and the two recall priorities."""
    import matplotlib.pyplot as plt
    import numpy as np

    use_project_style()
    metrics = [
        ("macro_f1", "Macro-F1"),
        ("recall_resolved_slow", "Resolved Slow recall"),
        ("recall_unresolved", "Unresolved recall"),
    ]
    models = []
    for row in comparison.itertuples(index=False):
        name = str(row.model)
        if getattr(row, "approval", "approved") != "approved":
            name = f"{name} (pending)"
        models.append(name)
    positions = np.arange(len(models))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10.4, 4.8), constrained_layout=True)
    colours = ["#0072B2", "#E69F00", "#CC79A7"]
    for offset, ((column, label), colour) in enumerate(zip(metrics, colours)):
        axis.bar(positions + (offset - 1) * width, comparison[column], width=width, color=colour, label=label)
    axis.set_xticks(positions)
    axis.set_xticklabels(models, rotation=15, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Validation score")
    axis.set_title("Validation comparison")
    axis.legend(loc="upper right")
    return _finish(figure, filename, directory, apply_layout=False)


def plot_train_validation_macro_f1(
    metrics,
    filename: str = "05_train_validation_macro_f1.png",
    directory: Path | None = None,
):
    """Compare full-period training and validation macro-F1 for each model."""
    import matplotlib.pyplot as plt
    import numpy as np

    use_project_style()
    labels = []
    train_scores = []
    validation_scores = []
    for model in list(dict.fromkeys(metrics["model"])):
        rows = metrics.loc[metrics["model"].eq(model)]
        approval = str(rows["approval"].iloc[0])
        labels.append(model if approval == "approved" else f"{model} (pending)")
        train_scores.append(float(rows.loc[rows["split"].eq("train"), "macro_f1"].iloc[0]))
        validation_scores.append(float(rows.loc[rows["split"].eq("validation"), "macro_f1"].iloc[0]))
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10.4, 4.8), constrained_layout=True)
    axis.bar(positions - width / 2, train_scores, width=width, color="#0072B2", label="Training")
    axis.bar(positions + width / 2, validation_scores, width=width, color="#D55E00", label="Validation")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=15, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Macro-F1")
    axis.set_title("Training and validation macro-F1")
    axis.legend()
    return _finish(figure, filename, directory, apply_layout=False)
