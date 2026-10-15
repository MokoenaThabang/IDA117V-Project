#!/usr/bin/env python3
"""Five required EDA figures from the MAIN training fold only (no val/test peek)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TRAIN = Path("data/processed/train.parquet")
FIG = Path("figures")
CLASSES = ["Resolved Fast", "Resolved Slow", "Unresolved"]
COLORS = ["#2ca02c", "#ff7f0e", "#d62728"]


def main() -> None:
    if not TRAIN.exists():
        raise SystemExit(
            "Missing data/processed/train.parquet. See data/README.md "
            "(Google Drive folder or copy from team-artifacts)."
        )
    FIG.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = pd.read_parquet(TRAIN)
    n = len(df)
    vc = df["target"].value_counts().reindex(CLASSES)

    fig, ax = plt.subplots(figsize=(8, 5))
    vc.plot(kind="bar", ax=ax, color=COLORS)
    ax.set_title(f"Target distribution on train (n={n:,}; seed 42 main split)")
    ax.set_xlabel("Resolution class")
    ax.set_ylabel("Number of service requests")
    fig.tight_layout()
    fig.savefig(FIG / "fig01_target_distribution.png", dpi=140)
    plt.close(fig)

    miss = pd.Series(
        {
            "Ward": 100 * df["Ward"].isna().mean(),
            "Sub_Council": 100 * df["Sub_Council"].isna().mean(),
            "suburb_group is OTHER": 100 * (df["suburb_group"].astype(str) == "OTHER").mean(),
            "suburb_group is #": 100 * (df["suburb_group"].astype(str) == "#").mean(),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    miss.plot(kind="barh", ax=ax, color="#4c78a8")
    ax.set_title("Train missingness and grouped-location shares (percent)")
    ax.set_xlabel("Percent of training rows")
    fig.tight_layout()
    fig.savefig(FIG / "fig02_missingness.png", dpi=140)
    plt.close(fig)

    ct = pd.crosstab(df["department"], df["target"], normalize="index").reindex(columns=CLASSES)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ct.plot(kind="barh", stacked=True, ax=ax, color=COLORS)
    ax.set_title("Resolution class mix by department (train)")
    ax.set_xlabel("Share of training requests")
    fig.tight_layout()
    fig.savefig(FIG / "fig03_department.png", dpi=140)
    plt.close(fig)

    top = df["C3_Complaint_Type"].value_counts().head(12).index
    sub = df[df["C3_Complaint_Type"].isin(top)]
    ct = pd.crosstab(sub["C3_Complaint_Type"], sub["target"], normalize="index").reindex(columns=CLASSES)
    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="barh", stacked=True, ax=ax, color=COLORS)
    ax.set_title("Resolution class mix for the 12 most common complaint types (train)")
    ax.set_xlabel("Share of training requests")
    fig.tight_layout()
    fig.savefig(FIG / "fig04_complaint_type.png", dpi=140)
    plt.close(fig)

    ts = pd.to_datetime(df["Created_On_Date"])
    df = df.assign(_month=ts.dt.to_period("M").astype(str))
    monthly = (
        df.groupby(["_month", "target"]).size().unstack(fill_value=0).reindex(columns=CLASSES)
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    monthly.plot(kind="area", stacked=True, ax=ax, color=COLORS, alpha=0.85)
    ax.set_title("Train tickets by created month and class (random split still spans 2023–2026)")
    ax.set_xlabel("Month created")
    ax.set_ylabel("Number of service requests")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_temporal_train.png", dpi=140)
    plt.close(fig)

    notes = Path("figures/interpretations.md")
    notes.write_text(
        f"""# EDA interpretations (train only, n={n:,})

## fig01_target_distribution.png

Fast {int(vc['Resolved Fast']):,} ({100*vc['Resolved Fast']/n:.1f}%),
Slow {int(vc['Resolved Slow']):,} ({100*vc['Resolved Slow']/n:.1f}%),
Unresolved {int(vc['Unresolved']):,} ({100*vc['Unresolved']/n:.1f}%).
Macro-F1 is required; a majority dummy will look strong on accuracy.

## fig02_missingness.png

Ward is missing on {miss['Ward']:.1f}% of train rows and Sub_Council on
{miss['Sub_Council']:.1f}%. The pipeline maps those to MISSING rather than
imputing the mode. Coordinates are unused. Raw suburb streets never appear.

## fig03_department.png

Department (from work centre) is known at logging and changes the Fast/Slow/Unresolved
mix. Public housing and open space run much higher Unresolved rates in the audit —
partly ticket-closing practice, not only delay. This is an EDA chart, not a second model.

## fig04_complaint_type.png

Complaint type is the operational code at logging. Volume is dominated by power,
sewer, street-light and bin faults. The dummy ignores this mix; later models must beat it.

## fig05_temporal_train.png

The main split is stratified random, so train still covers 1 Jan 2023 – 20 May 2026.
Time drift is checked on the separate time split, not by peeking at test.
""",
        encoding="utf-8",
    )
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
