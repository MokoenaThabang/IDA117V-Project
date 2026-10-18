"""Shared metrics and plots. Never used to pick models from test."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

CLASSES = ["Resolved Fast", "Resolved Slow", "Unresolved"]


def scores(y_true, y_pred, y_proba=None) -> dict:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    for cls, pp, rr, ff in zip(CLASSES, p, r, f):
        key = cls.replace(" ", "_").lower()
        out[f"precision_{key}"] = float(pp)
        out[f"recall_{key}"] = float(rr)
        out[f"f1_{key}"] = float(ff)
    if y_proba is not None:
        y_bin = np.stack([(y_true == c).astype(int) for c in CLASSES], axis=1)
        try:
            out["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
            )
        except ValueError:
            out["roc_auc_ovr_macro"] = float("nan")
        try:
            out["pr_auc_macro"] = float(
                average_precision_score(y_bin, y_proba, average="macro")
            )
        except ValueError:
            out["pr_auc_macro"] = float("nan")
    return out


def report_text(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, labels=CLASSES, zero_division=0, digits=4
    )


def plot_confusion(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASSES, rotation=20, ha="right")
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc_together(y_true, proba_by_model: dict, path: Path) -> None:
    y_true = np.asarray(y_true).astype(str)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in proba_by_model.items():
        mean_tpr = []
        fpr_grid = np.linspace(0, 1, 100)
        tprs = []
        for i, cls in enumerate(CLASSES):
            yt = (y_true == cls).astype(int)
            fpr, tpr, _ = roc_curve(yt, proba[:, i])
            tprs.append(np.interp(fpr_grid, fpr, tpr))
        ax.plot(fpr_grid, np.mean(tprs, axis=0), label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC (macro average of class curves) — validation")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_comparison_bar(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(df["model"], df["macro_f1_val"])
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Validation macro-F1 (champion chosen here, not on test)")
    ax.set_ylim(0, max(0.5, float(df["macro_f1_val"].max()) + 0.08))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
