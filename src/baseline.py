#!/usr/bin/env python3
"""Naïve DummyClassifier on the frozen main split. Split before fit. No test tuning."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.prep import FEATURES, SEED, build_preprocessor

PROC = Path("data/processed")
MODELS = Path("models")
EXPERIMENTS = Path("experiments")
CLASSES = ["Resolved Fast", "Resolved Slow", "Unresolved"]


def _load(name: str) -> pd.DataFrame:
    path = PROC / name
    if not path.exists():
        raise SystemExit(f"Missing {path}. See data/README.md.")
    return pd.read_parquet(path)


def main() -> None:
    train, val, test = (_load(n) for n in ("train.parquet", "val.parquet", "test.parquet"))
    MODELS.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for strategy in ("most_frequent", "stratified"):
        pipe = Pipeline(
            [
                ("prep", build_preprocessor()),
                ("model", DummyClassifier(strategy=strategy, random_state=SEED)),
            ]
        )
        t0 = time.time()
        pipe.fit(train[FEATURES], train["target"].astype(str))
        fit_s = time.time() - t0
        row = {
            "run_id": f"dummy_{strategy}",
            "model": f"DummyClassifier({strategy})",
            "seed": SEED,
            "split": "main stratified 70/15/15",
            "runtime_sec_fit": round(fit_s, 2),
            "notes": "Naïve baseline. Not one of the four substantive models. No test tuning.",
        }
        for split_name, df in ("train", train), ("val", val), ("test", test):
            pred = pipe.predict(df[FEATURES])
            y = df["target"].astype(str)
            row[f"accuracy_{split_name}"] = float(accuracy_score(y, pred))
            row[f"macro_f1_{split_name}"] = float(f1_score(y, pred, average="macro"))
            report = classification_report(
                y, pred, labels=CLASSES, zero_division=0, output_dict=True
            )
            row[f"recall_slow_{split_name}"] = float(report["Resolved Slow"]["recall"])
            row[f"recall_unresolved_{split_name}"] = float(report["Unresolved"]["recall"])
            if split_name == "val":
                print(
                    classification_report(y, pred, labels=CLASSES, zero_division=0),
                    flush=True,
                )
        if strategy == "most_frequent":
            joblib.dump(pipe, MODELS / "dummy_most_frequent.joblib")
        rows.append(row)
        print(
            row["model"],
            "val_macro_f1",
            round(row["macro_f1_val"], 4),
            "test_macro_f1",
            round(row["macro_f1_test"], 4),
            "fit_s",
            row["runtime_sec_fit"],
            flush=True,
        )
    pd.DataFrame(rows).to_csv(EXPERIMENTS / "experiment_log.csv", index=False)
    print("wrote", EXPERIMENTS / "experiment_log.csv")


if __name__ == "__main__":
    main()
