#!/usr/bin/env python3
"""Q3 only: Dummy + LR + Random Forest + MLP + tabular embedding net.

Identical frozen main split (seed 42). No test look. No champion. No tuning.
Random Forest / MLP / embedding fit on a documented *train* subsample.
Official val stays the full frozen fold.
"""

from __future__ import annotations

import gc
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.metrics_util import CLASSES, scores
from src.neural import TabularEmbeddingEstimator, TorchMLPEstimator
from src.prep import FEATURES, SEED, build_preprocessor

PROC = Path("data/processed")
EXP = Path("experiments")
MEDIA = Path("/cursor/stores/self/media/q3")
REPO_FIG = Path("figures")

RF_TRAIN_N = 150_000
NN_TRAIN_N = 150_000
VAL_CHUNK = 16_384


def _media_dirs() -> list[Path]:
    dirs = []
    try:
        MEDIA.mkdir(parents=True, exist_ok=True)
        dirs.append(MEDIA)
    except OSError:
        pass
    REPO_FIG.mkdir(parents=True, exist_ok=True)
    dirs.append(REPO_FIG)
    return dirs


def _save(fig, name: str) -> None:
    for d in _media_dirs():
        fig.savefig(d / name, dpi=150)
    plt.close(fig)


def load_fold(name: str) -> pd.DataFrame:
    path = PROC / name
    if not path.exists():
        raise SystemExit(f"Missing {path}. See data/README.md.")
    return pd.read_parquet(path)


def subsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    sub, _ = train_test_split(
        df, train_size=n, stratify=df["target"], random_state=SEED
    )
    return sub.reset_index(drop=True)


def dense_chunk(prep, frame: pd.DataFrame, start: int, end: int) -> np.ndarray:
    Z = prep.transform(frame.iloc[start:end][FEATURES])
    if hasattr(Z, "toarray"):
        return Z.astype(np.float32).toarray()
    return np.asarray(Z, dtype=np.float32)


def batched_predict(predict_fn, prep, frame: pd.DataFrame, proba: bool = False):
    parts = []
    for i in range(0, len(frame), VAL_CHUNK):
        X = dense_chunk(prep, frame, i, i + VAL_CHUNK)
        parts.append(predict_fn(X))
        del X
    return np.concatenate(parts, axis=0)


def plot_history(history: list[dict], title: str, name: str) -> None:
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(df["epoch"], df["train_loss"], label="train loss")
    ax.plot(df["epoch"], df["val_loss"], label="internal-val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, name)


def plot_val_bar(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(df["model"], df["macro_f1_val"], color="#4c78a8")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Q3 validation macro-F1 (not a champion pick; Q4 compares fairly)")
    ax.set_ylim(0, max(0.55, float(df["macro_f1_val"].max()) + 0.08))
    ax.tick_params(axis="x", rotation=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig_q3_val_macro_f1.png")


def main() -> pd.DataFrame:
    EXP.mkdir(parents=True, exist_ok=True)
    compute = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": SEED,
        "rf_train_n": RF_TRAIN_N,
        "nn_train_n": NN_TRAIN_N,
        "note": (
            "Official val/test are frozen Drive folds. "
            "RF/MLP/embedding subsample train only. "
            "Neural early stopping uses 15% of that train subsample, not official val. "
            "Test is not loaded in Q3."
        ),
    }
    (EXP / "q3_compute.json").write_text(json.dumps(compute, indent=2) + "\n")
    print(json.dumps(compute, indent=2), flush=True)

    print("loading train/val (test stays locked)…", flush=True)
    train = load_fold("train.parquet")
    val = load_fold("val.parquet")
    y_train = train["target"].astype(str)
    y_val = val["target"].astype(str)

    log_rows = []
    val_table = []

    # --- 3.1 Dummy (ignores features; same frozen y) ---
    print("DummyClassifier…", flush=True)
    for strategy in ("most_frequent", "stratified"):
        t0 = time.time()
        dummy = DummyClassifier(strategy=strategy, random_state=SEED)
        dummy.fit(np.zeros((len(train), 1)), y_train)
        pred = dummy.predict(np.zeros((len(val), 1)))
        s = scores(y_val, pred)
        fit_s = time.time() - t0
        log_rows.append(
            {
                "run_id": f"dummy_{strategy}",
                "model": f"DummyClassifier({strategy})",
                "family": "naïve baseline",
                "seed": SEED,
                "train_rows_used": int(len(train)),
                "runtime_sec_fit": round(fit_s, 2),
                "notes": "Naïve baseline. Not one of the four substantive models. No test look in Q3.",
                "macro_f1_val": s["macro_f1"],
                "accuracy_val": s["accuracy"],
                "recall_resolved_slow_val": s["recall_resolved_slow"],
                "recall_unresolved_val": s["recall_unresolved"],
            }
        )
        val_table.append(
            {
                "model": f"Dummy ({strategy})",
                "family": "baseline",
                "macro_f1_val": s["macro_f1"],
                "accuracy_val": s["accuracy"],
                "recall_resolved_slow_val": s["recall_resolved_slow"],
                "recall_unresolved_val": s["recall_unresolved"],
            }
        )
        print(f"  {strategy} val macro-F1 {s['macro_f1']:.4f}", flush=True)

    print("fitting shared preprocessor on FULL train…", flush=True)
    prep = build_preprocessor()
    t0 = time.time()
    Xt_full = prep.fit_transform(train[FEATURES])
    Xv = prep.transform(val[FEATURES])
    print(f"  prep {time.time()-t0:.1f}s  train={Xt_full.shape} val={Xv.shape}", flush=True)

    rf_df = subsample(train, RF_TRAIN_N)
    nn_df = subsample(train, NN_TRAIN_N)

    # --- Logistic Regression (full train, sparse) ---
    print("Logistic Regression (full train, saga, sparse)…", flush=True)
    lr = LogisticRegression(
        solver="saga",
        class_weight="balanced",
        max_iter=80,
        C=1.0,
        random_state=SEED,
        tol=1e-3,
    )
    t0 = time.time()
    lr.fit(Xt_full, y_train)
    lr_s = time.time() - t0
    pred = lr.predict(Xv)
    proba = lr.predict_proba(Xv)
    s = scores(y_val, pred, proba)
    log_rows.append(
        {
            "run_id": "lr_balanced_C1",
            "model": "LogisticRegression",
            "family": "traditional",
            "seed": SEED,
            "train_rows_used": int(Xt_full.shape[0]),
            "runtime_sec_fit": round(lr_s, 2),
            "notes": "class_weight=balanced, saga, C=1, max_iter=80, tol=1e-3. Full train. No test.",
            **{f"{k}_val": v for k, v in s.items() if k in (
                "macro_f1", "accuracy", "recall_resolved_slow", "recall_unresolved",
                "roc_auc_ovr_macro", "pr_auc_macro",
            )},
        }
    )
    val_table.append(
        {
            "model": "LogisticRegression",
            "family": "traditional",
            "macro_f1_val": s["macro_f1"],
            "accuracy_val": s["accuracy"],
            "recall_resolved_slow_val": s["recall_resolved_slow"],
            "recall_unresolved_val": s["recall_unresolved"],
            "roc_auc_ovr_macro_val": s.get("roc_auc_ovr_macro"),
            "pr_auc_macro_val": s.get("pr_auc_macro"),
        }
    )
    print(f"  val macro-F1 {s['macro_f1']:.4f} fit_s {lr_s:.1f}", flush=True)
    del pred, proba, lr
    gc.collect()

    # --- Random Forest (train subsample, dense chunks on val) ---
    print(f"Random Forest (train subsample n={RF_TRAIN_N})…", flush=True)
    Xt_rf = prep.transform(rf_df[FEATURES])
    if hasattr(Xt_rf, "toarray"):
        Xt_rf = Xt_rf.astype(np.float32).toarray()
    else:
        Xt_rf = np.asarray(Xt_rf, dtype=np.float32)
    rf = RandomForestClassifier(
        n_estimators=40,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=1,
        random_state=SEED,
    )
    t0 = time.time()
    rf.fit(Xt_rf, rf_df["target"].astype(str))
    rf_s = time.time() - t0
    del Xt_rf
    gc.collect()
    pred = batched_predict(rf.predict, prep, val)
    proba = batched_predict(rf.predict_proba, prep, val)
    s = scores(y_val, pred, proba)
    log_rows.append(
        {
            "run_id": "rf_d12_n40",
            "model": "RandomForest",
            "family": "traditional",
            "seed": SEED,
            "train_rows_used": int(len(rf_df)),
            "runtime_sec_fit": round(rf_s, 2),
            "notes": f"balanced, n_estimators=40, max_depth=12, min_samples_leaf=20. Train subsample n={RF_TRAIN_N}.",
            "macro_f1_val": s["macro_f1"],
            "accuracy_val": s["accuracy"],
            "recall_resolved_slow_val": s["recall_resolved_slow"],
            "recall_unresolved_val": s["recall_unresolved"],
            "roc_auc_ovr_macro_val": s.get("roc_auc_ovr_macro"),
            "pr_auc_macro_val": s.get("pr_auc_macro"),
        }
    )
    val_table.append(
        {
            "model": "RandomForest",
            "family": "traditional",
            "macro_f1_val": s["macro_f1"],
            "accuracy_val": s["accuracy"],
            "recall_resolved_slow_val": s["recall_resolved_slow"],
            "recall_unresolved_val": s["recall_unresolved"],
            "roc_auc_ovr_macro_val": s.get("roc_auc_ovr_macro"),
            "pr_auc_macro_val": s.get("pr_auc_macro"),
        }
    )
    print(f"  val macro-F1 {s['macro_f1']:.4f} fit_s {rf_s:.1f}", flush=True)
    del pred, proba, rf
    gc.collect()

    # --- MLP ---
    print(f"MLP (train subsample n={NN_TRAIN_N}; early stop on 15% of subsample)…", flush=True)
    Xt_nn = prep.transform(nn_df[FEATURES])
    if hasattr(Xt_nn, "toarray"):
        Xt_nn = Xt_nn.astype(np.float32).toarray()
    else:
        Xt_nn = np.asarray(Xt_nn, dtype=np.float32)
    mlp = TorchMLPEstimator(hidden=128, dropout=0.2, max_epochs=8, patience=2, batch_size=2048)
    t0 = time.time()
    mlp.fit(Xt_nn, nn_df["target"].astype(str))
    mlp_s = time.time() - t0
    pd.DataFrame(mlp.history_).to_csv(EXP / "mlp_history.csv", index=False)
    plot_history(mlp.history_, "MLP train vs internal-val loss (early stopping; not official val)", "fig_q3_mlp_history.png")
    pred = batched_predict(mlp.predict, prep, val)
    proba = batched_predict(mlp.predict_proba, prep, val)
    s = scores(y_val, pred, proba)
    log_rows.append(
        {
            "run_id": "mlp_h128",
            "model": "MLP",
            "family": "neural",
            "seed": SEED,
            "train_rows_used": int(len(nn_df)),
            "runtime_sec_fit": round(mlp_s, 2),
            "notes": (
                f"PyTorch MLP, CE + class weights, Adam lr=1e-3, weight_decay=1e-4, "
                f"dropout=0.2, hidden=128, batch=2048, max_epochs=8, patience=2. "
                f"Train subsample n={NN_TRAIN_N}. Early stop on 15% of subsample, not official val."
            ),
            "macro_f1_val": s["macro_f1"],
            "accuracy_val": s["accuracy"],
            "recall_resolved_slow_val": s["recall_resolved_slow"],
            "recall_unresolved_val": s["recall_unresolved"],
            "roc_auc_ovr_macro_val": s.get("roc_auc_ovr_macro"),
            "pr_auc_macro_val": s.get("pr_auc_macro"),
        }
    )
    val_table.append(
        {
            "model": "MLP",
            "family": "neural",
            "macro_f1_val": s["macro_f1"],
            "accuracy_val": s["accuracy"],
            "recall_resolved_slow_val": s["recall_resolved_slow"],
            "recall_unresolved_val": s["recall_unresolved"],
            "roc_auc_ovr_macro_val": s.get("roc_auc_ovr_macro"),
            "pr_auc_macro_val": s.get("pr_auc_macro"),
        }
    )
    print(f"  val macro-F1 {s['macro_f1']:.4f} fit_s {mlp_s:.1f}", flush=True)
    del pred, proba, mlp, Xt_nn
    gc.collect()

    # --- Embedding net (raw frame, not one-hot) ---
    print(f"Tabular embedding net (train subsample n={NN_TRAIN_N})…", flush=True)
    emb = TabularEmbeddingEstimator(emb_dim=12, hidden=64, dropout=0.2, max_epochs=8, patience=2)
    t0 = time.time()
    emb.fit(nn_df[FEATURES], nn_df["target"].astype(str))
    emb_s = time.time() - t0
    pd.DataFrame(emb.history_).to_csv(EXP / "embedding_history.csv", index=False)
    plot_history(emb.history_, "Embedding net train vs internal-val loss (early stopping; not official val)", "fig_q3_embedding_history.png")
    pred = emb.predict(val[FEATURES])
    proba = emb.predict_proba(val[FEATURES])
    s = scores(y_val, pred, proba)
    log_rows.append(
        {
            "run_id": "embed_d12_h64",
            "model": "EmbeddingNet",
            "family": "neural",
            "seed": SEED,
            "train_rows_used": int(len(nn_df)),
            "runtime_sec_fit": round(emb_s, 2),
            "notes": (
                f"Embeddings for type/dept/suburb_group/ward/subcouncil/month/weekday, "
                f"emb_dim=12, hidden=64, dropout=0.2, Adam, CE class weights. "
                f"Train subsample n={NN_TRAIN_N}. Early stop on 15% of subsample."
            ),
            "macro_f1_val": s["macro_f1"],
            "accuracy_val": s["accuracy"],
            "recall_resolved_slow_val": s["recall_resolved_slow"],
            "recall_unresolved_val": s["recall_unresolved"],
            "roc_auc_ovr_macro_val": s.get("roc_auc_ovr_macro"),
            "pr_auc_macro_val": s.get("pr_auc_macro"),
        }
    )
    val_table.append(
        {
            "model": "EmbeddingNet",
            "family": "neural",
            "macro_f1_val": s["macro_f1"],
            "accuracy_val": s["accuracy"],
            "recall_resolved_slow_val": s["recall_resolved_slow"],
            "recall_unresolved_val": s["recall_unresolved"],
            "roc_auc_ovr_macro_val": s.get("roc_auc_ovr_macro"),
            "pr_auc_macro_val": s.get("pr_auc_macro"),
        }
    )
    print(f"  val macro-F1 {s['macro_f1']:.4f} fit_s {emb_s:.1f}", flush=True)
    del pred, proba, emb
    gc.collect()

    val_df = pd.DataFrame(val_table)
    val_df.to_csv(EXP / "q3_val_metrics.csv", index=False)
    pd.DataFrame(log_rows).to_csv(EXP / "q3_experiment_log.csv", index=False)
    plot_val_bar(val_table)

    # append to the existing dummy-only experiment log without test columns for new rows
    old = pd.read_csv(EXP / "experiment_log.csv") if (EXP / "experiment_log.csv").exists() else pd.DataFrame()
    new = pd.DataFrame(log_rows)
    # keep dummy rows from old; replace/add Q3 substantive rows
    keep = old[~old["run_id"].astype(str).isin(new["run_id"].astype(str))] if len(old) else old
    combined = pd.concat([keep, new], ignore_index=True)
    combined.to_csv(EXP / "experiment_log.csv", index=False)

    (EXP / "q3_summary.json").write_text(
        json.dumps({"compute": compute, "val_table": val_table, "test_loaded": False}, indent=2)
        + "\n"
    )
    print(val_df.to_string(index=False), flush=True)
    return val_df


if __name__ == "__main__":
    main()
