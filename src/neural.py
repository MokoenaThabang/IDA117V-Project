"""CPU PyTorch MLP (on one-hot) and tabular embedding net. Fit on a train subsample."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics_util import CLASSES
from src.prep import CAT_COLS, SEED, time_features

DEVICE = torch.device("cpu")


def _seed_all(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _class_index(y) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CLASSES)}
    return np.array([idx[str(v)] for v in y], dtype=np.int64)


def _macro_f1_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(1).cpu().numpy()
    yt = y.cpu().numpy()
    f1s = []
    for k in range(len(CLASSES)):
        tp = np.sum((pred == k) & (yt == k))
        fp = np.sum((pred == k) & (yt != k))
        fn = np.sum((pred != k) & (yt == k))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


class _MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int, dropout: float, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class _EmbedNet(nn.Module):
    def __init__(self, cardinalities: list[int], emb_dim: int, n_num: int, hidden: int, dropout: float):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(c + 1, emb_dim, padding_idx=c) for c in cardinalities]
        )
        n_in = emb_dim * len(cardinalities) + n_num
        self.head = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, x_cat, x_num):
        parts = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(parts + [x_num], dim=1)
        return self.head(x)


def _fit_torch(model, optimizer, loss_fn, loaders, max_epochs, patience, history):
    best_state = None
    best_f1 = -1.0
    bad = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_loss = 0.0
        nobs = 0
        for batch in loaders["train"]:
            optimizer.zero_grad()
            if len(batch) == 2:
                xb, yb = batch
                logits = model(xb)
            else:
                xc, xn, yb = batch
                logits = model(xc, xn)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item()) * len(yb)
            nobs += len(yb)
        model.eval()
        val_loss = 0.0
        nval = 0
        logits_all = []
        y_all = []
        with torch.no_grad():
            for batch in loaders["val"]:
                if len(batch) == 2:
                    xb, yb = batch
                    logits = model(xb)
                else:
                    xc, xn, yb = batch
                    logits = model(xc, xn)
                val_loss += float(loss_fn(logits, yb).item()) * len(yb)
                nval += len(yb)
                logits_all.append(logits)
                y_all.append(yb)
        logits_c = torch.cat(logits_all)
        y_c = torch.cat(y_all)
        vf1 = _macro_f1_from_logits(logits_c, y_c)
        row = {
            "epoch": epoch,
            "train_loss": tr_loss / max(nobs, 1),
            "val_loss": val_loss / max(nval, 1),
            "val_macro_f1": vf1,
        }
        history.append(row)
        print(f"  epoch {epoch} train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} val_macro_f1={vf1:.4f}", flush=True)
        if vf1 > best_f1 + 1e-4:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1


@dataclass
class TorchMLPEstimator(BaseEstimator, ClassifierMixin):
    hidden: int = 128
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 2048
    max_epochs: int = 8
    patience: int = 2
    seed: int = SEED
    history_: list | None = None

    def fit(self, X, y, X_val=None, y_val=None):
        _seed_all(self.seed)
        self.classes_ = np.array(CLASSES)
        X = np.asarray(X, dtype=np.float32)
        if hasattr(X, "toarray"):
            X = X.toarray().astype(np.float32)
        y_i = _class_index(y)
        if X_val is None:
            X, X_val, y_i, y_val_i = train_test_split(
                X, y_i, test_size=0.15, stratify=y_i, random_state=self.seed
            )
        else:
            X_val = np.asarray(X_val, dtype=np.float32)
            if hasattr(X_val, "toarray"):
                X_val = X_val.toarray().astype(np.float32)
            y_val_i = _class_index(y_val)
        self.n_in_ = X.shape[1]
        counts = np.bincount(y_i, minlength=3).astype(np.float32)
        w = counts.sum() / np.maximum(counts, 1) / 3.0
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w))
        self.model_ = _MLP(self.n_in_, self.hidden, self.dropout).to(DEVICE)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        tr = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y_i)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        va = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_i)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.history_ = []
        _fit_torch(self.model_, opt, loss_fn, {"train": tr, "val": va}, self.max_epochs, self.patience, self.history_)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        X = np.asarray(X, dtype=np.float32)
        if hasattr(X, "toarray"):
            X = X.toarray().astype(np.float32)
        self.model_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[i : i + self.batch_size])
                out.append(torch.softmax(self.model_(xb), 1).numpy())
        return np.concatenate(out, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(1)]


class TabularEmbeddingEstimator(BaseEstimator, ClassifierMixin):
    """Embeddings for suburb/ward/type/etc + weekend flag. Encoders fit in `fit`."""

    def __init__(
        self,
        emb_dim: int = 12,
        hidden: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 4096,
        max_epochs: int = 8,
        patience: int = 2,
        seed: int = SEED,
        drop: tuple = (),
    ):
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.drop = drop

    def _cat_cols(self):
        return [c for c in CAT_COLS + ["month", "weekday"] if c not in self.drop]

    def _prepare_frame(self, X):
        frame = time_features(X)
        cats = self._cat_cols()
        return frame, cats

    def fit(self, X, y, X_val=None, y_val=None):
        _seed_all(self.seed)
        self.classes_ = np.array(CLASSES)
        frame, cats = self._prepare_frame(X)
        self.cat_cols_ = cats
        self.ord_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.ord_.fit(frame[cats])
        self.scaler_ = StandardScaler()
        self.scaler_.fit(frame[["is_weekend"]])
        Xc, Xn = self._encode(frame)
        y_i = _class_index(y)
        if X_val is None:
            Xc, Xc_v, Xn, Xn_v, y_i, y_v = train_test_split(
                Xc, Xn, y_i, test_size=0.15, stratify=y_i, random_state=self.seed
            )
        else:
            fv, _ = self._prepare_frame(X_val)
            Xc_v, Xn_v = self._encode(fv)
            y_v = _class_index(y_val)
        card = [int(len(c)) for c in self.ord_.categories_]
        self.cardinalities_ = card
        counts = np.bincount(y_i, minlength=3).astype(np.float32)
        w = counts.sum() / np.maximum(counts, 1) / 3.0
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w))
        self.model_ = _EmbedNet(card, self.emb_dim, 1, self.hidden, self.dropout).to(DEVICE)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        tr = DataLoader(
            TensorDataset(torch.from_numpy(Xc), torch.from_numpy(Xn), torch.from_numpy(y_i)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        va = DataLoader(
            TensorDataset(torch.from_numpy(Xc_v), torch.from_numpy(Xn_v), torch.from_numpy(y_v)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.history_ = []
        _fit_torch(self.model_, opt, loss_fn, {"train": tr, "val": va}, self.max_epochs, self.patience, self.history_)
        return self

    def _encode(self, frame):
        cats = self.cat_cols_
        Xc = self.ord_.transform(frame[cats]).astype(np.int64)
        for j, cats_j in enumerate(self.ord_.categories_):
            Xc[Xc[:, j] < 0, j] = len(cats_j)
        Xn = self.scaler_.transform(frame[["is_weekend"]]).astype(np.float32)
        return Xc, Xn

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        frame, _ = self._prepare_frame(X)
        Xc, Xn = self._encode(frame)
        self.model_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xc), self.batch_size):
                xc = torch.from_numpy(Xc[i : i + self.batch_size])
                xn = torch.from_numpy(Xn[i : i + self.batch_size])
                out.append(torch.softmax(self.model_(xc, xn), 1).numpy())
        return np.concatenate(out, axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
