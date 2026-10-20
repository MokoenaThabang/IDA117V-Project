"""PyTorch models for the three-class resolution task.

The multilayer perceptron is an approved experiment. Lecturer approval has
been recorded for the existing embedding network. No approval date was
supplied, and this module does not change the architecture. Neither model
reads the locked test split.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import TARGET_LABELS, set_random_seeds
from src.evaluation import classification_scores
from src.training import (
    EMBEDDING_PARAMETERS,
    MLP_PARAMETERS,
    assert_split_may_be_scored,
    balanced_class_weights,
    class_indices,
)

APPROVED_EMBEDDING_NOTE = (
    "Lecturer approval has been recorded for the existing embedding network. "
    "No approval date was supplied, and the architecture was not changed."
)


class FeedForwardNet(nn.Module):
    """One-hot features to three class logits.

    Dropout is applied after each hidden ReLU. The output layer is linear,
    and cross-entropy applies the log-softmax.
    """

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...] = (128, 64), dropout: float = 0.3):
        super().__init__()
        if n_features < 1:
            raise ValueError(f"The MLP needs at least one input. Got {n_features}.")
        if not hidden_sizes:
            raise ValueError("The MLP needs at least one hidden layer.")
        layers: list[nn.Module] = []
        previous = n_features
        for size in hidden_sizes:
            layers.extend([nn.Linear(previous, int(size)), nn.ReLU(), nn.Dropout(float(dropout))])
            previous = int(size)
        layers.append(nn.Linear(previous, len(TARGET_LABELS)))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class EmbeddingNet(nn.Module):
    """Categorical embeddings plus the weekend indicator, then three logits.

    Index 0 of every embedding is padding. Categories that were absent from
    training use that index and contribute a zero vector. Lecturer approval
    has been recorded for this existing architecture.
    """

    def __init__(
        self,
        cardinalities: list[int],
        embedding_dim: int = 16,
        hidden_size: int = 64,
        dropout: float = 0.3,
        n_numeric: int = 1,
    ):
        super().__init__()
        if any(int(size) < 2 for size in cardinalities):
            raise ValueError("Each embedding needs a padding row and at least one trained level.")
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(size), int(embedding_dim), padding_idx=0) for size in cardinalities]
        )
        for layer in self.embeddings:
            with torch.no_grad():
                layer.weight[0].zero_()
        in_features = int(embedding_dim) * len(cardinalities) + int(n_numeric)
        self.head = nn.Sequential(
            nn.Linear(in_features, int(hidden_size)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), len(TARGET_LABELS)),
        )

    def forward(self, codes: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        embedded = [layer(codes[:, index]) for index, layer in enumerate(self.embeddings)]
        hidden = torch.cat([*embedded, numeric], dim=1)
        return self.head(hidden)


def _as_csr(matrix):
    import scipy.sparse as sparse

    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
    else:
        array = np.asarray(matrix)
        if array.ndim != 2:
            raise ValueError(f"MLP features must be a 2-D matrix. Got shape {array.shape}.")
        matrix = sparse.csr_matrix(array)
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)
    return matrix


def _labels_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    indices = np.argmax(probabilities, axis=1)
    return np.asarray(TARGET_LABELS, dtype=object)[indices]


def _tensor(array, dtype) -> torch.Tensor:
    """Copy into a writable tensor. Some pandas arrays are read-only views."""
    copied = np.array(array, dtype=dtype, copy=True)
    return torch.from_numpy(copied)


def _dense_rows(matrix, positions: np.ndarray) -> torch.Tensor:
    return _tensor(matrix[positions].toarray(), np.float32)


def _epoch_batches(n_rows: int, batch_size: int, generator: np.random.Generator, shuffle: bool):
    order = np.arange(n_rows)
    if shuffle:
        generator.shuffle(order)
    for start in range(0, n_rows, batch_size):
        yield order[start : start + batch_size]


def _parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _restore_best(module: nn.Module, state: dict | None) -> None:
    if state is None:
        raise RuntimeError("Training finished without a saved validation checkpoint.")
    module.load_state_dict(state)
    module.eval()


def train_feedforward(
    X_train,
    y_train,
    X_validation,
    y_validation,
    parameters: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Train the MLP with early stopping on validation macro-F1.

    Preprocessing is assumed to have been fit on the training period already.
    ``X_train`` and ``X_validation`` are the transformed matrices.
    """
    assert_split_may_be_scored("validation")
    settings = dict(MLP_PARAMETERS)
    if parameters:
        settings.update(parameters)
    seed = int(settings["seed"])
    set_random_seeds(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    generator = np.random.default_rng(seed)

    train_matrix = _as_csr(X_train)
    validation_matrix = _as_csr(X_validation)
    y_train_index = class_indices(y_train)
    y_validation_index = class_indices(y_validation)
    if train_matrix.shape[0] != len(y_train_index) or validation_matrix.shape[0] != len(y_validation_index):
        raise ValueError("Feature rows and labels have different lengths.")
    weights = balanced_class_weights(y_train)
    hidden = tuple(int(size) for size in settings["hidden_sizes"])
    model = FeedForwardNet(
        n_features=train_matrix.shape[1],
        hidden_sizes=hidden,
        dropout=float(settings["dropout"]),
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    batch_size = int(settings["batch_size"])
    max_epochs = int(settings["max_epochs"])
    patience = int(settings["patience"])
    best_score = -1.0
    best_state = None
    best_epoch = 0
    wait = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        for positions in _epoch_batches(train_matrix.shape[0], batch_size, generator, shuffle=True):
            features = _dense_rows(train_matrix, positions)
            labels = _tensor(y_train_index[positions], np.int64)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimiser.step()
        train_predictions, train_proba, train_loss = predict_feedforward(
            model, train_matrix, y_train_index, batch_size, criterion
        )
        validation_predictions, validation_proba, validation_loss = predict_feedforward(
            model, validation_matrix, y_validation_index, batch_size, criterion
        )
        train_f1 = classification_scores(y_train, train_predictions)["macro_f1"]
        validation_f1 = classification_scores(y_validation, validation_predictions)["macro_f1"]
        improved = validation_f1 > best_score
        if improved:
            best_score = validation_f1
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_macro_f1": train_f1,
                "validation_macro_f1": validation_f1,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "improved": improved,
            }
        )
        if verbose:
            print(
                f"mlp epoch {epoch}: train_loss {train_loss:.4f}, val_loss {validation_loss:.4f}, "
                f"train_macro_f1 {train_f1:.4f}, val_macro_f1 {validation_f1:.4f}",
                flush=True,
            )
        if wait >= patience:
            break

    fit_seconds = time.perf_counter() - started
    _restore_best(model, best_state)
    # The last history row may be past the restored epoch. Callers score the restored weights.
    del train_predictions, train_proba, validation_predictions, validation_proba
    return {
        "model": model,
        "history": pd.DataFrame(history),
        "fit_seconds": float(fit_seconds),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "stopped_early": bool(len(history) < max_epochs),
        "best_validation_macro_f1": float(best_score),
        "n_parameters": _parameter_count(model),
        "n_features": int(train_matrix.shape[1]),
        "class_weights": [float(weight) for weight in weights],
        "parameters": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in settings.items()
        },
        "approval": "approved",
    }


def predict_feedforward(model, matrix, y_index, batch_size: int, criterion=None):
    """Return class names, probabilities in label order, and mean weighted loss."""
    model.eval()
    csr = _as_csr(matrix)
    probabilities = []
    losses = []
    rows = []
    with torch.inference_mode():
        for start in range(0, csr.shape[0], batch_size):
            positions = np.arange(start, min(start + batch_size, csr.shape[0]))
            features = _dense_rows(csr, positions)
            logits = model(features)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            if criterion is not None:
                labels = _tensor(y_index[positions], np.int64)
                losses.append(float(criterion(logits, labels)) * len(positions))
                rows.append(len(positions))
    proba = np.vstack(probabilities)
    mean_loss = float(sum(losses) / sum(rows)) if rows else float("nan")
    return _labels_from_probabilities(proba), proba, mean_loss


def train_embedding_network(
    train_codes: np.ndarray,
    train_numeric: np.ndarray,
    y_train,
    validation_codes: np.ndarray,
    validation_numeric: np.ndarray,
    y_validation,
    cardinalities: list[int],
    parameters: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Train the approved embedding network.

    Early stopping uses validation macro-F1. This function does not read the
    locked test and it does not decide the champion.
    """
    assert_split_may_be_scored("validation")
    settings = dict(EMBEDDING_PARAMETERS)
    if parameters:
        settings.update(parameters)
    seed = int(settings["seed"])
    set_random_seeds(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    generator = np.random.default_rng(seed)
    y_train_index = class_indices(y_train)
    y_validation_index = class_indices(y_validation)
    weights = balanced_class_weights(y_train)
    model = EmbeddingNet(
        cardinalities=cardinalities,
        embedding_dim=int(settings["embedding_dim"]),
        hidden_size=int(settings["hidden_size"]),
        dropout=float(settings["dropout"]),
        n_numeric=int(train_numeric.shape[1]),
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    batch_size = int(settings["batch_size"])
    max_epochs = int(settings["max_epochs"])
    patience = int(settings["patience"])
    best_score = -1.0
    best_state = None
    best_epoch = 0
    wait = 0
    history = []
    started = time.perf_counter()
    train_codes = np.asarray(train_codes, dtype=np.int64)
    validation_codes = np.asarray(validation_codes, dtype=np.int64)
    train_numeric = np.asarray(train_numeric, dtype=np.float32)
    validation_numeric = np.asarray(validation_numeric, dtype=np.float32)

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        for positions in _epoch_batches(len(train_codes), batch_size, generator, shuffle=True):
            codes = _tensor(train_codes[positions], np.int64)
            numeric = _tensor(train_numeric[positions], np.float32)
            labels = _tensor(y_train_index[positions], np.int64)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(codes, numeric), labels)
            loss.backward()
            optimiser.step()
        train_predictions, _train_proba, train_loss = predict_embedding(
            model, train_codes, train_numeric, y_train_index, batch_size, criterion
        )
        validation_predictions, _validation_proba, validation_loss = predict_embedding(
            model, validation_codes, validation_numeric, y_validation_index, batch_size, criterion
        )
        train_f1 = classification_scores(y_train, train_predictions)["macro_f1"]
        validation_f1 = classification_scores(y_validation, validation_predictions)["macro_f1"]
        improved = validation_f1 > best_score
        if improved:
            best_score = validation_f1
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_macro_f1": train_f1,
                "validation_macro_f1": validation_f1,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "improved": improved,
            }
        )
        if verbose:
            print(
                f"embedding epoch {epoch}: train_loss {train_loss:.4f}, val_loss {validation_loss:.4f}, "
                f"train_macro_f1 {train_f1:.4f}, val_macro_f1 {validation_f1:.4f}",
                flush=True,
            )
        if wait >= patience:
            break

    _restore_best(model, best_state)
    return {
        "model": model,
        "history": pd.DataFrame(history),
        "fit_seconds": float(time.perf_counter() - started),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "stopped_early": bool(len(history) < max_epochs),
        "best_validation_macro_f1": float(best_score),
        "n_parameters": _parameter_count(model),
        "cardinalities": [int(size) for size in cardinalities],
        "class_weights": [float(weight) for weight in weights],
        "parameters": settings,
        "approval": "approved",
        "approval_note": APPROVED_EMBEDDING_NOTE,
    }


def predict_embedding(model, codes, numeric, y_index, batch_size: int, criterion=None):
    """Return class names, probabilities in label order, and mean weighted loss."""
    model.eval()
    codes = np.asarray(codes, dtype=np.int64)
    numeric = np.asarray(numeric, dtype=np.float32)
    probabilities = []
    losses = []
    rows = []
    with torch.inference_mode():
        for start in range(0, len(codes), batch_size):
            stop = min(start + batch_size, len(codes))
            logits = model(
                _tensor(codes[start:stop], np.int64),
                _tensor(numeric[start:stop], np.float32),
            )
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            if criterion is not None:
                labels = _tensor(y_index[start:stop], np.int64)
                losses.append(float(criterion(logits, labels)) * (stop - start))
                rows.append(stop - start)
    proba = np.vstack(probabilities)
    mean_loss = float(sum(losses) / sum(rows)) if rows else float("nan")
    return _labels_from_probabilities(proba), proba, mean_loss
