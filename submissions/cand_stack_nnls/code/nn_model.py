"""PyTorch tabular NN wrapped as a sklearn-compatible regressor.

The architecture mirrors the teacher's lab notebook for our competition:
  in_dim -> hidden -> hidden//2 -> hidden//4 -> 1
with ReLU activations and a single Dropout after the second linear.

`SklearnNNRegressor` follows the sklearn estimator contract so it plugs into
the existing Pipeline / GridSearchCV / cross_val_predict stack with no
special-casing in train.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

from .config import SEED


class TabularNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        h2 = max(hidden // 2, 1)
        h3 = max(hidden // 4, 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SklearnNNRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-style regressor backed by TabularNN.

    All __init__ parameters are stored verbatim (sklearn clone contract);
    training state is created in fit() with a trailing underscore.
    """

    def __init__(
        self,
        hidden: int = 128,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 30,
        weight_decay: float = 0.0,
        early_stopping: bool = True,
        patience: int = 5,
        val_frac: float = 0.1,
        seed: int = SEED,
        device: str | None = None,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.val_frac = val_frac
        self.seed = seed
        self.device = device

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_seeds(self) -> None:
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def fit(self, X, y):
        self._set_seeds()
        device = self._resolve_device()

        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float32).ravel()
        in_dim = X.shape[1]

        if self.early_stopping and self.val_frac > 0:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=self.val_frac, random_state=self.seed
            )
        else:
            X_tr, y_tr = X, y
            X_va = y_va = None

        model = TabularNN(in_dim, hidden=self.hidden, dropout=self.dropout).to(device)
        optim = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        X_tr_t = torch.from_numpy(X_tr).to(device)
        y_tr_t = torch.from_numpy(y_tr).to(device)
        if X_va is not None:
            X_va_t = torch.from_numpy(X_va).to(device)
            y_va_t = torch.from_numpy(y_va).to(device)

        n_train = X_tr_t.shape[0]
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        best_val = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(n_train, generator=gen)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optim.zero_grad()
                pred = model(X_tr_t[idx])
                loss = loss_fn(pred, y_tr_t[idx])
                loss.backward()
                optim.step()

            if X_va is not None:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_va_t)
                    val_loss = float(loss_fn(val_pred, y_va_t).item())
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if self.early_stopping and bad_epochs >= self.patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model_ = model
        self.device_ = device
        self.in_dim_ = in_dim
        self.best_val_loss_ = best_val if best_state is not None else None
        return self

    def predict(self, X) -> np.ndarray:
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        X_t = torch.from_numpy(X).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(X_t).cpu().numpy().astype(np.float64)
        return preds
