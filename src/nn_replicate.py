"""Replicate the teacher's lab-notebook NN exactly and report val RMSE.

Mirrors Part 1 of the official course lab for `cse-281-spring-26-item-price-prediction`
**verbatim**: same imputation, same OneHotEncoder/StandardScaler split, same
DataLoader+shuffle batching, same model init order, same Adam+MSE, same 10
epochs flat with no early stopping, same single 80/20 split.

Lab's reported best val RMSE: 0.5834. Our replication should land within
~0.05 of that — perfect bit-match isn't expected because:
  - The lab ran on a Tesla T4 (CUDA); we run on CPU. Different float32
    accumulation order produces small but non-zero gradient drift.
  - Our pandas reads strings as `str` dtype (3.0+) vs the lab's `object`
    (2.x); train_test_split sees the same row partition either way.

Usage: python -m src.nn_replicate
Output: results/teacher_replication.txt with the val RMSE on a single line.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from .config import RESULTS_DIR, SEED, TRAIN_CSV
from .nn_model import TabularNN

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3


class _TabularDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def _build_preprocessor(numerical_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", ohe, categorical_cols),
        ]
    )


def main() -> None:
    # Seed in the same order the lab does it.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)
    X = df.drop(columns=["Y"])
    y = df["Y"].astype(np.float32)

    # Lab's imputation: median for numerics, 'Unknown' for everything else.
    # Pandas 3.0 reads strings as `str` dtype, not `object`, so split on numeric.
    X = X.copy()
    is_num = {col: pd.api.types.is_numeric_dtype(X[col]) for col in X.columns}
    for col, numeric in is_num.items():
        if numeric:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna("Unknown")
    numerical_cols = [c for c, n in is_num.items() if n]
    categorical_cols = [c for c, n in is_num.items() if not n]
    print(f"  categorical_cols: {categorical_cols}")
    print(f"  numerical_cols:   {numerical_cols}")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"  X_train: {X_tr.shape}  | X_val: {X_va.shape}")

    pre = _build_preprocessor(numerical_cols, categorical_cols)
    X_tr_p = pre.fit_transform(X_tr).astype(np.float32)
    X_va_p = pre.transform(X_va).astype(np.float32)
    print(f"  Processed train: {X_tr_p.shape}  | Processed val: {X_va_p.shape}")

    train_loader = DataLoader(
        _TabularDS(X_tr_p, y_tr.values), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        _TabularDS(X_va_p, y_va.values), batch_size=BATCH_SIZE, shuffle=False
    )

    in_dim = X_tr_p.shape[1]
    model = TabularNN(in_dim, hidden=128, dropout=0.2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()

        model.eval()
        preds = []
        targs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
                targs.append(yb.numpy())
        val_pred = np.concatenate(preds)
        val_targ = np.concatenate(targs)
        val_rmse = float(np.sqrt(mean_squared_error(val_targ, val_pred)))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse

    print(f"\n  Final-epoch val RMSE: {val_rmse:.4f}")
    print(f"  Best-of-10  val RMSE: {best_val_rmse:.4f}")
    print(f"  Lab's reported number: 0.5834  (best-of-10 over their training)")
    print(f"  Delta (best vs lab):  {best_val_rmse - 0.5834:+.4f}")

    out = RESULTS_DIR / "teacher_replication.txt"
    out.write_text(f"{best_val_rmse:.6f}\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
