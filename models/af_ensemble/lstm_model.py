"""PyTorch LSTM 진입 필터 — (20, 7) 시퀀스 → P(label=1)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _auto_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LSTMBlock(nn.Module):
    def __init__(self, input_size: int = 7, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class LSTMEntryFilter:
    def __init__(self, model_path: str | None = None, device=None,
                 input_size: int = 7, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        self.device = device or _auto_device()
        self.input_size = input_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.net = LSTMBlock(input_size, hidden, num_layers, dropout).to(self.device)
        if model_path is not None:
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.net.eval()

    def fit(self, X_train, y_train, X_val, y_val,
            epochs: int = 100, batch_size: int = 256, patience: int = 10, lr: float = 1e-3):
        Xtr = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        ytr = torch.tensor(np.asarray(y_train), dtype=torch.float32)
        Xva = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(self.device)
        yva = torch.tensor(np.asarray(y_val), dtype=torch.float32).to(self.device)

        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        best_val = float("inf")
        best_state = None
        no_improve = 0

        for _ in range(epochs):
            self.net.train()
            for xb, yb in loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(Xva)
                val_loss = criterion(val_pred, yva).item()
            scheduler.step(val_loss)

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_proba(self, X) -> np.ndarray:
        """X: (n,20,7) 또는 (20,7). P(label=1) 배열 (n,) 반환."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[np.newaxis]
        self.net.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            out = self.net(xb).cpu().numpy()
        return out.reshape(-1)

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    @classmethod
    def load(cls, path: str, device=None) -> "LSTMEntryFilter":
        return cls(model_path=path, device=device)
