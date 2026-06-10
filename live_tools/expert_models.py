"""
[DL] 3개의 전문가 모델 (PyTorch)
  - LongExpert  : 매수 타점 이진 분류
  - ShortExpert : 매도 타점 이진 분류
  - ContextExpert: 고변동/추세 구간 이진 분류

공통 아키텍처: Temporal Conv + Attention + FC Head
클래스 불균형 → BCEWithLogitsLoss(pos_weight) 로 대응
"""
from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
import joblib

from data_pipeline import FEATURE_COLS, LABEL_COLS, train_val_test_split


# ── 설정 ──────────────────────────────────────────────────────────────────────
SEQ_LEN      = 60       # 입력 시퀀스 길이 (봉 수)
HIDDEN_DIM   = 128
NUM_HEADS    = 4
NUM_TCN_LAYERS = 4
DROPOUT      = 0.3
BATCH_SIZE   = 256
EPOCHS       = 50
LR           = 1e-3
PATIENCE     = 7        # early stopping
MODEL_DIR    = "models"

EXPERT_CONFIG = {
    "long":    {"label": "label_long",    "save": "models/expert_long.pt"},
    "short":   {"label": "label_short",   "save": "models/expert_short.pt"},
    "context": {"label": "label_context", "save": "models/expert_context.pt"},
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """슬라이딩 윈도우 방식으로 (seq_len, n_features) 시퀀스를 생성."""

    def __init__(self, df: pd.DataFrame, label_col: str, seq_len: int = SEQ_LEN):
        self.seq_len   = seq_len
        self.features  = df[FEATURE_COLS].values.astype(np.float32)
        self.labels    = df[label_col].values.astype(np.float32)

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]          # (seq_len, n_feat)
        y = self.labels[idx + self.seq_len]                   # scalar
        return torch.from_numpy(x), torch.tensor(y)


# ── 모델 구성 요소 ─────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """인과적(미래 정보 누수 없음) 1D Dilated Conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv    = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.chomp   = pad  # 미래 패딩 제거량
        self.act     = nn.GELU()
        self.norm    = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv(x)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        out = self.act(out)
        out = out.transpose(1, 2)   # (B, T, C)
        out = self.norm(out)
        out = out.transpose(1, 2)   # (B, C, T)
        return self.dropout(out)


class TCNBlock(nn.Module):
    """Residual Temporal Convolutional Block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1    = CausalConv1d(in_ch,  out_ch, kernel, dilation)
        self.conv2    = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.residual(x)


class TemporalAttention(nn.Module):
    """Multi-Head Self-Attention (시간 축)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=DROPOUT, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, C)
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


# ── 전문가 모델 ───────────────────────────────────────────────────────────────

class ExpertModel(nn.Module):
    """
    TCN + Temporal Attention + FC Head 이진 분류 모델.

    입력: (B, seq_len, n_features)
    출력: (B,) logit
    """

    def __init__(self, n_features: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        # 입력 투영
        self.input_proj = nn.Linear(n_features, hidden)

        # TCN (지수적 dilation: 1, 2, 4, 8)
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden, hidden, kernel=3, dilation=2 ** i)
            for i in range(NUM_TCN_LAYERS)
        ])

        # Temporal Attention
        self.attn = TemporalAttention(hidden, NUM_HEADS)

        # 분류 헤드
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_proj(x)          # (B, T, H)
        x = x.transpose(1, 2)           # (B, H, T)  — TCN expects channel-first
        for block in self.tcn_blocks:
            x = block(x)
        x = x.transpose(1, 2)           # (B, T, H)
        x = self.attn(x)                # (B, T, H)
        x = x[:, -1, :]                 # 마지막 타임스텝만 사용
        return self.head(x).squeeze(-1) # (B,)


# ── 학습 유틸 ─────────────────────────────────────────────────────────────────

def _compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """양성 클래스 비율 기반 pos_weight 계산 (BCEWithLogitsLoss용)."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    weight = n_neg / (n_pos + 1e-9)
    return torch.tensor(weight, dtype=torch.float32)


def _make_loader(df: pd.DataFrame, label_col: str, shuffle: bool) -> DataLoader:
    ds = SequenceDataset(df, label_col)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= 0.5).astype(int)
    return {
        "auc":  roc_auc_score(labels, probs),
        "f1":   f1_score(labels, preds, zero_division=0),
        "loss": float(nn.BCEWithLogitsLoss()(torch.tensor(logits), torch.tensor(labels))),
    }


# ── 학습 루프 ─────────────────────────────────────────────────────────────────

def train_expert(
    name: str,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    device:   torch.device | None = None,
) -> ExpertModel:
    """전문가 모델 1개를 학습하여 반환 (best checkpoint 자동 저장)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg       = EXPERT_CONFIG[name]
    label_col = cfg["label"]
    save_path = cfg["save"]

    print(f"\n{'='*60}")
    print(f"[DL] {name.upper()} Expert 학습 시작 | device={device}")

    # DataLoader
    train_loader = _make_loader(train_df, label_col, shuffle=True)
    val_loader   = _make_loader(val_df,   label_col, shuffle=False)

    # pos_weight
    train_labels = train_df[label_col].values
    pos_weight   = _compute_pos_weight(train_labels).to(device)
    print(f"[DL] pos_weight={pos_weight.item():.3f} (양성:{train_labels.mean():.3f})")

    # 모델
    n_features = len(FEATURE_COLS)
    model      = ExpertModel(n_features).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc   = 0.0
    no_improve = 0
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            metrics = _evaluate(model, val_loader, device)
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                  f"val_auc={metrics['auc']:.4f} | val_f1={metrics['f1']:.4f}")

            if metrics["auc"] > best_auc:
                best_auc   = metrics["auc"]
                no_improve = 0
                torch.save(model.state_dict(), save_path)
            else:
                no_improve += 5
                if no_improve >= PATIENCE * 5:
                    print(f"[DL] Early stopping at epoch {epoch}")
                    break

    # 최고 체크포인트 로드
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"[DL] {name.upper()} Expert 완료 | best_val_auc={best_auc:.4f} | 저장: {save_path}")
    return model


def train_all_experts(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    device:   torch.device | None = None,
) -> dict[str, ExpertModel]:
    """Long / Short / Context 세 전문가 모델을 순차 학습."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name in EXPERT_CONFIG:
        models[name] = train_expert(name, train_df, val_df, device)
    return models


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

def load_expert(name: str, device: torch.device | None = None) -> ExpertModel:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = len(FEATURE_COLS)
    model = ExpertModel(n_features).to(device)
    path  = EXPERT_CONFIG[name]["save"]
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_all_experts(device: torch.device | None = None) -> dict[str, ExpertModel]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {name: load_expert(name, device) for name in EXPERT_CONFIG}


# ── 실행 예시 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/train.parquet"
    val_path   = sys.argv[2] if len(sys.argv) > 2 else "data/val.parquet"

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DL] 사용 디바이스: {device}")

    models = train_all_experts(train_df, val_df, device)
    print("\n[DL] 전체 전문가 모델 학습 완료.")
