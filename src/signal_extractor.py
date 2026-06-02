"""
[Signal] 학습된 전문가 모델로 전체 데이터셋에서 시그널(예측 확률값)을 추출.

출력 컬럼:
  - signal_long    : Long Expert 예측 확률 (0~1)
  - signal_short   : Short Expert 예측 확률 (0~1)
  - signal_context : Context Expert 예측 확률 (0~1)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_pipeline import FEATURE_COLS
from expert_models import ExpertModel, SequenceDataset, SEQ_LEN, BATCH_SIZE


@torch.no_grad()
def extract_signals_from_df(
    df: pd.DataFrame,
    models: dict,
    device: torch.device = None,
    batch_size: int = BATCH_SIZE * 4,
) -> pd.DataFrame:
    """
    DataFrame 전체에 대해 세 전문가 모델의 예측 확률을 계산하여 추가.

    Parameters
    ----------
    df      : 스케일링 완료된 FEATURE_COLS 포함 DataFrame
    models  : {"long": model, "short": model, "context": model}
    device  : torch device (None이면 자동 감지)

    Returns
    -------
    df_out  : 원본 df에 signal_long / signal_short / signal_context 컬럼 추가.
              앞 SEQ_LEN 행은 시퀀스 구성 불가 → NaN.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model in models.values():
        model.eval()
        model.to(device)

    n = len(df)
    signal_arrays = {name: np.full(n, np.nan) for name in models}

    _dummy_label = "__dummy_seq_label__"
    df = df.copy()
    df[_dummy_label] = 0.0

    dataset = SequenceDataset(df, label_col=_dummy_label, seq_len=SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for name, model in models.items():
        all_probs = []
        for x, _ in loader:
            logits = model(x.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

        probs_full = np.concatenate(all_probs)
        signal_arrays[name][SEQ_LEN:] = probs_full

    df_out = df.copy()
    df_out["signal_long"]    = signal_arrays["long"]
    df_out["signal_short"]   = signal_arrays["short"]
    df_out["signal_context"] = signal_arrays["context"]
    df_out.drop(columns=[_dummy_label], inplace=True, errors="ignore")

    n_valid = int(np.sum(~np.isnan(signal_arrays["long"])))
    print(f"[Signal] 추출 완료 | 전체:{n} | 유효:{n_valid} | NaN(앞부분):{n - n_valid}")
    _log_signal_stats(df_out)

    return df_out


def _log_signal_stats(df: pd.DataFrame):
    for col in ["signal_long", "signal_short", "signal_context"]:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        print(f"  {col:20s} | mean={s.mean():.3f} | std={s.std():.3f} | "
              f"p25={s.quantile(0.25):.3f} | p75={s.quantile(0.75):.3f}")
