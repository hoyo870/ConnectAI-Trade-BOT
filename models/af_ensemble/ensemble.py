"""LGBM + LSTM 앙상블 + 임계값 캘리브레이션."""
import json
import os

import numpy as np

from .feature_extractor import SNAPSHOT_COLS
from .lgbm_model import LGBMEntryFilter
from .lstm_model import LSTMEntryFilter


class AFEnsemble:
    """LGBM + LSTM ensemble with threshold calibration."""

    def __init__(self, lgbm: LGBMEntryFilter, lstm: LSTMEntryFilter,
                 threshold: float = 0.5,
                 lgbm_weight: float = 0.5,
                 lstm_weight: float = 0.5):
        self.lgbm = lgbm
        self.lstm = lstm
        self.threshold = threshold
        self.lgbm_weight = lgbm_weight
        self.lstm_weight = lstm_weight

    def predict_proba_from_features(self, snapshot_dict: dict, sequence: np.ndarray) -> float:
        """단일 예측. snapshot_dict keys = SNAPSHOT_COLS, sequence shape (20,7)."""
        X_snap = np.array([[snapshot_dict[c] for c in SNAPSHOT_COLS]], dtype=np.float32)
        p_lgbm = self.lgbm.predict_proba(X_snap)[0]
        p_lstm = self.lstm.predict_proba(sequence[np.newaxis])[0]
        return self.lgbm_weight * p_lgbm + self.lstm_weight * p_lstm

    def should_enter(self, snapshot_dict: dict, sequence: np.ndarray) -> bool:
        return self.predict_proba_from_features(snapshot_dict, sequence) >= self.threshold

    def calibrate_threshold(self, X_snap_val, X_seq_val, y_val,
                            theta_range=(0.30, 0.70, 0.05)):
        """
        theta 스윕 → pass_rate 유지 가능 범위에서 품질 점수 최대화.
        실제 TPD 검증은 backtest_af_exact.py 또는 AntifragileBacktestRunner로 수행.
        """
        thetas = np.arange(*theta_range)
        p_lgbm = self.lgbm.predict_proba(X_snap_val)
        p_lstm = self.lstm.predict_proba(X_seq_val)
        p_combined = self.lgbm_weight * p_lgbm + self.lstm_weight * p_lstm

        best_theta = 0.5
        best_score = -np.inf
        for theta in thetas:
            mask = p_combined >= theta
            keep_rate = mask.mean()
            if keep_rate < 0.3:  # 거래 너무 줄면 TPD 붕괴
                continue
            score = p_combined[mask].mean() * keep_rate
            if score > best_score:
                best_score = score
                best_theta = theta
        self.threshold = float(best_theta)
        return float(best_theta)

    def save(self, directory: str):
        """모든 컴포넌트를 directory에 저장."""
        os.makedirs(directory, exist_ok=True)
        self.lgbm.save(os.path.join(directory, "lgbm.pkl"))
        self.lstm.save(os.path.join(directory, "lstm.pt"))
        meta = {
            "threshold": self.threshold,
            "lgbm_weight": self.lgbm_weight,
            "lstm_weight": self.lstm_weight,
            "input_size": self.lstm.input_size,
            "hidden": self.lstm.hidden,
            "num_layers": self.lstm.num_layers,
            "dropout": self.lstm.dropout,
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "AFEnsemble":
        """directory에서 모든 컴포넌트 로드."""
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
        lgbm = LGBMEntryFilter.load(os.path.join(directory, "lgbm.pkl"))
        lstm = LSTMEntryFilter(
            model_path=os.path.join(directory, "lstm.pt"),
            input_size=meta.get("input_size", 7),
            hidden=meta.get("hidden", 64),
            num_layers=meta.get("num_layers", 2),
            dropout=meta.get("dropout", 0.2),
        )
        return cls(lgbm, lstm,
                   threshold=meta["threshold"],
                   lgbm_weight=meta["lgbm_weight"],
                   lstm_weight=meta["lstm_weight"])
