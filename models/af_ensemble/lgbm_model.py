"""LightGBM 진입 필터 — 스냅샷 피처 → P(label=1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier

from .feature_extractor import SNAPSHOT_COLS


class LGBMEntryFilter:
    def __init__(self, model_path: str | None = None):
        self.model: LGBMClassifier | None = None
        if model_path is not None:
            self.model = lgb.Booster(model_file=model_path) if model_path.endswith(".txt") \
                else self._load_pickle(model_path)

    @staticmethod
    def _load_pickle(path):
        import joblib
        return joblib.load(path)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_samples=20,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,   # -1이면 LightGBM OpenMP가 PyTorch OpenMP와 데드락 유발
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        """P(label=1) 배열 반환."""
        X = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=SNAPSHOT_COLS)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "LGBMEntryFilter":
        obj = cls()
        import joblib
        obj.model = joblib.load(path)
        return obj

    def feature_importance(self) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return (pd.DataFrame({"feature": SNAPSHOT_COLS[:len(imp)], "importance": imp})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))
