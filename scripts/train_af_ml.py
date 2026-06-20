"""
AF ML 앙상블 학습 CLI.

Usage:
  python scripts/train_af_ml.py --train-end 2024-12-31 --val-end 2025-12-31 --out models/af_ensemble/saved/

Flow:
  1. data/ml_labels/ 의 전 코인 라벨 CSV/NPZ 로드
  2. date 기준 train(<=train-end) / val(train-end<date<=val-end) 분할
  3. LGBM, LSTM 각각 학습
  4. AFEnsemble 생성 + val에서 임계값 캘리브레이션
  5. --out 디렉터리에 저장
  6. train AUC / val AUC / calibrated theta 출력
"""
import os, sys
# LightGBM과 PyTorch OpenMP 라이브러리 충돌(세그폴트) 방지
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, "src")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from models.af_ensemble.feature_extractor import SNAPSHOT_COLS
from models.af_ensemble.lgbm_model import LGBMEntryFilter
from models.af_ensemble.lstm_model import LSTMEntryFilter
from models.af_ensemble.ensemble import AFEnsemble


def load_all_labels(labels_dir: Path):
    """전 코인 snapshot CSV + sequence NPZ 병합. (snap_df, X_seq) 반환."""
    snap_frames = []
    seq_chunks = []
    for csv_path in sorted(labels_dir.glob("*_snapshot.csv")):
        coin = csv_path.name.replace("_snapshot.csv", "")
        npz_path = labels_dir / f"{coin}_sequences.npz"
        if not npz_path.exists():
            print(f"  ⚠️  {coin}: 시퀀스 파일 없음, 스킵")
            continue
        sdf = pd.read_csv(csv_path)
        arr = np.load(npz_path)
        X = arr["X"]
        if len(sdf) != len(X):
            print(f"  ⚠️  {coin}: snapshot/sequence 행 수 불일치 ({len(sdf)} vs {len(X)}), 스킵")
            continue
        snap_frames.append(sdf)
        seq_chunks.append(X)
    if not snap_frames:
        raise RuntimeError("라벨 데이터 없음. 먼저 generate_ml_labels.py 실행 필요.")
    snap_df = pd.concat(snap_frames, ignore_index=True)
    X_seq = np.concatenate(seq_chunks, axis=0)
    return snap_df, X_seq


def main():
    parser = argparse.ArgumentParser(description="Train AF ML ensemble")
    parser.add_argument("--labels", default="data/ml_labels/")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--val-end", default="2025-12-31")
    parser.add_argument("--out", default="models/af_ensemble/saved/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default=None, help="torch device (cpu, cuda, mps). 기본: 자동 감지")
    args = parser.parse_args()

    labels_dir = ROOT / args.labels
    snap_df, X_seq = load_all_labels(labels_dir)

    dates = pd.to_datetime(snap_df["date"])
    train_mask = (dates <= args.train_end).to_numpy()
    val_mask = ((dates > args.train_end) & (dates <= args.val_end)).to_numpy()

    X_snap = snap_df[SNAPSHOT_COLS].to_numpy(dtype=np.float32)
    y = snap_df["label"].to_numpy(dtype=np.int64)

    Xs_tr, Xs_va = X_snap[train_mask], X_snap[val_mask]
    Xq_tr, Xq_va = X_seq[train_mask], X_seq[val_mask]
    y_tr, y_va = y[train_mask], y[val_mask]

    print(f"train={len(y_tr)} (pos {y_tr.mean()*100:.1f}%)  val={len(y_va)} (pos {y_va.mean()*100:.1f}%)")
    if len(y_tr) == 0 or len(y_va) == 0:
        raise RuntimeError("train/val 분할 후 데이터 부족. --train-end / --val-end 확인.")

    # LGBM
    lgbm = LGBMEntryFilter()
    lgbm.fit(Xs_tr, y_tr, Xs_va, y_va)
    p_tr = lgbm.predict_proba(Xs_tr)
    p_va = lgbm.predict_proba(Xs_va)
    lgbm_train_auc = roc_auc_score(y_tr, p_tr) if len(np.unique(y_tr)) > 1 else float("nan")
    lgbm_val_auc = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) > 1 else float("nan")

    # LSTM
    lstm = LSTMEntryFilter(device=args.device)
    lstm.fit(Xq_tr, y_tr, Xq_va, y_va, epochs=args.epochs)
    pl_tr = lstm.predict_proba(Xq_tr)
    pl_va = lstm.predict_proba(Xq_va)
    lstm_train_auc = roc_auc_score(y_tr, pl_tr) if len(np.unique(y_tr)) > 1 else float("nan")
    lstm_val_auc = roc_auc_score(y_va, pl_va) if len(np.unique(y_va)) > 1 else float("nan")

    # Ensemble + 캘리브레이션
    ens = AFEnsemble(lgbm, lstm)
    theta = ens.calibrate_threshold(Xs_va, Xq_va, y_va)

    out_dir = ROOT / args.out
    ens.save(str(out_dir))

    print("\n" + "=" * 50)
    print(f"  LGBM  train AUC={lgbm_train_auc:.4f}  val AUC={lgbm_val_auc:.4f}")
    print(f"  LSTM  train AUC={lstm_train_auc:.4f}  val AUC={lstm_val_auc:.4f}")
    print(f"  calibrated theta = {theta:.3f}")
    print(f"  saved → {out_dir}")
    print("=" * 50)
    print("\n  LGBM feature importance (top 10):")
    print(lgbm.feature_importance().head(10).to_string(index=False))


if __name__ == "__main__":
    main()
