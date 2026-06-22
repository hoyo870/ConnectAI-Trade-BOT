"""
ML 라벨 생성 — AntifragileStrategy를 "피처 로깅 모드"로 실행하여
진입 시점 스냅샷/시퀀스 + 실현 PnL 라벨을 CSV/NPZ로 출력.

live_trader.py와 동일한 로직(AntifragileStrategy + bb_sigma=0.5 트렌드)을 사용하므로
라벨이 실거래 결과를 정확히 반영함.

Usage:
  python scripts/generate_ml_labels.py --coins all --out data/ml_labels/
  python scripts/generate_ml_labels.py --coins btc --train-end 2024-12-31
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, "src")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.af_params import FEE_TOTAL, DEFAULT_PARAMS
from config.loader import load_coin_raw
from strategies.antifragile import AntifragileStrategy
from models.af_ensemble.feature_extractor import (
    extract_snapshot, extract_sequence, SNAPSHOT_COLS,
)
from scripts.backtest_af_exact import build_ml_context_df


def generate_labels_for_coin(
    df_ml: pd.DataFrame,
    coin: str,
    leverage: int = 5,
    seq_len: int = 20,
    **kwargs,
) -> tuple:
    """
    AntifragileStrategy 기반 라벨 생성 — live_trader.py 완벽 모방.

    df_ml: build_ml_context_df() 결과 (bb_sigma=0.5 트렌드 + 2σ BB + ML 피처)
    반환: (snapshot_records, X_seq, y)
    """
    p = {**DEFAULT_PARAMS, **kwargs, "leverage": leverage}
    lev = leverage
    strategy = AntifragileStrategy(p)

    snapshot_records = []
    seq_list = []
    y_list = []
    pending = None          # 미결 진입 {snapshot, sequence, direction, date}
    partial_pnl_accum = 0.0

    for idx in range(1, len(df_ml)):
        row      = df_ml.iloc[idx]
        price    = float(row["close"])
        atr      = float(row["_atr"])
        rsi      = float(row["_rsi"])
        trend_up = bool(row["_trend_up"])
        trend_dn = bool(row["_trend_down"])

        result = strategy.process_tick(price, atr, rsi, trend_up, trend_dn)

        for event in result["events"]:
            etype = event["type"]

            if etype == "partial" and pending is not None:
                # 절반 익절 PnL 누적 (최종 청산 시 합산)
                ppnl = max(event["pnl_raw"] * lev * event["rr"], -event["rr"])
                partial_pnl_accum += ppnl

            elif etype == "close":
                if pending is not None:
                    final_pnl = max(event["pnl_raw"] * lev * event["rr"], -event["rr"]) - FEE_TOTAL
                    total_pnl = partial_pnl_accum + final_pnl
                    label = 1 if total_pnl > 0.001 else 0
                    rec = dict(pending["snapshot"])
                    rec.update({"pnl": total_pnl, "label": label,
                                "coin": coin, "date": pending["date"]})
                    snapshot_records.append(rec)
                    seq_list.append(pending["sequence"])
                    y_list.append(label)
                pending = None
                partial_pnl_accum = 0.0

            elif etype == "entry":
                direction = event["direction"]
                # 진입 시점의 RSI 임계값 재현 (process_tick이 이미 계산한 값과 동일)
                trend_str = AntifragileStrategy._trend_str(trend_up, trend_dn)
                rsi_lo_t, rsi_hi_t = strategy._get_thresholds(trend_str)
                loss_streak = 0  # 방향 쿨링 제거 후 고정값
                snap = extract_snapshot(df_ml, idx, direction, rsi_lo_t, rsi_hi_t, loss_streak)
                seq  = extract_sequence(df_ml, idx, seq_len)
                pending = {
                    "snapshot":  snap,
                    "sequence":  seq,
                    "direction": direction,
                    "date":      str(df_ml.index[idx]),
                }
                partial_pnl_accum = 0.0

    X_seq = np.array(seq_list,  dtype=np.float32) if seq_list  else np.zeros((0, seq_len, 7), dtype=np.float32)
    y     = np.array(y_list,    dtype=np.int64)   if y_list    else np.zeros((0,),            dtype=np.int64)
    return snapshot_records, X_seq, y


def main():
    parser = argparse.ArgumentParser(description="Generate ML labels (AntifragileStrategy 기반)")
    parser.add_argument("--coins",     default="all")
    parser.add_argument("--train-end", default=None, help="이 날짜 이하 데이터만 사용 (YYYY-MM-DD)")
    parser.add_argument("--out",       default="data/ml_labels/")
    parser.add_argument("--leverage",  type=int, default=5)
    args = parser.parse_args()

    coin_map = {"all": ["btc", "eth", "sol", "xrp"], "both": ["btc", "eth"]}
    coins = coin_map.get(args.coins, [args.coins])

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for coin in coins:
        print(f"{coin.upper()} 데이터 로드 중...")
        try:
            raw = load_coin_raw(coin)
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}")
            continue

        if args.train_end:
            raw = raw[raw.index <= args.train_end]

        print(f"  {raw.index[0].date()} ~ {raw.index[-1].date()}  ({len(raw):,}행)")
        df_ml = build_ml_context_df(raw)

        snap_records, X_seq, y = generate_labels_for_coin(
            df_ml, coin, leverage=args.leverage
        )
        if not snap_records:
            print(f"  ⚠️  {coin}: 거래 0건")
            continue

        snap_df = pd.DataFrame(snap_records, columns=SNAPSHOT_COLS + ["pnl", "label", "coin", "date"])
        snap_path = out_dir / f"{coin}_snapshot.csv"
        seq_path  = out_dir / f"{coin}_sequences.npz"
        snap_df.to_csv(snap_path, index=False)
        np.savez_compressed(seq_path, X=X_seq, y=y)

        n_trades = len(snap_df)
        win_rate = (snap_df["pnl"] > 0).mean() * 100
        pos_rate = snap_df["label"].mean() * 100
        print(f"  {coin.upper()}: n_trades={n_trades}  win_rate={win_rate:.1f}%  "
              f"label_pos_rate={pos_rate:.1f}%  → {snap_path.name}, {seq_path.name}")


if __name__ == "__main__":
    main()
