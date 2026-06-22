"""
backtest_af_exact.py — live_trader.py 완벽 모방 백테스트 CLI

유일한 공식 백테스트 엔진: strategies/backtest_engine.AntifragileBacktestRunner
ML 앙상블 필터는 필수이며 비활성화 불가.

Usage:
  python scripts/backtest_af_exact.py --mode 2026
  python scripts/backtest_af_exact.py --mode 2026 --coin all
  python scripts/backtest_af_exact.py --mode hist --windows 10 --seed 42
  python scripts/backtest_af_exact.py --mode jun1819
"""

import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from config.loader import load_coin_raw
from strategies.indicators import add_indicators_af
from strategies.backtest_engine import AntifragileBacktestRunner

_DEFAULT_MODEL = str(ROOT / "models/af_ensemble/saved")
_BB_SIGMA = 0.5


# ── main ──────────────────────────────────────────────────────────────────────

HIST_START = {
    "btc": "2020-01-01", "eth": "2021-04-01",
    "sol": "2021-06-01", "xrp": "2020-06-01",
}


def main():
    parser = argparse.ArgumentParser(description="live_trader 완벽 모방 백테스트 (ML 필수)")
    parser.add_argument("--coin",        default="all", choices=["btc", "eth", "sol", "xrp", "all"])
    parser.add_argument("--mode",        default="2026", choices=["2026", "hist", "jun1819"])
    parser.add_argument("--windows",     type=int, default=10)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--window-days", type=int, default=91)
    parser.add_argument("--model",       default=_DEFAULT_MODEL,
                        help=f"AFEnsemble 저장 디렉터리 (기본: {_DEFAULT_MODEL})")
    args = parser.parse_args()

    runner = AntifragileBacktestRunner.from_saved(args.model)
    print(f"[ML 필터] 로드 완료: {args.model}  theta={runner.ensemble.threshold:.3f}")

    coins = ["btc", "eth", "sol", "xrp"] if args.coin == "all" else [args.coin]

    for coin in coins:
        label = coin.upper()
        print(f"\n{'█'*66}")
        print(f"  {label}/USDT — live_trader 완벽 모방 백테스트 [ML 필수]")
        print(f"  BB σ={_BB_SIGMA}  ML theta={runner.ensemble.threshold:.3f}")
        print(f"{'█'*66}")

        if args.mode == "2026":
            df, df_ml = runner.load_coin(coin, start="2026-01-01", end="2026-06-01")
            if len(df) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df)}봉)"); continue
            days = (df.index[-1] - df.index[0]).days
            print(f"\n[{label}] 2026 OOS: {df.index[0].date()} ~ {df.index[-1].date()}  ({days}일)")
            res = runner.run(df, df_ml)
            runner.print_result(f"{label} 2026 OOS", res, days)

        elif args.mode == "hist":
            hs = HIST_START.get(coin, "2020-01-01")
            print(f"\n[{label}] 랜덤 히스토리 검증")
            runner.run_hist_validation(coin, seed=args.seed, windows=args.windows,
                                       window_days=args.window_days, hist_start=hs)

        elif args.mode == "jun1819":
            df, df_ml = runner.load_coin(coin, start="2026-06-18 03:37", end="2026-06-19 12:00")
            if len(df) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df)}봉)"); continue
            days = (df.index[-1] - df.index[0]).total_seconds() / 86400
            print(f"\n[{label}] Jun 18~19 실거래 기간")
            res = runner.run(df, df_ml)
            runner.print_result(f"{label} Jun18~19", res, days)
            print(f"\n  상세 거래:")
            for t in res["trade_log"]:
                dr   = "롱" if t["direction"] == 1 else "숏"
                flip = " ◀FLIP" if t.get("reason") == "reverse_flip" else ""
                print(f"    {dr}  pnl={t['pnl']*100:+.3f}%  {t.get('reason','')}{flip}")


if __name__ == "__main__":
    main()
