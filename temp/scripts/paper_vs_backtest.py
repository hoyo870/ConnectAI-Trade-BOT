"""
temp/scripts/paper_vs_backtest.py
paper 테스트 vs 백테스트 거래 단위 대조 (2026-06-25~07-03).

paper와 동일 설정(candidate 프리셋 + ML θ=0.45 + 3x)으로 같은 기간 백테스트 후,
paper_trades*.csv 와 백테스트 trade_log 를 진입/청산/PnL 단위로 비교.
순환성 없는 유일한 진짜 검증 = 백테스트가 실제 실행(paper)을 얼마나 정확히 예측했나.

Usage: .venv/bin/python temp/scripts/paper_vs_backtest.py
"""
import os, sys, csv
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "src")

import argparse
import pandas as pd
from strategies.backtest_engine import AntifragileBacktestRunner
from strategies.indicators import add_indicators_af
from config.af_params import get_preset, FEE_TOTAL, FUNDING_RATE_8H


def load_bybit_raw(coin):
    """data/bybit/ 의 Bybit CSV 로드 → datetime 인덱스 raw df (실거래/paper와 동일 소스)."""
    import glob
    files = sorted(glob.glob(str(ROOT / f"data/bybit/{coin.upper()}USDT_5m_*.csv")))
    if not files:
        return None
    frames = [pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp") for f in files]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index = df.index.tz_convert(None) if df.index.tz else df.index
    return df


def load_coin_source(runner, coin, source, start, end, warmup_bars=0):
    """source='binance'이면 runner.load_coin(기존 data/raw), 'bybit'이면 data/bybit 로드.
    warmup_bars>0이면 START 이전 지표 warmup을 그만큼으로 제한(live FETCH_LIMIT 모방)."""
    if source == "bybit":
        raw = load_bybit_raw(coin)
        if warmup_bars > 0:
            # START 직전 warmup_bars 개만 남겨 지표 계산 → live의 rolling 창과 동일 조건
            start_ts = pd.Timestamp(start)
            pre = raw[raw.index < start_ts]
            raw = pd.concat([pre.iloc[-warmup_bars:], raw[raw.index >= start_ts]])
        df = add_indicators_af(raw, 0)
        df_ml = runner._build_ml_context(raw)
        df = df[(df.index >= start) & (df.index < end)]
        df_ml = df_ml[(df_ml.index >= start) & (df_ml.index < end)]
        return df.copy(), df_ml.copy()
    return runner.load_coin(coin, start=start, end=end)


def paper_net(trades):
    """paper gross 거래에 백테스트와 동일 비용(왕복 수수료+funding) 적용 → net PnL 목록."""
    out = []
    for t in trades:
        cost = 2 * FEE_TOTAL + (t.get("hold_bars", 0) / 96.0) * FUNDING_RATE_8H
        net = max(t["pnl"] - cost * LEV * 0.1, -0.1)  # rr_base=0.1
        d = dict(t); d["pnl_net"] = net; out.append(d)
    return out

# paper 기간 (UTC). paper 시작 KST 06-25 11:03 = 02:03 UTC, 종료 07-04 00:35 KST = 07-03 15:35 UTC.
START, END = "2026-06-25 00:00", "2026-07-03 16:00"
LEV, THETA = 3, 0.45
COINS = ["btc", "eth", "sol", "xrp"]
PAPER_CSV = {"btc": "logs/paper_trades.csv", "eth": "logs/paper_trades_eth.csv",
             "sol": "logs/paper_trades_sol.csv", "xrp": "logs/paper_trades_xrp.csv"}


def load_paper(path):
    p = ROOT / path
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for r in csv.DictReader(f):
            out.append({
                "ts": pd.Timestamp(str(r["timestamp"]).replace(" UTC", "")),
                "dir": 1 if r["direction"] == "long" else -1,
                "entry": float(r["entry_price"]), "exit": float(r["exit_price"]),
                "pnl": float(r["pnl"]), "reason": r["reason"],
                "hold_bars": int(float(r.get("hold_bars", 0))),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="binance", choices=["binance", "bybit"],
                    help="백테스트 데이터 소스 (bybit=실거래/paper와 동일)")
    ap.add_argument("--warmup-bars", type=int, default=0,
                    help="백테스트 지표 warmup을 N봉으로 제한 (live FETCH_LIMIT=1000 모방). 0=full")
    args = ap.parse_args()

    params = {**get_preset("candidate"), "leverage": LEV}
    runner = AntifragileBacktestRunner.from_saved("models/af_ensemble/saved", params=params)
    runner.ensemble.threshold = THETA
    print(f"[paper vs backtest] {START}~{END} UTC | candidate θ={THETA} {LEV}x | 소스={args.source.upper()}"
          f" | warmup={'full' if args.warmup_bars==0 else args.warmup_bars}봉")
    print(f"  paper는 net(수수료 반영) 기준. (시작가치 코인당 $2500)\n")

    tot_paper_ret = tot_bt_ret = 0.0
    for coin in COINS:
        df, dfml = load_coin_source(runner, coin, args.source, START, END, args.warmup_bars)
        res = runner.run(df, dfml)
        bt = [t for t in res["trade_log"] if t.get("exit_ts") is not None]
        paper = paper_net(load_paper(PAPER_CSV[coin]))

        bt_ret = res["metrics"]["total_return"]
        paper_ret = sum(t["pnl_net"] for t in paper) * 100  # net PnL (수수료 반영)
        tot_bt_ret += bt_ret; tot_paper_ret += paper_ret

        print(f"── {coin.upper()} ──  paper(net) {len(paper)}거래 {paper_ret:+.3f}%  |  "
              f"backtest {len(bt)}거래 {bt_ret:+.3f}%")
        # 거래 나열 비교
        print(f"   {'PAPER':<44} {'BACKTEST'}")
        mx = max(len(paper), len(bt))
        for i in range(mx):
            ps = (f"{paper[i]['ts']:%m-%d %H:%M} {'L' if paper[i]['dir']==1 else 'S'} "
                  f"e{paper[i]['entry']:.2f}→x{paper[i]['exit']:.2f} {paper[i]['pnl_net']*100:+.3f}%"
                  if i < len(paper) else "—")
            bs = (f"{pd.Timestamp(bt[i]['exit_ts']):%m-%d %H:%M} "
                  f"{'L' if bt[i]['direction']==1 else 'S'} "
                  f"e{bt[i]['entry']:.2f}→x{bt[i]['exit']:.2f} {bt[i]['pnl']*100:+.3f}%"
                  if i < len(bt) else "—")
            print(f"   {ps:<44} {bs}")
        print()

    print(f"{'='*70}")
    print(f"  합산  paper {tot_paper_ret:+.3f}%  |  backtest {tot_bt_ret:+.3f}%  "
          f"|  차이 {tot_paper_ret - tot_bt_ret:+.3f}%p")
    print(f"  (paper 총자본: {10000*(1+tot_paper_ret/400):.2f} / $10,000)")


if __name__ == "__main__":
    main()
