"""
temp/scripts/keltner_breakout.py
STRAT_06 변형 — Keltner '역추세 슈퍼밴드 돌파(모멘텀)' 전략. (RSI 미사용)

아이디어:
  - SHORT: close > 200EMA(상승추세)인데 종가가 하단밴드(2.0x) 도달 → 하락 모멘텀 베팅.
           TP = 종가가 super_lower(3.0x) 이탈(추가 하락) 시 익절.
  - LONG : close < 200EMA(하락추세)인데 종가가 상단밴드(2.0x) 도달 → 상승 모멘텀 베팅.
           TP = 종가가 super_upper(3.0x) 이탈 시 익절.
  - 진입은 슈퍼밴드 '안쪽'에서만(TP 여지 확보), SL = 최초진입가 ±1%, 단일진입, per-chunk 회계.

기존 keltner_super_bands.py의 지표·회계 헬퍼 재사용.

Usage:
  .venv/bin/python temp/scripts/keltner_breakout.py --coin btc --start 2026-01-01 --end 2026-06-01
  .venv/bin/python temp/scripts/keltner_breakout.py --validate
  .venv/bin/python temp/scripts/keltner_breakout.py --selftest
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from config.loader import load_coin_raw
from config.af_params import FEE_TOTAL
from keltner_super_bands import add_keltner, realize, _mdd, selftest

SL_PCT = 0.01
ENTRY_PCT = 0.50


def run(df, leverage=1.0, initial_capital=1000.0, ma_type="SMA",
        min_width_pct=0.0, cost_mult=1.0) -> dict:
    df = (add_keltner(df, ma_type)
          .dropna(subset=["basis", "rangema", "trend_ema"])
          .reset_index())
    df.rename(columns={"index": "_ts", "timestamp": "_ts"}, inplace=True)

    cap = initial_capital
    pos = 0
    chunks = []
    first_px = None
    trades = []
    equity = [cap]

    def close_all(exit_px, ts, reason):
        nonlocal cap, pos, chunks, first_px
        if pos == 0:
            return
        ret = realize(chunks, exit_px, pos, cost_mult)
        cap *= 1 + ret
        trades.append({"dir": pos, "ret": ret, "reason": reason, "ts": ts})
        pos = 0; chunks = []; first_px = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        prev = df.iloc[i - 1]
        ts = row["_ts"]

        # ── 1) 포지션 관리 (SL 우선 → 슈퍼밴드 돌파 TP) ──
        if pos != 0:
            sl = first_px * (1 - SL_PCT) if pos == 1 else first_px * (1 + SL_PCT)
            hit_sl = (pos == 1 and l <= sl) or (pos == -1 and h >= sl)
            if pos == 1:   # 롱: super_upper 상향 돌파 익절
                tp_px = float(row["super_upper"]); tp_touch = h >= tp_px
            else:          # 숏: super_lower 하향 돌파 익절
                tp_px = float(row["super_lower"]); tp_touch = l <= tp_px

            if hit_sl:
                exec_sl = o if (pos == 1 and o <= sl) or (pos == -1 and o >= sl) else sl
                close_all(exec_sl, ts, "SL")
            elif tp_touch:
                exec_tp = o if (pos == 1 and o >= tp_px) or (pos == -1 and o <= tp_px) else tp_px
                close_all(exec_tp, ts, "TP")

        # ── 2) 진입 (직전 확정봉 기준, 슈퍼밴드 안쪽에서만) ──
        if pos == 0:
            pc = float(prev["close"])
            p_lo, p_up = float(prev["lower"]), float(prev["upper"])
            p_slo, p_sup = float(prev["super_lower"]), float(prev["super_upper"])
            p_ema = float(prev["trend_ema"])
            width_ok = ((p_up - p_lo) / p_lo) >= min_width_pct

            short_sig = (pc > p_ema) and (p_slo < pc <= p_lo) and width_ok
            long_sig = (pc < p_ema) and (p_up <= pc < p_sup) and width_ok

            fill = o
            w = ENTRY_PCT * leverage
            if short_sig:
                pos = -1; chunks = [(fill, w)]; first_px = fill
            elif long_sig:
                pos = 1; chunks = [(fill, w)]; first_px = fill

        equity.append(cap * (1 + realize(chunks, c, pos, cost_mult)) if pos != 0 else cap)

    if pos != 0:
        close_all(float(df.iloc[-1]["close"]), df.iloc[-1]["_ts"], "end")

    rets = [t["ret"] for t in trades]
    wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
    n = len(trades)
    return {
        "n_trades": n,
        "total_return": (cap - initial_capital) / initial_capital * 100,
        "win_rate": (len(wins) / n * 100 if n else 0),
        "mdd": _mdd(equity),
        "profit_factor": (abs(sum(wins) / sum(losses)) if losses and sum(losses) else float("inf")),
        "avg_win": (np.mean(wins) * 100 if wins else 0.0),
        "avg_loss": (np.mean(losses) * 100 if losses else 0.0),
        "n_long": sum(1 for t in trades if t["dir"] == 1),
        "n_short": sum(1 for t in trades if t["dir"] == -1),
        "n_tp": sum(1 for t in trades if t["reason"] == "TP"),
        "n_sl": sum(1 for t in trades if t["reason"] == "SL"),
        "trades": trades,
    }


def validate(leverage, ma_type, min_width):
    from strategies.backtest_engine import robust_metrics
    TRAIN = ("2023-01-01", "2025-01-01")
    TEST = ("2025-01-01", "2026-06-25")
    coins = ["btc", "eth", "sol", "xrp"]
    print(f"[validate] 역추세 슈퍼밴드 돌파 | lev={leverage} | TRAIN {TRAIN[0]}~{TRAIN[1]} "
          f"→ held-out TEST {TEST[0]}~{TEST[1]}")
    print(f"  {'coin':<5} {'TRAIN수익':>10} {'TR거래':>6} | {'TEST수익':>9} {'TEST@2x':>9} "
          f"{'TE거래':>6} {'WR':>5} {'MDD':>6} {'Sharpe':>7} {'top10%':>7}  판정")
    print(f"  {'-'*98}")
    passes = 0
    for coin in coins:
        raw = load_coin_raw(coin)
        raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
        tr = raw[(raw.index >= TRAIN[0]) & (raw.index < TRAIN[1])]
        te = raw[(raw.index >= TEST[0]) & (raw.index < TEST[1])]
        if len(tr) < 200 or len(te) < 200:
            print(f"  {coin.upper():<5} 데이터 부족"); continue
        kw = dict(leverage=leverage, ma_type=ma_type, min_width_pct=min_width)
        m_tr = run(tr, cost_mult=1.0, **kw)
        m_te = run(te, cost_mult=1.0, **kw)
        m_te2 = run(te, cost_mult=2.0, **kw)
        rm = robust_metrics([{"pnl": t["ret"]} for t in m_te["trades"]])
        ok = (m_te["total_return"] > 0 and m_te2["total_return"] > 0
              and rm["sharpe"] > 0 and 0 < rm["pct_from_top10"] <= 100 and m_te["n_trades"] >= 20)
        passes += ok
        print(f"  {coin.upper():<5} {m_tr['total_return']:>+9.1f}% {m_tr['n_trades']:>6} | "
              f"{m_te['total_return']:>+8.1f}% {m_te2['total_return']:>+8.1f}% {m_te['n_trades']:>6} "
              f"{m_te['win_rate']:>4.0f}% {m_te['mdd']:>5.1f}% {rm['sharpe']:>+6.2f} "
              f"{rm['pct_from_top10']:>6.0f}%  {'✅' if ok else '❌'}")
    print(f"  → TEST robust + 비용2배 통과: {passes}/{len(coins)} 코인")


def main():
    ap = argparse.ArgumentParser(description="Keltner 역추세 슈퍼밴드 돌파 백테스트")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--ma-type", default="SMA", choices=["SMA", "EMA"])
    ap.add_argument("--min-width", type=float, default=0.0)
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("회계 자기검증"); selftest(); return
    if args.validate:
        validate(args.leverage, args.ma_type, args.min_width); return

    for coin in ["btc", "eth", "sol", "xrp"]:
        raw = load_coin_raw(coin)
        raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
        df = raw[(raw.index >= args.start) & (raw.index < args.end)]
        if len(df) < 200:
            print(f"\n{coin.upper()}: 데이터 부족"); continue
        days = (df.index[-1] - df.index[0]).total_seconds() / 86400
        m = run(df, leverage=args.leverage, ma_type=args.ma_type,
                min_width_pct=args.min_width, cost_mult=args.cost_mult)
        print(f"\n{'='*60}")
        print(f"  Keltner 역추세 돌파 | {coin.upper()} {args.start}~{args.end} lev={args.leverage} cost×{args.cost_mult}")
        print(f"{'='*60}")
        print(f"  거래수:   {m['n_trades']} (롱 {m['n_long']}/숏 {m['n_short']} | TP {m['n_tp']}/SL {m['n_sl']})  TPD {m['n_trades']/days:.2f}")
        print(f"  총수익:   {m['total_return']:+.2f}%")
        print(f"  WR:       {m['win_rate']:.1f}%   MDD: {m['mdd']:.1f}%   PF: {m['profit_factor']:.3f}")
        print(f"  avg_win:  {m['avg_win']:+.3f}%   avg_loss: {m['avg_loss']:+.3f}%")


if __name__ == "__main__":
    main()
