"""
temp/scripts/keltner_reversal_2r.py
Keltner '극단 되돌림 + 2R + 추세 trailing' 전략. (사용자 아이디어, 2026-06-25)

롱 로직 (숏은 대칭):
  - 무장: 저가가 super_lower(3.0x) 하향 관통 → swing_low(관통 후 최저가) 추적.
  - 진입: 무장 상태에서 종가가 중심선(basis) 회복 시 → 다음봉 open 롱.
  - SL    = swing_low (이전 저점).
  - target= entry + 2*(entry - SL)   (2:1 RR)
  - trail : 상단밴드(upper) 돌파 시 trailing stop으로 전환(2R 너머 라이드). 미돌파면 2R에서 청산.
  - 단일진입, per-chunk 가중 회계, cost_mult, --validate.

기하학 주: upper(+~2ATR≈+0.67R)가 2R(+~6ATR)보다 먼저 닿으므로 대개 trail로 전환됨
(= 되돌림 진입 + 추세추종 trailing). 의도 그대로 구현, 검증으로 판정.

Usage:
  .venv/bin/python temp/scripts/keltner_reversal_2r.py --start 2026-01-01 --end 2026-06-01
  .venv/bin/python temp/scripts/keltner_reversal_2r.py --validate
  .venv/bin/python temp/scripts/keltner_reversal_2r.py --selftest
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
from keltner_super_bands import add_keltner, realize, _mdd, selftest

ENTRY_PCT = 0.50
RR = 2.0
TRAIL_ATR = 2.0


def run(df, leverage=1.0, initial_capital=1000.0, ma_type="SMA",
        rr=RR, trail_atr=TRAIL_ATR, cost_mult=1.0) -> dict:
    df = (add_keltner(df, ma_type)
          .dropna(subset=["basis", "rangema", "trend_ema"])
          .reset_index())
    df.rename(columns={"index": "_ts", "timestamp": "_ts"}, inplace=True)

    cap = initial_capital
    pos = 0
    chunks = []
    sl = tgt = peak = trail = None
    trailing = False
    # 무장 상태
    armed_long = armed_short = False
    swing_low = swing_high = None
    trades = []
    equity = [cap]

    def close_all(exit_px, reason):
        nonlocal cap, pos, chunks, sl, tgt, peak, trail, trailing
        if pos == 0:
            return
        ret = realize(chunks, exit_px, pos, cost_mult)
        cap *= 1 + ret
        trades.append({"dir": pos, "ret": ret, "reason": reason})
        pos = 0; chunks = []; sl = tgt = peak = trail = None; trailing = False

    for i in range(1, len(df)):
        row = df.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        prev = df.iloc[i - 1]
        atr = float(row["rangema"])
        upper, lower = float(row["upper"]), float(row["lower"])

        # ── 1) 포지션 관리 ──
        if pos == 1:
            if l <= sl:
                close_all(o if o <= sl else sl, "SL")
            else:
                peak = max(peak, h)
                if not trailing:
                    if h >= upper:                       # 상단 돌파 → trailing 전환
                        trailing = True; trail = peak - trail_atr * atr
                    elif h >= tgt:                       # 2R 도달(상단 미돌파)
                        close_all(o if o >= tgt else tgt, "2R")
                if trailing and pos == 1:
                    trail = max(trail, peak - trail_atr * atr)
                    if l <= trail:
                        close_all(o if o <= trail else trail, "trail")
        elif pos == -1:
            if h >= sl:
                close_all(o if o >= sl else sl, "SL")
            else:
                peak = min(peak, l)
                if not trailing:
                    if l <= lower:
                        trailing = True; trail = peak + trail_atr * atr
                    elif l <= tgt:
                        close_all(o if o <= tgt else tgt, "2R")
                if trailing and pos == -1:
                    trail = min(trail, peak + trail_atr * atr)
                    if h >= trail:
                        close_all(o if o >= trail else trail, "trail")

        # ── 2) 진입 (직전봉 기준) ──
        if pos == 0:
            pc, pbasis = float(prev["close"]), float(prev["basis"])
            fill = o; w = ENTRY_PCT * leverage
            if armed_long and pc > pbasis and swing_low is not None and swing_low < fill:
                pos = 1; chunks = [(fill, w)]; sl = swing_low
                tgt = fill + rr * (fill - sl); peak = h; trailing = False
                armed_long = False; swing_low = None
            elif armed_short and pc < pbasis and swing_high is not None and swing_high > fill:
                pos = -1; chunks = [(fill, w)]; sl = swing_high
                tgt = fill - rr * (sl - fill); peak = l; trailing = False
                armed_short = False; swing_high = None

        # ── 3) 무장 상태 갱신 (현재봉) ──
        if l < float(row["super_lower"]):
            armed_long = True
            swing_low = l if swing_low is None else min(swing_low, l)
        elif armed_long:
            swing_low = min(swing_low, l)
        if h > float(row["super_upper"]):
            armed_short = True
            swing_high = h if swing_high is None else max(swing_high, h)
        elif armed_short:
            swing_high = max(swing_high, h)

        equity.append(cap * (1 + realize(chunks, c, pos, cost_mult)) if pos != 0 else cap)

    if pos != 0:
        close_all(float(df.iloc[-1]["close"]), "end")

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
        "n_2r": sum(1 for t in trades if t["reason"] == "2R"),
        "n_trail": sum(1 for t in trades if t["reason"] == "trail"),
        "n_sl": sum(1 for t in trades if t["reason"] == "SL"),
        "trades": trades,
    }


def validate(leverage, ma_type):
    from strategies.backtest_engine import robust_metrics
    TRAIN = ("2023-01-01", "2025-01-01")
    TEST = ("2025-01-01", "2026-06-25")
    coins = ["btc", "eth", "sol", "xrp"]
    print(f"[validate] Keltner 되돌림+2R+trail | lev={leverage} | TRAIN {TRAIN[0]}~{TRAIN[1]} "
          f"→ held-out TEST {TEST[0]}~{TEST[1]}")
    for trail_atr in (1.5, 2.0, 3.0):
        print(f"\n{'='*100}\n  trail_atr={trail_atr}\n{'='*100}")
        print(f"  {'coin':<5} {'TRAIN수익':>10} {'TR거래':>6} | {'TEST수익':>9} {'TEST@2x':>9} "
              f"{'TE거래':>6} {'WR':>5} {'MDD':>6} {'avgW/avgL':>11} {'Sharpe':>7}  판정")
        print(f"  {'-'*98}")
        passes = 0
        for coin in coins:
            raw = load_coin_raw(coin)
            raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
            tr = raw[(raw.index >= TRAIN[0]) & (raw.index < TRAIN[1])]
            te = raw[(raw.index >= TEST[0]) & (raw.index < TEST[1])]
            if len(tr) < 300 or len(te) < 300:
                print(f"  {coin.upper():<5} 데이터 부족"); continue
            kw = dict(leverage=leverage, ma_type=ma_type, trail_atr=trail_atr)
            m_tr = run(tr, cost_mult=1.0, **kw)
            m_te = run(te, cost_mult=1.0, **kw)
            m_te2 = run(te, cost_mult=2.0, **kw)
            rm = robust_metrics([{"pnl": t["ret"]} for t in m_te["trades"]])
            ok = (m_te["total_return"] > 0 and m_te2["total_return"] > 0
                  and rm["sharpe"] > 0 and 0 < rm["pct_from_top10"] <= 100
                  and m_te["n_trades"] >= 20)
            passes += ok
            print(f"  {coin.upper():<5} {m_tr['total_return']:>+9.1f}% {m_tr['n_trades']:>6} | "
                  f"{m_te['total_return']:>+8.1f}% {m_te2['total_return']:>+8.1f}% {m_te['n_trades']:>6} "
                  f"{m_te['win_rate']:>4.0f}% {m_te['mdd']:>5.1f}% "
                  f"{m_te['avg_win']:>+4.2f}/{m_te['avg_loss']:>+4.2f} {rm['sharpe']:>+6.2f}  "
                  f"{'✅' if ok else '❌'}")
        print(f"  → TEST robust + 비용2배 통과: {passes}/{len(coins)} 코인")


def main():
    ap = argparse.ArgumentParser(description="Keltner 되돌림+2R+trail 백테스트")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--ma-type", default="SMA", choices=["SMA", "EMA"])
    ap.add_argument("--trail-atr", type=float, default=2.0)
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("회계 자기검증"); selftest(); return
    if args.validate:
        validate(args.leverage, args.ma_type); return

    for coin in ["btc", "eth", "sol", "xrp"]:
        raw = load_coin_raw(coin)
        raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
        df = raw[(raw.index >= args.start) & (raw.index < args.end)]
        if len(df) < 300:
            print(f"\n{coin.upper()}: 데이터 부족"); continue
        days = (df.index[-1] - df.index[0]).total_seconds() / 86400
        m = run(df, leverage=args.leverage, ma_type=args.ma_type,
                trail_atr=args.trail_atr, cost_mult=args.cost_mult)
        print(f"\n{'='*62}")
        print(f"  Keltner 되돌림+2R | {coin.upper()} {args.start}~{args.end} trail_atr={args.trail_atr} cost×{args.cost_mult}")
        print(f"{'='*62}")
        print(f"  거래수:   {m['n_trades']} (롱 {m['n_long']}/숏 {m['n_short']} | 2R {m['n_2r']}/trail {m['n_trail']}/SL {m['n_sl']})  TPD {m['n_trades']/days:.2f}")
        print(f"  총수익:   {m['total_return']:+.2f}%")
        print(f"  WR:       {m['win_rate']:.1f}%   MDD: {m['mdd']:.1f}%   PF: {m['profit_factor']:.3f}")
        print(f"  avg_win:  {m['avg_win']:+.3f}%   avg_loss: {m['avg_loss']:+.3f}%")


if __name__ == "__main__":
    main()
