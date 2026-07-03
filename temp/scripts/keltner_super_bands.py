"""
temp/scripts/keltner_super_bands.py
STRAT_06 → Keltner Channel BREAKOUT 전략 (돌파매매). [전면 재작성 2026-06-25]

배경: 앞선 Keltner 변형(평균회귀/필터/모멘텀)은 전부 출구 기하학으로 죽었다 —
작은 고정 TP < −1% SL → avg_win < avg_loss → WR 높아도 음의 기대값(+수수료 출혈).
돌파매매는 이를 정반대로 뒤집는다: 승자를 끝까지 태우고(trailing/중심선 복귀까지) 손실은
빠르게 끊어 avg_win > avg_loss 를 확보한다.

전략:
  - 롱 진입: 종가가 상단밴드(upper) 상향 돌파 + close > 200EMA (추세 동조)
  - 숏 진입: 종가가 하단밴드(lower) 하향 돌파 + close < 200EMA
  - 출구 (--exit-mode):
      basis : 종가가 중심선(basis) 복귀 시 청산 (추세 종료) — 승자 라이드
      trail : ATR trailing stop (peak ∓ trail_atr×ATR) — 승자 라이드 + 손실 제한
  - 단일진입, per-chunk 가중 회계, cost_mult 비용 스트레스, --validate 정직검증.

Usage:
  .venv/bin/python temp/scripts/keltner_super_bands.py --start 2026-01-01 --end 2026-06-01
  .venv/bin/python temp/scripts/keltner_super_bands.py --exit-mode trail --trail-atr 2.5
  .venv/bin/python temp/scripts/keltner_super_bands.py --validate
  .venv/bin/python temp/scripts/keltner_super_bands.py --selftest
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from config.af_params import FEE_TOTAL
from config.loader import load_coin_raw

# ── 파라미터 ──────────────────────────────────────────────────────────────────
LEN = 20
MULT = 2.0
ATR_LEN = 14
ENTRY_PCT = 0.50          # 청크당 자본 비중
EMA_TREND_LEN = 200       # 추세 동조 필터용 EMA


# ── 지표 ──────────────────────────────────────────────────────────────────────
def _ma(s: pd.Series, length: int, kind: str = "SMA") -> pd.Series:
    if kind == "SMA":
        return s.rolling(length).mean()
    if kind == "EMA":
        return s.ewm(span=length, adjust=False).mean()
    raise ValueError(f"미지원 MA: {kind}")


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    """Pine ta.atr = RMA(TrueRange, length)."""
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


def add_keltner(df: pd.DataFrame, ma_type: str = "SMA", ma_trend_len: int = EMA_TREND_LEN) -> pd.DataFrame:
    df = df.copy()
    df["basis"] = _ma(df["close"], LEN, ma_type)
    df["rangema"] = _atr(df, ATR_LEN)
    df["upper"] = df["basis"] + df["rangema"] * MULT
    df["lower"] = df["basis"] - df["rangema"] * MULT
    df["super_upper"] = df["basis"] + df["rangema"] * (MULT * 1.5)
    df["super_lower"] = df["basis"] - df["rangema"] * (MULT * 1.5)
    df["test_rangema"] = _ma(_atr(df, 50), 14, ma_type)
    df["test_upper"] = _ma(df["basis"] + df["test_rangema"] * (MULT * 3.0), 14, ma_type)
    df["test_lower"] = _ma(df["basis"] - df["test_rangema"] * (MULT * 3.0), 14, ma_type)
    df["trend_ema"] = _ma(df["close"], ma_trend_len, "EMA")
    return df


# ── 회계: per-chunk 가중 실현손익 ──────────────────────────────────────────────
def realize(chunks, exit_px, direction, cost_mult: float = 1.0) -> float:
    r = 0.0
    for entry_px, w in chunks:
        gross = direction * (exit_px - entry_px) / entry_px
        r += w * (gross - 2 * FEE_TOTAL * cost_mult)
    return r


def _mdd(equity) -> float:
    eq = np.array(equity, dtype=float)
    if len(eq) == 0:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    return float(np.max((peaks - eq) / (peaks + 1e-9)) * 100)


# ── 백테스트 코어 (돌파 진입 + 승자 라이드 출구) ──────────────────────────────
def run(df, leverage=1.0, initial_capital=1000.0, ma_type="SMA",
        exit_mode="basis", trail_atr=2.0, use_super=False, cost_mult=1.0) -> dict:
    df = (add_keltner(df, ma_type)
          .dropna(subset=["basis", "rangema", "trend_ema"])
          .reset_index())
    df.rename(columns={"index": "_ts", "timestamp": "_ts"}, inplace=True)
    up_key = "super_upper" if use_super else "upper"
    lo_key = "super_lower" if use_super else "lower"

    cap = initial_capital
    pos = 0
    chunks = []
    trail = peak = None
    trades = []
    equity = [cap]

    def close_all(exit_px, ts, reason):
        nonlocal cap, pos, chunks, trail, peak
        if pos == 0:
            return
        ret = realize(chunks, exit_px, pos, cost_mult)
        cap *= 1 + ret
        trades.append({"dir": pos, "ret": ret, "reason": reason, "ts": ts})
        pos = 0; chunks = []; trail = peak = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        prev = df.iloc[i - 1]
        ts = row["_ts"]
        atr = float(row["rangema"]); basis = float(row["basis"])

        # ── 1) 포지션 관리 (승자 라이드 출구) ──
        if pos != 0:
            if exit_mode == "trail":
                if pos == 1:
                    peak = max(peak, h)
                    trail = max(trail, peak - trail_atr * atr)
                    if l <= trail:  # intrabar 체결 (갭하락이면 시가)
                        close_all(o if o <= trail else trail, ts, "trail")
                else:
                    peak = min(peak, l)
                    trail = min(trail, peak + trail_atr * atr)
                    if h >= trail:
                        close_all(o if o >= trail else trail, ts, "trail")
            else:  # basis: 종가가 중심선 복귀 시 청산
                if pos == 1 and c < basis:
                    close_all(c, ts, "basis")
                elif pos == -1 and c > basis:
                    close_all(c, ts, "basis")

        # ── 2) 돌파 진입 (직전 확정봉 기준, 추세 동조) ──
        if pos == 0:
            pc = float(prev["close"])
            p_up, p_lo = float(prev[up_key]), float(prev[lo_key])
            p_ema = float(prev["trend_ema"])
            long_sig = pc > p_up and pc > p_ema     # 상단 돌파 + 상승추세
            short_sig = pc < p_lo and pc < p_ema    # 하단 돌파 + 하락추세
            fill = o
            w = ENTRY_PCT * leverage
            if long_sig:
                pos = 1; chunks = [(fill, w)]; peak = h; trail = fill - trail_atr * atr
            elif short_sig:
                pos = -1; chunks = [(fill, w)]; peak = l; trail = fill + trail_atr * atr

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
        "trades": trades,
    }


# ── 회계 자기검증 ──────────────────────────────────────────────────────────────
def selftest():
    chunks = [(100.0, 0.5), (90.0, 0.5)]
    manual = 0.5 * (10 / 100) + 0.5 * (20 / 90)
    got = realize(chunks, 110.0, 1) + 2 * FEE_TOTAL * (0.5 + 0.5)
    assert abs(got - manual) < 1e-9, f"per-chunk 회계 불일치: {got} vs {manual}"
    single = realize([(100.0, 0.5)], 110.0, 1)
    simple = 0.5 * ((110 - 100) / 100 - 2 * FEE_TOTAL)
    assert abs(single - simple) < 1e-12, "단일진입 불일치"
    print("  [selftest] per-chunk 가중 회계 정확 + 단일진입 동등 → ✅ PASS")


# ── 정직 검증 (TRAIN/TEST 분리 + 비용 스트레스 + robust) ────────────────────────
def validate(leverage, ma_type, trail_atr):
    from strategies.backtest_engine import robust_metrics
    TRAIN = ("2023-01-01", "2025-01-01")
    TEST = ("2025-01-01", "2026-06-25")
    coins = ["btc", "eth", "sol", "xrp"]
    configs = [("basis", False), ("trail", False), ("basis", True), ("trail", True)]

    raws = {}
    for coin in coins:
        r = load_coin_raw(coin)
        r.index = r.index.tz_localize(None) if r.index.tz else r.index
        raws[coin] = r

    print(f"[validate] Keltner 돌파 | lev={leverage} trail_atr={trail_atr} | "
          f"TRAIN {TRAIN[0]}~{TRAIN[1]} → held-out TEST {TEST[0]}~{TEST[1]}")
    for exit_mode, use_super in configs:
        tag = f"{exit_mode}{'+super' if use_super else '+norm'}"
        print(f"\n{'='*98}\n  출구={tag}\n{'='*98}")
        print(f"  {'coin':<5} {'TRAIN수익':>10} {'TR거래':>6} | {'TEST수익':>9} {'TEST@2x':>9} "
              f"{'TE거래':>6} {'WR':>5} {'MDD':>6} {'avgW/avgL':>11} {'Sharpe':>7}  판정")
        print(f"  {'-'*96}")
        passes = 0
        for coin in coins:
            raw = raws[coin]
            tr = raw[(raw.index >= TRAIN[0]) & (raw.index < TRAIN[1])]
            te = raw[(raw.index >= TEST[0]) & (raw.index < TEST[1])]
            if len(tr) < 300 or len(te) < 300:
                print(f"  {coin.upper():<5} 데이터 부족"); continue
            kw = dict(leverage=leverage, ma_type=ma_type, exit_mode=exit_mode,
                      trail_atr=trail_atr, use_super=use_super)
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
    ap = argparse.ArgumentParser(description="Keltner Channel 돌파매매 백테스트")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--ma-type", default="SMA", choices=["SMA", "EMA"])
    ap.add_argument("--exit-mode", default="basis", choices=["basis", "trail"])
    ap.add_argument("--trail-atr", type=float, default=2.0)
    ap.add_argument("--use-super", action="store_true", help="돌파 기준을 슈퍼밴드(3.0x)로")
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("회계 자기검증"); selftest(); return
    if args.validate:
        validate(args.leverage, args.ma_type, args.trail_atr); return

    for coin in ["btc", "eth", "sol", "xrp"]:
        raw = load_coin_raw(coin)
        raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
        df = raw[(raw.index >= args.start) & (raw.index < args.end)]
        if len(df) < 300:
            print(f"\n{coin.upper()}: 데이터 부족"); continue
        days = (df.index[-1] - df.index[0]).total_seconds() / 86400
        m = run(df, leverage=args.leverage, ma_type=args.ma_type, exit_mode=args.exit_mode,
                trail_atr=args.trail_atr, use_super=args.use_super, cost_mult=args.cost_mult)
        print(f"\n{'='*62}")
        print(f"  Keltner 돌파 | {coin.upper()} {args.start}~{args.end} | "
              f"exit={args.exit_mode}{'+super' if args.use_super else ''} lev={args.leverage} cost×{args.cost_mult}")
        print(f"{'='*62}")
        print(f"  거래수:   {m['n_trades']} (롱 {m['n_long']}/숏 {m['n_short']})  TPD {m['n_trades']/days:.2f}")
        print(f"  총수익:   {m['total_return']:+.2f}%")
        print(f"  WR:       {m['win_rate']:.1f}%   MDD: {m['mdd']:.1f}%   PF: {m['profit_factor']:.3f}")
        print(f"  avg_win:  {m['avg_win']:+.3f}%   avg_loss: {m['avg_loss']:+.3f}%  (승자라이드 → avgW>avgL 기대)")


if __name__ == "__main__":
    main()
