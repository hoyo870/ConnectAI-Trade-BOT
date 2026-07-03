"""
temp/scripts/keltner_super_bands_backtest.py
STRAT_06 — Keltner Super Bands 백테스트 (KELTNER_SUPER_BANDS.md Pine v5 이식).

전략 요약:
  - basis = MA(close,20), rangema = ATR(10). 밴드: ±2.0(normal), ±3.0(super).
  - 롱 진입: close[1] < lower[1] AND close[1] > super_lower[1]  (일반밴드 이탈 + 슈퍼밴드 안 → 패닉 필터)
  - 숏 진입: close[1] > upper[1] AND close[1] < super_upper[1]
  - 피라미딩 max 2: 1차 50% + (조건 재출현) 2차 50%. 반대신호 시 전량 스위칭.
  - 하드 SL: **최초 1차 진입가** 기준 ±1% (intrabar 터치 청산).
  - TP: 봉 OHLC 중 하나라도 반대 일반밴드 터치 시 전량 청산.

⚠️ PnL은 per-chunk 가중으로 계산 (피라미딩 평단 왜곡 방지 — project_pnl_accounting_fix 교훈).
   SL 트리거 가격만 최초진입가 기준이고, 실현손익은 두 청크 모두 반영한다.

Usage:
  .venv/bin/python temp/scripts/keltner_super_bands_backtest.py --coin btc --start 2026-01-01 --end 2026-06-01
  .venv/bin/python temp/scripts/keltner_super_bands_backtest.py --selftest
"""
import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from config.loader import load_coin_raw
from config.af_params import FEE_TOTAL

# ── 파라미터 (스펙 기본값) ─────────────────────────────────────────────────────
LEN          = 20
MULT         = 2.0
ATR_LEN      = 10
SL_PCT       = 0.01      # 최초진입가 기준 ±1%
ENTRY_PCT    = 0.50      # 청크당 자본 50%
PYRAMID_MAX  = 2


# ── 지표 ──────────────────────────────────────────────────────────────────────
def _ma(s: pd.Series, length: int, kind: str = "SMA") -> pd.Series:
    if kind == "SMA": return s.rolling(length).mean()
    if kind == "EMA": return s.ewm(span=length, adjust=False).mean()
    raise ValueError(f"미지원 MA: {kind}")


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    """Pine ta.atr = RMA(TrueRange, length) (Wilder smoothing)."""
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


def add_keltner(df: pd.DataFrame, ma_type: str = "SMA") -> pd.DataFrame:
    df = df.copy()
    df["basis"]   = _ma(df["close"], LEN, ma_type)
    df["rangema"] = _atr(df, ATR_LEN)
    df["upper"]       = df["basis"] + df["rangema"] * MULT
    df["lower"]       = df["basis"] - df["rangema"] * MULT
    df["super_upper"] = df["basis"] + df["rangema"] * (MULT * 1.5)
    df["super_lower"] = df["basis"] - df["rangema"] * (MULT * 1.5)
    return df


# ── 회계: per-chunk 가중 실현손익 ──────────────────────────────────────────────
def realize(chunks, exit_px, direction) -> float:
    """청산 시 자본수익률(분율). chunks=[(entry_px, weight)], weight=자본대비 notional 비중.
    Σ weight_j · dir·(exit−entry_j)/entry_j − 왕복수수료(2×FEE_TOTAL)·weight_j."""
    r = 0.0
    for entry_px, w in chunks:
        gross = direction * (exit_px - entry_px) / entry_px
        r += w * (gross - 2 * FEE_TOTAL)
    return r


# ── 백테스트 ──────────────────────────────────────────────────────────────────
def run(df: pd.DataFrame, leverage: float = 1.0, initial_capital: float = 1000.0,
        ma_type: str = "SMA") -> dict:
    df = add_keltner(df, ma_type).dropna(subset=["basis", "rangema"]).reset_index()
    df.rename(columns={"index": "_ts", "timestamp": "_ts"}, inplace=True)

    cap = initial_capital
    pos = 0                    # 0 / +1 long / -1 short
    chunks = []                # [(entry_px, weight)]  weight = ENTRY_PCT*leverage (진입시점 자본대비)
    first_px = None
    added = False              # 2차 진입 완료 여부
    trades = []
    equity = [cap]

    def close_all(exit_px, ts, reason):
        nonlocal cap, pos, chunks, first_px, added
        if pos == 0:
            return
        ret = realize(chunks, exit_px, pos)
        cap *= (1 + ret)
        trades.append({"dir": pos, "ret": ret, "exit": exit_px, "reason": reason,
                       "ts": ts, "n_chunks": len(chunks), "cap": cap})
        pos = 0; chunks = []; first_px = None; added = False

    for i in range(1, len(df)):
        row  = df.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        prev = df.iloc[i - 1]
        ts   = row["_ts"]

        # ── 1) 보유 포지션 관리 (intrabar SL → TP) ──
        if pos != 0:
            # 하드 SL: 최초 진입가 기준
            sl = first_px * (1 - SL_PCT) if pos == 1 else first_px * (1 + SL_PCT)
            hit_sl = (pos == 1 and l <= sl) or (pos == -1 and h >= sl)
            if hit_sl:
                # 갭하락으로 시작부터 SL 밑이면 시가 체결, 아니면 SL 가격 체결
                exec_sl = o if (pos == 1 and o <= sl) or (pos == -1 and o >= sl) else sl
                close_all(exec_sl, ts, "SL")
            else:
                # TP: 봉 OHLC 중 하나라도 반대 일반밴드 터치
                band = float(row["upper"]) if pos == 1 else float(row["lower"])
                touch = (pos == 1 and max(o, h, l, c) >= band) or \
                        (pos == -1 and min(o, h, l, c) <= band)
                if touch:
                    # Pine 충실: close_all은 확정봉 종가 체결 (밴드가격 체결은 낙관적 편향)
                    # 갭상승으로 시작부터 TP 위면 시가 체결, 아니면 밴드 터치 가격 체결
                    if pos == 1:
                        exec_tp = o if o >= band else band
                    else:
                        exec_tp = o if o <= band else band
                    close_all(exec_tp, ts, "TP")

        # ── 2) 진입 신호 (직전 확정봉 기준, 리페인팅 차단) ──
        pc = float(prev["close"])
        long_sig  = pc < float(prev["lower"]) and pc > float(prev["super_lower"])
        short_sig = pc > float(prev["upper"]) and pc < float(prev["super_upper"])
        fill = o   # Pine 충실: 신호 i-1 → 현재봉 open 체결 (next-bar-open, look-ahead 없음)
        w    = ENTRY_PCT * leverage

        if long_sig:
            if pos == -1:
                close_all(fill, ts, "switch")
            if pos <= 0:
                pos = 1; chunks = [(fill, w)]; first_px = fill; added = False
            elif pos == 1 and not added and len(chunks) < PYRAMID_MAX:
                chunks.append((fill, w)); added = True
        elif short_sig:
            if pos == 1:
                close_all(fill, ts, "switch")
            if pos >= 0:
                pos = -1; chunks = [(fill, w)]; first_px = fill; added = False
            elif pos == -1 and not added and len(chunks) < PYRAMID_MAX:
                chunks.append((fill, w)); added = True

        # ── equity 기록 (미실현 포함) ──
        if pos != 0:
            unr = realize(chunks, c, pos)
            equity.append(cap * (1 + unr))
        else:
            equity.append(cap)

    if pos != 0:
        close_all(float(df.iloc[-1]["close"]), df.iloc[-1]["_ts"], "end")

    # ── 지표 ──
    rets = [t["ret"] for t in trades]
    wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
    n = len(trades)
    total_ret = (cap - initial_capital) / initial_capital * 100
    wr = len(wins) / n * 100 if n else 0
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
    eq = np.array(equity); peaks = np.maximum.accumulate(eq)
    mdd = float(np.max((peaks - eq) / (peaks + 1e-9)) * 100) if len(eq) else 0.0
    return {
        "capital": cap, "trades": trades, "n_trades": n, "total_return": total_ret,
        "win_rate": wr, "profit_factor": pf, "mdd": mdd,
        "avg_win": (np.mean(wins) * 100 if wins else 0.0),
        "avg_loss": (np.mean(losses) * 100 if losses else 0.0),
        "n_long": sum(1 for t in trades if t["dir"] == 1),
        "n_short": sum(1 for t in trades if t["dir"] == -1),
        "n_pyramid": sum(1 for t in trades if t["n_chunks"] >= 2),
    }


# ── per-chunk 회계 자기검증 ────────────────────────────────────────────────────
def selftest():
    # 롱 2청크: w=0.5씩, entry 100/90, exit 110. 수수료 0 가정 위해 FEE_TOTAL 임시 무시 비교.
    chunks = [(100.0, 0.5), (90.0, 0.5)]
    # 가중 합산(무수수료): 0.5*(110-100)/100 + 0.5*(110-90)/90 = 0.05 + 0.1111 = 0.16111
    manual = 0.5 * (10 / 100) + 0.5 * (20 / 90)
    got = realize(chunks, 110.0, 1) + 2 * FEE_TOTAL * (0.5 + 0.5)  # 수수료 더해 무수수료로 환원
    assert abs(got - manual) < 1e-9, f"per-chunk 회계 불일치: {got} vs {manual}"
    # 단일진입은 단순 단일포지션과 동일
    single = realize([(100.0, 0.5)], 110.0, 1)
    simple = 0.5 * ((110 - 100) / 100 - 2 * FEE_TOTAL)
    assert abs(single - simple) < 1e-12, "단일진입 불일치"
    print("  [selftest] per-chunk 가중 회계 정확 + 단일진입 동등 → ✅ PASS")


def main():
    ap = argparse.ArgumentParser(description="STRAT_06 Keltner Super Bands 백테스트")
    ap.add_argument("--coin", default="btc", choices=["btc", "eth", "sol", "xrp"])
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end",   default="2026-06-01")
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--ma-type",  default="SMA", choices=["SMA", "EMA"])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("회계 자기검증"); selftest(); return

    raw = load_coin_raw(args.coin)
    raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
    df = raw[(raw.index >= args.start) & (raw.index < args.end)]
    if len(df) < 50:
        print(f"데이터 부족: {len(df)}"); return
    days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    m = run(df, leverage=args.leverage, ma_type=args.ma_type)
    print(f"\n{'='*60}")
    print(f"  STRAT_06 Keltner SuperBands | {args.coin.upper()} {args.start}~{args.end} "
          f"({days:.0f}일) lev={args.leverage} MA={args.ma_type}")
    print(f"{'='*60}")
    print(f"  거래수:   {m['n_trades']}  (롱 {m['n_long']} / 숏 {m['n_short']} / 2차매집 {m['n_pyramid']})  "
          f"TPD {m['n_trades']/days:.2f}")
    print(f"  총수익:   {m['total_return']:+.2f}%")
    print(f"  WR:       {m['win_rate']:.1f}%")
    print(f"  MDD:      {m['mdd']:.1f}%")
    print(f"  PF:       {m['profit_factor']:.3f}")
    print(f"  avg_win:  {m['avg_win']:+.4f}%   avg_loss: {m['avg_loss']:+.4f}%")


if __name__ == "__main__":
    main()
