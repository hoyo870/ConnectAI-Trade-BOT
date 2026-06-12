"""
48_trend_filter_test.py — 횡보 whipsaw 방지 필터 3종 비교

  Baseline : 기존 (필터 없음)
  Filter A  : ADX14 > 20  (추세 강도 필터)
  Filter B  : 1h EMA20 기울기 3봉 연속 일관성
  Filter C  : 1h + 4h EMA20 방향 일치 (MTF)

Usage:
  python temp/scripts/48_trend_filter_test.py --coin btc --mode both
  python temp/scripts/48_trend_filter_test.py --coin all  --mode 2026
"""
import sys, argparse, random
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path
from hybrid_engine import compute_metrics

ROOT        = Path(__file__).parent.parent.parent
TRADING_FEE = 0.0005
SLIPPAGE    = 0.0002
FEE_TOTAL   = TRADING_FEE + SLIPPAGE


# ─── 지표 ────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close = df["close"]; high = df["high"]; low = df["low"]

    # RSI
    d  = close.diff()
    ag = d.clip(lower=0).ewm(com=13, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["_rsi"] = 100 - 100 / (1 + ag / (al + 1e-9))

    # ATR
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    df["_atr"] = tr.ewm(span=14, adjust=False).mean()

    # 1h EMA20 — 기준 추세
    cl1h   = close.resample("1h").last().ffill()
    ema1h  = cl1h.ewm(span=20, adjust=False).mean()
    df["_trend_up"]   = (cl1h > ema1h).reindex(df.index, method="ffill").fillna(False).astype(int)
    df["_trend_down"] = (cl1h < ema1h).reindex(df.index, method="ffill").fillna(False).astype(int)

    # ── Filter A: ADX14 (Wilder smoothing) ───────────────────────────────────
    dm_p = (high - high.shift()).clip(lower=0)
    dm_m = (low.shift() - low).clip(lower=0)
    atr14   = tr.ewm(com=13, adjust=False).mean()
    dmp14   = dm_p.ewm(com=13, adjust=False).mean()
    dmm14   = dm_m.ewm(com=13, adjust=False).mean()
    di_p    = 100 * dmp14 / (atr14 + 1e-9)
    di_m    = 100 * dmm14 / (atr14 + 1e-9)
    dx      = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    df["_adx"] = dx.ewm(com=13, adjust=False).mean()

    # ── Filter B: 1h EMA20 기울기 (3봉 변화량) ───────────────────────────────
    slope3h = ema1h - ema1h.shift(3)
    df["_ema_slope"] = slope3h.reindex(df.index, method="ffill").fillna(0)

    # ── Filter C: 4h EMA20 방향 (MTF) ────────────────────────────────────────
    cl4h  = close.resample("4h").last().ffill()
    ema4h = cl4h.ewm(span=20, adjust=False).mean()
    df["_trend_up_4h"]   = (cl4h > ema4h).reindex(df.index, method="ffill").fillna(False).astype(int)
    df["_trend_down_4h"] = (cl4h < ema4h).reindex(df.index, method="ffill").fillna(False).astype(int)

    return df


# ─── 백테스트 엔진 ────────────────────────────────────────────────────────────

def run_af(
    df,
    initial_capital = 10_000.0,
    dt_rsi_lo=22, dt_rsi_hi=65,
    rg_rsi_lo=30, rg_rsi_hi=70,
    ut_rsi_lo=35, ut_rsi_hi=78,
    leverage=3,
    rr_base=0.10, rr_add=0.15,
    add_levels=3, atr_add_step=0.5,
    trail_atr_init=0.5, trail_atr_tight=0.8,
    max_hold_bars=288, cooling_bars=100, max_dd_cb=0.30,
    # 필터 파라미터
    filter_mode = "none",   # none | adx | slope | mtf
    adx_thr     = 20,       # Filter A
    slope_thr   = 0.0,      # Filter B (EMA slope 절댓값 기준, 0=부호만 확인)
    slope_bars  = 3,        # Filter B: 기울기 확인 봉 수 (현재 고정)
):
    df = df.reset_index(drop=True).copy()
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    df = df.reset_index(drop=True)

    capital = initial_capital
    peak_cap = initial_capital
    pos = 0; entry_price = 0.0; current_rr = 0.0
    add_count = 0; trail_sl = 0.0; peak_price = 0.0
    entry_bar = 0; cooling_left = 0; cb_triggers = 0

    equity_curve = [capital]
    trade_log    = []

    for idx in range(1, len(df)):
        row   = df.iloc[idx]
        price = float(row["close"])
        rsi   = float(row["_rsi"])
        atr   = float(row["_atr"])
        tup   = int(row.get("_trend_up",   0))
        tdn   = int(row.get("_trend_down", 0))

        rsi_lo = dt_rsi_lo if tdn else (ut_rsi_lo if tup else rg_rsi_lo)
        rsi_hi = dt_rsi_hi if tdn else (ut_rsi_hi if tup else rg_rsi_hi)

        # equity & circuit breaker
        if pos != 0:
            unr    = pos * (price - entry_price) / (entry_price + 1e-9)
            equity = capital * (1 + unr * leverage * current_rr)
        else:
            equity = capital
        peak_cap = max(peak_cap, equity)
        dd = (peak_cap - equity) / (peak_cap + 1e-9)

        if dd > max_dd_cb and cooling_left == 0:
            cooling_left = cooling_bars; cb_triggers += 1
            if pos != 0:
                cp  = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_bar,
                                   "lev": leverage, "rr": current_rr,
                                   "forced": True, "direction": pos})
                pos = 0

        if cooling_left > 0:
            cooling_left -= 1
            if pos != 0:
                cp  = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_bar,
                                   "lev": leverage, "rr": current_rr,
                                   "forced": True, "direction": pos})
                pos = 0
            equity_curve.append(capital); continue

        # 포지션 관리 (trailing stop + 피라미딩)
        if pos != 0:
            hold = idx - entry_bar
            if pos == 1:
                peak_price = max(peak_price, price)
                mult       = trail_atr_tight if add_count > 0 else trail_atr_init
                trail_sl   = max(trail_sl, peak_price - mult * atr)
                hit_stop   = price <= trail_sl
            else:
                peak_price = min(peak_price, price)
                mult       = trail_atr_tight if add_count > 0 else trail_atr_init
                trail_sl   = min(trail_sl, peak_price + mult * atr)
                hit_stop   = price >= trail_sl

            if hit_stop or hold >= max_hold_bars:
                cp  = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": hold,
                                   "lev": leverage, "rr": current_rr,
                                   "forced": False, "direction": pos})
                pos = 0; add_count = 0; current_rr = 0.0
            else:
                fav = pos * (price - entry_price) / (atr + 1e-9)
                if add_count < add_levels and fav >= (add_count + 1) * atr_add_step:
                    current_rr += rr_add; add_count += 1
                    if pos == 1: trail_sl = max(trail_sl, price - trail_atr_tight * atr)
                    else:        trail_sl = min(trail_sl, price + trail_atr_tight * atr)

        # 신규 진입
        if pos == 0:
            long_sig  = rsi <= rsi_lo
            short_sig = rsi >= rsi_hi

            # ── 필터 적용 ──────────────────────────────────────────────────
            if filter_mode == "adx":
                adx_val = float(row.get("_adx", 0))
                trend_ok_long  = adx_val >= adx_thr
                trend_ok_short = adx_val >= adx_thr

            elif filter_mode == "slope":
                slope = float(row.get("_ema_slope", 0))
                trend_ok_long  = slope > slope_thr
                trend_ok_short = slope < -slope_thr

            elif filter_mode == "mtf":
                tup4h = int(row.get("_trend_up_4h",   0))
                tdn4h = int(row.get("_trend_down_4h", 0))
                trend_ok_long  = (tup  == 1 and tup4h  == 1)
                trend_ok_short = (tdn  == 1 and tdn4h  == 1)

            else:  # none (baseline)
                trend_ok_long  = True
                trend_ok_short = True
            # ──────────────────────────────────────────────────────────────

            if long_sig and trend_ok_long:
                ep          = price * (1 + FEE_TOTAL)
                entry_price = ep; current_rr = rr_base; add_count = 0
                trail_sl    = ep - trail_atr_init * atr; peak_price = ep
                pos = 1; entry_bar = idx
            elif short_sig and trend_ok_short:
                ep          = price * (1 - FEE_TOTAL)
                entry_price = ep; current_rr = rr_base; add_count = 0
                trail_sl    = ep + trail_atr_init * atr; peak_price = ep
                pos = -1; entry_bar = idx

        equity_curve.append(capital)

    m = compute_metrics(equity_curve, trade_log)
    m["cb_triggers"] = cb_triggers
    if trade_log:
        days = len(df) / 288
        m["tpd"] = round(len(trade_log) / days, 2)
        wins  = [t["pnl"] for t in trade_log if t["pnl"] > 0]
        losss = [t["pnl"] for t in trade_log if t["pnl"] < 0]
        m["long_cnt"]  = sum(1 for t in trade_log if t.get("direction") ==  1)
        m["short_cnt"] = sum(1 for t in trade_log if t.get("direction") == -1)
        if wins  : m["avg_win"]  = round(float(np.mean(wins))  * 100, 4)
        if losss : m["avg_loss"] = round(float(np.mean(losss)) * 100, 4)
        if wins and losss:
            m["pf_ratio"] = round(abs(np.mean(wins) / np.mean(losss)), 3)
    return {"metrics": m, "trade_log": trade_log}


# ─── 리포트 ──────────────────────────────────────────────────────────────────

def remove_top_n(trades, n, base=10_000.0):
    top_idx = set(sorted(range(len(trades)), key=lambda i: trades[i]["pnl"], reverse=True)[:n])
    cap = base
    for i, t in enumerate(trades):
        if i not in top_idx:
            cap *= 1 + t["pnl"]
    return (cap / base - 1) * 100


def print_result(label, result, days):
    m  = result["metrics"]
    tl = result["trade_log"]
    tpd = m.get("tpd", round(len(tl) / max(days, 1), 2))
    r5  = remove_top_n(tl, 5)
    r3  = remove_top_n(tl, 3)
    ok_ret = m["total_return"] > 0
    ok_tpd = tpd >= 1.5
    ok_top = r5 > 0
    p = sum([ok_ret, ok_tpd, ok_top])
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  거래수:   {m['n_trades']}  (롱 {m.get('long_cnt',0)} / 숏 {m.get('short_cnt',0)})")
    print(f"  WR:       {m['win_rate']:.1f}%")
    print(f"  TPD:      {tpd:.2f}  {'✅' if ok_tpd else '❌'}")
    print(f"  수익률:   {m['total_return']:+.2f}%  {'✅' if ok_ret else '❌'}")
    print(f"  MDD:      {m['mdd']:.1f}%")
    print(f"  PF:       {m.get('profit_factor', 0):.3f}")
    print(f"  Top5제거: {r5:+.2f}%  {'✅' if ok_top else '❌'}")
    print(f"  Top3제거: {r3:+.2f}%")
    print(f"  판정:     {p}/3  {'✅ 통과' if p==3 else ('⚠️ 부분' if p>=2 else '❌ 탈락')}")
    return p


def run_hist(all_df, label, cfg, seed=42, windows=10, window_days=91, hist_start="2020-06-01"):
    all_df = all_df.dropna(subset=["_rsi", "_atr"])
    rng = random.Random(seed)
    possible = all_df[
        (all_df.index >= hist_start) &
        (all_df.index <= all_df.index[-1] - pd.Timedelta(days=window_days))
    ].index
    chosen = sorted(rng.choices(possible, k=windows))

    passes = []; returns = []
    for i, sd in enumerate(chosen):
        ed  = sd + pd.Timedelta(days=window_days)
        seg = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
        if len(seg) < 500: continue
        res = run_af(seg, **cfg)
        m   = res["metrics"]; tl = res["trade_log"]
        tpd = m.get("tpd", 0); r5 = remove_top_n(tl, 5); ret = m["total_return"]
        ok  = sum([ret > 0, tpd >= 1.5, r5 > 0])
        passes.append(ok); returns.append(ret)

    n3 = sum(p == 3 for p in passes)
    mark = "✅" if n3 >= 6 else "❌"
    return n3, np.mean(returns) if returns else 0.0, mark, passes, returns


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def _norm(df):
    df = df.copy()
    df.index = df.index.tz_convert(None) if df.index.tz else df.index
    return df


def load_ohlcv_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_index()


def load_coin(coin: str) -> pd.DataFrame:
    coin = coin.lower()
    if coin == "btc":
        pieces = []
        for f in ["data/raw/BTCUSDT_5m_20200101_20251231.csv",
                  "data/raw/BTCUSDT_5m_20260101_20260520.csv"]:
            try: pieces.append(_norm(load_ohlcv_csv(ROOT / f)))
            except Exception: pass
        if not pieces:
            par = pd.read_parquet(ROOT / "data/signals_2026/backtest_2026_signals.parquet")
            pieces.append(_norm(par[["open","high","low","close","volume"]].copy()))
    elif coin == "eth":
        pieces = [
            _norm(pd.read_parquet(ROOT / "data/eth/ETHUSDT_5m_history.parquet")),
            _norm(pd.read_parquet(ROOT / "data/eth/ETHUSDT_5m_2026.parquet")),
        ]
    else:
        sym = f"{coin.upper()}USDT"
        candidates = sorted((ROOT / "data/raw").glob(f"{sym}_5m_*.csv"))
        if not candidates:
            raise FileNotFoundError(f"{sym} 데이터 없음")
        pieces = [_norm(load_ohlcv_csv(f)) for f in candidates]

    df = pd.concat(pieces).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df["close"].notna() & (df["close"] > 0)]
    return add_indicators(df)


HIST_START = {"btc": "2020-01-01", "eth": "2021-04-01",
              "sol": "2021-06-01", "xrp": "2020-06-01"}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin",   default="btc",
                        choices=["btc","eth","sol","xrp","both","all"])
    parser.add_argument("--mode",   default="both", choices=["2026","random","both"])
    parser.add_argument("--windows",type=int, default=10)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--adx-thr",  type=float, default=20.0)
    parser.add_argument("--slope-thr",type=float, default=0.0)
    args = parser.parse_args()

    coin_map = {"both": ["btc","eth"], "all": ["btc","eth","sol","xrp"]}
    coins    = coin_map.get(args.coin, [args.coin])

    base_cfg = dict(leverage=3, rr_base=0.10, rr_add=0.15,
                    add_levels=3, atr_add_step=0.5,
                    trail_atr_init=0.5, trail_atr_tight=0.8)

    FILTERS = [
        ("Baseline", "none",  {}),
        ("A. ADX",   "adx",   {"adx_thr": args.adx_thr}),
        ("B. Slope", "slope", {"slope_thr": args.slope_thr}),
        ("C. MTF",   "mtf",   {}),
    ]

    for coin in coins:
        label = coin.upper()
        print(f"\n{'█'*66}")
        print(f"  {label}/USDT — 횡보 필터 3종 비교")
        print(f"  adx_thr={args.adx_thr}  slope_thr={args.slope_thr}")
        print(f"{'█'*66}")

        try:
            all_df = load_coin(coin)
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}"); continue

        # ── 2026 OOS ─────────────────────────────────────────────────────────
        if args.mode in ("2026", "both"):
            if coin == "btc":
                par  = pd.read_parquet(ROOT / "data/signals_2026/backtest_2026_signals.parquet")
                df26 = _norm(par[par.index >= "2026-01-01"].copy())
                df26 = add_indicators(df26)
            elif coin == "eth":
                df26 = _norm(pd.read_parquet(ROOT / "data/eth/ETHUSDT_5m_2026.parquet"))
                df26 = add_indicators(df26)
            else:
                df26 = all_df[all_df.index >= "2026-01-01"].copy()

            days = (df26.index[-1] - df26.index[0]).days
            print(f"\n── 2026 OOS ({df26.index[0].date()} ~ {df26.index[-1].date()}, {days}일) ──")
            print(f"  {'필터':<12}  {'거래':>5}  {'WR':>6}  {'TPD':>5}  "
                  f"{'수익률':>9}  {'MDD':>5}  {'PF':>6}  {'Top5':>8}  {'판정'}")
            print(f"  {'─'*72}")

            for fname, fmode, fextra in FILTERS:
                cfg = {**base_cfg, "filter_mode": fmode, **fextra}
                r   = run_af(df26, **cfg)
                m   = r["metrics"]; tl = r["trade_log"]
                tpd = m.get("tpd", 0); r5 = remove_top_n(tl, 5)
                ok  = sum([m["total_return"] > 0, tpd >= 1.5, r5 > 0])
                mk  = "✅" if ok == 3 else ("⚠️" if ok >= 2 else "❌")
                print(f"  {fname:<12}  {m['n_trades']:>5}  {m['win_rate']:>5.1f}%  "
                      f"{tpd:>4.2f}  {m['total_return']:>+8.1f}%  {m['mdd']:>4.1f}%  "
                      f"{m.get('profit_factor',0):>5.3f}  {r5:>+7.1f}%  {mk} ({ok}/3)")

        # ── hist 랜덤 10창 ────────────────────────────────────────────────────
        if args.mode in ("random", "both"):
            print(f"\n── hist 랜덤 {args.windows}창 (91일, seed={args.seed}) ──")
            print(f"  {'필터':<12}  {'통과':>6}  {'평균수익':>9}  {'수익양수':>8}  {'판정'}")
            print(f"  {'─'*46}")

            for fname, fmode, fextra in FILTERS:
                cfg = {**base_cfg, "filter_mode": fmode, **fextra}
                n3, avg_r, mark, passes, returns = run_hist(
                    all_df, label, cfg,
                    seed=args.seed, windows=args.windows, window_days=91,
                    hist_start=HIST_START.get(coin, "2020-01-01")
                )
                pos_cnt = sum(r > 0 for r in returns)
                print(f"  {fname:<12}  {n3:>3}/{args.windows}  "
                      f"{avg_r:>+8.1f}%  {pos_cnt:>4}/{args.windows}  {mark}")

    print()


if __name__ == "__main__":
    main()
