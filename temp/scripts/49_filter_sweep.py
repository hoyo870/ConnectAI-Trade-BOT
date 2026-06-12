"""
49_filter_sweep.py — 횡보 필터 파라미터 스윕

  A. ADX    : threshold 15/18/20/22/25/28/30
  B. Slope  : 1h EMA 기울기 lookback 1/2/3/5/8/12 h
  C. MTF    : 상위 TF 2h/3h/4h/6h
  D. B+C    : Slope + MTF 조합 (best 파라미터)

Usage:
  python temp/scripts/49_filter_sweep.py --coin btc --mode both
  python temp/scripts/49_filter_sweep.py --coin all  --mode 2026
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

ADX_THRS   = [15, 18, 20, 22, 25, 28, 30]
SLOPE_LBK  = [1, 2, 3, 5, 8, 12]      # hours
MTF_TFS    = ["2h", "3h", "4h", "6h"]


# ─── 지표 ─────────────────────────────────────────────────────────────────────

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
    cl1h  = close.resample("1h").last().ffill()
    ema1h = cl1h.ewm(span=20, adjust=False).mean()
    df["_trend_up"]   = (cl1h > ema1h).reindex(df.index, method="ffill").fillna(False).astype(int)
    df["_trend_down"] = (cl1h < ema1h).reindex(df.index, method="ffill").fillna(False).astype(int)

    # ── A: ADX14 ──────────────────────────────────────────────────────────────
    dm_p  = (high - high.shift()).clip(lower=0)
    dm_m  = (low.shift() - low).clip(lower=0)
    atr14 = tr.ewm(com=13, adjust=False).mean()
    di_p  = 100 * dm_p.ewm(com=13, adjust=False).mean() / (atr14 + 1e-9)
    di_m  = 100 * dm_m.ewm(com=13, adjust=False).mean() / (atr14 + 1e-9)
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    df["_adx"] = dx.ewm(com=13, adjust=False).mean()

    # ── B: Slope (다양한 lookback) ─────────────────────────────────────────────
    for lb in SLOPE_LBK:
        slope = ema1h - ema1h.shift(lb)
        df[f"_slope_{lb}h"] = slope.reindex(df.index, method="ffill").fillna(0)

    # ── C: MTF EMA20 (다양한 TF) ──────────────────────────────────────────────
    for tf in MTF_TFS:
        cl_tf  = close.resample(tf).last().ffill()
        ema_tf = cl_tf.ewm(span=20, adjust=False).mean()
        df[f"_trend_up_{tf}"]   = (cl_tf > ema_tf).reindex(df.index, method="ffill").fillna(False).astype(int)
        df[f"_trend_down_{tf}"] = (cl_tf < ema_tf).reindex(df.index, method="ffill").fillna(False).astype(int)

    return df


# ─── 백테스트 엔진 ────────────────────────────────────────────────────────────

def run_af(df, initial_capital=10_000.0,
           dt_rsi_lo=22, dt_rsi_hi=65, rg_rsi_lo=30, rg_rsi_hi=70,
           ut_rsi_lo=35, ut_rsi_hi=78,
           leverage=3, rr_base=0.10, rr_add=0.15,
           add_levels=3, atr_add_step=0.5,
           trail_atr_init=0.5, trail_atr_tight=0.8,
           max_hold_bars=288, cooling_bars=100, max_dd_cb=0.30,
           filter_mode="none",
           adx_thr=20, slope_lb=3, mtf_tf="4h",
           slope_col=None, mtf_up_col=None, mtf_dn_col=None):

    df = df.reset_index(drop=True).copy()
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    df = df.reset_index(drop=True)

    # 열 이름 미리 결정
    _slope_col  = slope_col  or f"_slope_{slope_lb}h"
    _mtf_up_col = mtf_up_col or f"_trend_up_{mtf_tf}"
    _mtf_dn_col = mtf_dn_col or f"_trend_down_{mtf_tf}"

    capital = initial_capital; peak_cap = initial_capital
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
                pnl = max(pos * (cp - entry_price) / (entry_price + 1e-9) * leverage * current_rr,
                          -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_bar,
                                   "lev": leverage, "rr": current_rr,
                                   "forced": True, "direction": pos})
                pos = 0

        if cooling_left > 0:
            cooling_left -= 1
            if pos != 0:
                cp  = price - FEE_TOTAL * price * pos
                pnl = max(pos * (cp - entry_price) / (entry_price + 1e-9) * leverage * current_rr,
                          -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_bar,
                                   "lev": leverage, "rr": current_rr,
                                   "forced": True, "direction": pos})
                pos = 0
            equity_curve.append(capital); continue

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
                pnl = max(pos * (cp - entry_price) / (entry_price + 1e-9) * leverage * current_rr,
                          -current_rr)
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

        if pos == 0:
            long_sig  = rsi <= rsi_lo
            short_sig = rsi >= rsi_hi

            if filter_mode == "adx":
                adx_val = float(row.get("_adx", 0))
                ok_l = ok_s = adx_val >= adx_thr

            elif filter_mode == "slope":
                slope = float(row.get(_slope_col, 0))
                ok_l  = slope > 0
                ok_s  = slope < 0

            elif filter_mode == "mtf":
                tup_h = int(row.get(_mtf_up_col, 0))
                tdn_h = int(row.get(_mtf_dn_col, 0))
                ok_l  = (tup == 1 and tup_h == 1)
                ok_s  = (tdn == 1 and tdn_h == 1)

            elif filter_mode == "bc":   # Slope + MTF 조합
                slope  = float(row.get(_slope_col, 0))
                tup_h  = int(row.get(_mtf_up_col, 0))
                tdn_h  = int(row.get(_mtf_dn_col, 0))
                ok_l   = (slope > 0) and (tup == 1 and tup_h == 1)
                ok_s   = (slope < 0) and (tdn == 1 and tdn_h == 1)

            else:
                ok_l = ok_s = True

            if long_sig and ok_l:
                ep          = price * (1 + FEE_TOTAL)
                entry_price = ep; current_rr = rr_base; add_count = 0
                trail_sl    = ep - trail_atr_init * atr; peak_price = ep
                pos = 1; entry_bar = idx
            elif short_sig and ok_s:
                ep          = price * (1 - FEE_TOTAL)
                entry_price = ep; current_rr = rr_base; add_count = 0
                trail_sl    = ep + trail_atr_init * atr; peak_price = ep
                pos = -1; entry_bar = idx

        equity_curve.append(capital)

    m = compute_metrics(equity_curve, trade_log)
    if trade_log:
        days = len(df) / 288
        m["tpd"] = round(len(trade_log) / days, 2)
        wins  = [t["pnl"] for t in trade_log if t["pnl"] > 0]
        losss = [t["pnl"] for t in trade_log if t["pnl"] < 0]
        if wins and losss:
            m["pf_ratio"] = round(abs(np.mean(wins) / np.mean(losss)), 3)
    return {"metrics": m, "trade_log": trade_log}


# ─── 유틸 ────────────────────────────────────────────────────────────────────

def remove_top5(trades, base=10_000.0):
    top_idx = set(sorted(range(len(trades)), key=lambda i: trades[i]["pnl"], reverse=True)[:5])
    cap = base
    for i, t in enumerate(trades):
        if i not in top_idx: cap *= 1 + t["pnl"]
    return (cap / base - 1) * 100


def score(m, tl):
    tpd = m.get("tpd", 0)
    r5  = remove_top5(tl)
    return sum([m["total_return"] > 0, tpd >= 1.5, r5 > 0])


def run_hist(all_df, cfg, seed=42, windows=10, window_days=91, hist_start="2020-01-01"):
    all_df = all_df.dropna(subset=["_rsi", "_atr"])
    rng = random.Random(seed)
    possible = all_df[
        (all_df.index >= hist_start) &
        (all_df.index <= all_df.index[-1] - pd.Timedelta(days=window_days))
    ].index
    chosen = sorted(rng.choices(possible, k=windows))
    passes, returns = [], []
    for sd in chosen:
        ed  = sd + pd.Timedelta(days=window_days)
        seg = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
        if len(seg) < 500: continue
        res = run_af(seg, **cfg)
        m   = res["metrics"]; tl = res["trade_log"]
        passes.append(score(m, tl)); returns.append(m["total_return"])
    n3 = sum(p == 3 for p in passes)
    return n3, np.mean(returns) if returns else 0.0, sum(r > 0 for r in returns)


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def _norm(df):
    df = df.copy()
    df.index = df.index.tz_convert(None) if df.index.tz else df.index
    return df

def load_ohlcv_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.columns = [c.lower() for c in df.columns]
    for col in ["open","high","low","close","volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_index()

def load_coin(coin):
    coin = coin.lower()
    if coin == "btc":
        pieces = []
        for f in ["data/raw/BTCUSDT_5m_20200101_20251231.csv",
                  "data/raw/BTCUSDT_5m_20260101_20260520.csv"]:
            try: pieces.append(_norm(load_ohlcv_csv(ROOT / f)))
            except: pass
        if not pieces:
            par = pd.read_parquet(ROOT/"data/signals_2026/backtest_2026_signals.parquet")
            pieces.append(_norm(par[["open","high","low","close","volume"]].copy()))
    elif coin == "eth":
        pieces = [_norm(pd.read_parquet(ROOT/"data/eth/ETHUSDT_5m_history.parquet")),
                  _norm(pd.read_parquet(ROOT/"data/eth/ETHUSDT_5m_2026.parquet"))]
    else:
        sym = f"{coin.upper()}USDT"
        cands = sorted((ROOT/"data/raw").glob(f"{sym}_5m_*.csv"))
        if not cands: raise FileNotFoundError(f"{sym} 데이터 없음")
        pieces = [_norm(load_ohlcv_csv(f)) for f in cands]
    df = pd.concat(pieces).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df["close"].notna() & (df["close"] > 0)]
    return add_indicators(df)

HIST_START = {"btc":"2020-01-01","eth":"2021-04-01","sol":"2021-06-01","xrp":"2020-06-01"}
BASE_CFG   = dict(leverage=3, rr_base=0.10, rr_add=0.15, add_levels=3, atr_add_step=0.5,
                  trail_atr_init=0.5, trail_atr_tight=0.8)


# ─── 스윕 실행 ────────────────────────────────────────────────────────────────

HDR = f"  {'설정':<20} {'거래':>5} {'TPD':>5} {'수익률':>9} {'MDD':>5} {'PF':>6} {'Top5':>8} {'판정':>4}"
SEP = "  " + "─" * 66

def print_row(label, m, tl, days):
    tpd = m.get("tpd", round(len(tl)/max(days,1), 2))
    r5  = remove_top5(tl)
    ok  = score(m, tl)
    mk  = "✅" if ok==3 else ("⚠️" if ok>=2 else "❌")
    print(f"  {label:<20} {m['n_trades']:>5} {tpd:>4.2f} {m['total_return']:>+8.1f}% "
          f"{m['mdd']:>4.1f}% {m.get('profit_factor',0):>5.3f} {r5:>+7.1f}% {mk}({ok}/3)")
    return ok, tpd, m["total_return"], m["mdd"], m.get("profit_factor",0), r5


def sweep_2026(df26, days, mode_label):
    print(f"\n{'─'*70}")
    print(f"  {mode_label}")
    print(HDR); print(SEP)

    results = {}

    # Baseline
    r = run_af(df26, **BASE_CFG, filter_mode="none")
    m, tl = r["metrics"], r["trade_log"]
    ok, tpd, ret, mdd, pf, r5 = print_row("Baseline", m, tl, days)
    results["Baseline"] = (ok, tpd, ret, mdd, pf, r5)

    print()
    # A: ADX sweep
    best_adx = None
    for thr in ADX_THRS:
        r = run_af(df26, **BASE_CFG, filter_mode="adx", adx_thr=thr)
        m, tl = r["metrics"], r["trade_log"]
        label = f"A.ADX≥{thr}"
        ok, tpd, ret, mdd, pf, r5 = print_row(label, m, tl, days)
        if best_adx is None or pf > best_adx[1]: best_adx = (thr, pf, ok)
        results[label] = (ok, tpd, ret, mdd, pf, r5)

    print()
    # B: Slope sweep
    best_slope = None
    for lb in SLOPE_LBK:
        r = run_af(df26, **BASE_CFG, filter_mode="slope", slope_lb=lb)
        m, tl = r["metrics"], r["trade_log"]
        label = f"B.Slope {lb}h"
        ok, tpd, ret, mdd, pf, r5 = print_row(label, m, tl, days)
        if best_slope is None or pf > best_slope[1]: best_slope = (lb, pf, ok)
        results[label] = (ok, tpd, ret, mdd, pf, r5)

    print()
    # C: MTF sweep
    best_mtf = None
    for tf in MTF_TFS:
        r = run_af(df26, **BASE_CFG, filter_mode="mtf", mtf_tf=tf)
        m, tl = r["metrics"], r["trade_log"]
        label = f"C.MTF {tf}"
        ok, tpd, ret, mdd, pf, r5 = print_row(label, m, tl, days)
        if best_mtf is None or pf > best_mtf[1]: best_mtf = (tf, pf, ok)
        results[label] = (ok, tpd, ret, mdd, pf, r5)

    print()
    # D: B+C 조합 (best 파라미터)
    if best_slope and best_mtf:
        for lb in [best_slope[0], max(1, best_slope[0]-1)]:
            for tf in [best_mtf[0]]:
                r = run_af(df26, **BASE_CFG, filter_mode="bc", slope_lb=lb, mtf_tf=tf)
                m, tl = r["metrics"], r["trade_log"]
                label = f"D.B+C {lb}h+{tf}"
                ok, tpd, ret, mdd, pf, r5 = print_row(label, m, tl, days)
                results[label] = (ok, tpd, ret, mdd, pf, r5)

    return results, best_adx, best_slope, best_mtf


def sweep_hist(all_df, hist_start, seed, windows):
    print(f"\n  hist 랜덤 {windows}창 (91일, seed={seed})")
    print(f"  {'설정':<20} {'통과':>6} {'평균수익':>9} {'수익양수':>8} {'판정'}")
    print(f"  {'─'*50}")

    def hrow(label, cfg):
        n3, avg_r, pos_cnt = run_hist(all_df, cfg, seed=seed,
                                       windows=windows, hist_start=hist_start)
        mk = "✅" if n3 >= 6 else "❌"
        print(f"  {label:<20} {n3:>3}/{windows} {avg_r:>+8.1f}% {pos_cnt:>4}/{windows}  {mk}")
        return n3, avg_r

    hrow("Baseline",    {**BASE_CFG, "filter_mode":"none"})
    print()
    best_adx_h, best_slope_h, best_mtf_h = None, None, None
    for thr in ADX_THRS:
        n3, avg_r = hrow(f"A.ADX≥{thr}", {**BASE_CFG, "filter_mode":"adx","adx_thr":thr})
        if best_adx_h is None or n3 > best_adx_h[1]: best_adx_h = (thr, n3, avg_r)
    print()
    for lb in SLOPE_LBK:
        n3, avg_r = hrow(f"B.Slope {lb}h", {**BASE_CFG, "filter_mode":"slope","slope_lb":lb})
        if best_slope_h is None or n3 > best_slope_h[1]: best_slope_h = (lb, n3, avg_r)
    print()
    for tf in MTF_TFS:
        n3, avg_r = hrow(f"C.MTF {tf}", {**BASE_CFG, "filter_mode":"mtf","mtf_tf":tf})
        if best_mtf_h is None or n3 > best_mtf_h[1]: best_mtf_h = (tf, n3, avg_r)
    print()
    if best_slope_h and best_mtf_h:
        for lb in [best_slope_h[0]]:
            for tf in [best_mtf_h[0]]:
                hrow(f"D.B+C {lb}h+{tf}", {**BASE_CFG,"filter_mode":"bc","slope_lb":lb,"mtf_tf":tf})


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin",   default="btc",
                        choices=["btc","eth","sol","xrp","both","all"])
    parser.add_argument("--mode",   default="both", choices=["2026","random","both"])
    parser.add_argument("--windows",type=int, default=10)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    coin_map = {"both":["btc","eth"], "all":["btc","eth","sol","xrp"]}
    coins    = coin_map.get(args.coin, [args.coin])

    for coin in coins:
        label = coin.upper()
        print(f"\n{'█'*70}")
        print(f"  {label}/USDT — 횡보 필터 파라미터 스윕")
        print(f"{'█'*70}")

        try:
            all_df = load_coin(coin)
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}"); continue

        if args.mode in ("2026", "both"):
            if coin == "btc":
                par  = pd.read_parquet(ROOT/"data/signals_2026/backtest_2026_signals.parquet")
                df26 = _norm(par[par.index >= "2026-01-01"].copy())
                df26 = add_indicators(df26)
            elif coin == "eth":
                df26 = _norm(pd.read_parquet(ROOT/"data/eth/ETHUSDT_5m_2026.parquet"))
                df26 = add_indicators(df26)
            else:
                df26 = all_df[all_df.index >= "2026-01-01"].copy()

            days = (df26.index[-1] - df26.index[0]).days
            sweep_2026(df26, days, f"{label} 2026 OOS ({df26.index[0].date()}~{df26.index[-1].date()}, {days}일)")

        if args.mode in ("random", "both"):
            sweep_hist(all_df, HIST_START.get(coin,"2020-01-01"), args.seed, args.windows)

    print()


if __name__ == "__main__":
    main()
