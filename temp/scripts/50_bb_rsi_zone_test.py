"""
50_bb_rsi_zone_test.py — 1h BB 횡보 구역 + rg_rsi 둔감화 테스트

기존: price > 1h EMA20 → trendup, price < 1h EMA20 → trenddown (rg 거의 미사용)
개선: price > BB upper  → trendup
      price < BB lower  → trenddown
      BB 사이            → ranging → rg_rsi (20/80) 으로 진입 엄격화

스윕:
  BB_sigma  : 1.0 / 1.5 / 2.0 / 2.5
  rg_rsi    : (lo=18,hi=82) / (20,80) / (22,78) / (25,75)

Usage:
  python temp/scripts/50_bb_rsi_zone_test.py --coin btc --mode both
  python temp/scripts/50_bb_rsi_zone_test.py --coin all  --mode 2026
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

BB_SIGMAS   = [0.5, 1.0, 1.5, 2.0, 2.5]
RG_PAIRS    = [(18, 82), (20, 80), (22, 78), (25, 75)]


# ─── 지표 ─────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame, bb_sigma: float = 2.0) -> pd.DataFrame:
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

    # 1h EMA20 + BB
    cl1h   = close.resample("1h").last().ffill()
    ema1h  = cl1h.ewm(span=20, adjust=False).mean()
    std1h  = cl1h.rolling(20).std()
    bb_up  = ema1h + bb_sigma * std1h
    bb_lo  = ema1h - bb_sigma * std1h

    # BB 기준 3-state 추세
    df["_trend_up"]   = (cl1h > bb_up).reindex(df.index, method="ffill").fillna(False).astype(int)
    df["_trend_down"] = (cl1h < bb_lo).reindex(df.index, method="ffill").fillna(False).astype(int)
    # ranging = 둘 다 0

    # 비교용 — 기존 EMA 기준 (baseline)
    df["_trend_up_ema"]   = (cl1h > ema1h).reindex(df.index, method="ffill").fillna(False).astype(int)
    df["_trend_down_ema"] = (cl1h < ema1h).reindex(df.index, method="ffill").fillna(False).astype(int)

    return df


# ─── 백테스트 엔진 ────────────────────────────────────────────────────────────

def run_af(df, initial_capital=10_000.0,
           dt_rsi_lo=22, dt_rsi_hi=65,
           rg_rsi_lo=30, rg_rsi_hi=70,
           ut_rsi_lo=35, ut_rsi_hi=78,
           leverage=3, rr_base=0.10, rr_add=0.15,
           add_levels=3, atr_add_step=0.5,
           trail_atr_init=0.5, trail_atr_tight=0.8,
           max_hold_bars=288, cooling_bars=100, max_dd_cb=0.30,
           use_ema_baseline=False):  # True=기존 EMA 방식, False=BB 방식

    df = df.reset_index(drop=True).copy()
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    df = df.reset_index(drop=True)

    up_col  = "_trend_up_ema"  if use_ema_baseline else "_trend_up"
    dn_col  = "_trend_down_ema" if use_ema_baseline else "_trend_down"

    capital = initial_capital; peak_cap = initial_capital
    pos = 0; entry_price = 0.0; current_rr = 0.0
    add_count = 0; trail_sl = 0.0; peak_price = 0.0
    entry_bar = 0; cooling_left = 0

    equity_curve = [capital]
    trade_log    = []

    for idx in range(1, len(df)):
        row   = df.iloc[idx]
        price = float(row["close"])
        rsi   = float(row["_rsi"])
        atr   = float(row["_atr"])
        tup   = int(row.get(up_col, 0))
        tdn   = int(row.get(dn_col, 0))

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
            cooling_left = cooling_bars
            if pos != 0:
                cp  = price - FEE_TOTAL * price * pos
                pnl = max(pos * (cp - entry_price) / (entry_price + 1e-9) * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_bar,
                                   "lev": leverage, "rr": current_rr,
                                   "forced": True, "direction": pos})
                pos = 0

        if cooling_left > 0:
            cooling_left -= 1
            if pos != 0:
                cp  = price - FEE_TOTAL * price * pos
                pnl = max(pos * (cp - entry_price) / (entry_price + 1e-9) * leverage * current_rr, -current_rr)
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
                pnl = max(pos * (cp - entry_price) / (entry_price + 1e-9) * leverage * current_rr, -current_rr)
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
            if rsi <= rsi_lo:
                ep = price * (1 + FEE_TOTAL)
                entry_price = ep; current_rr = rr_base; add_count = 0
                trail_sl = ep - trail_atr_init * atr; peak_price = ep
                pos = 1; entry_bar = idx
            elif rsi >= rsi_hi:
                ep = price * (1 - FEE_TOTAL)
                entry_price = ep; current_rr = rr_base; add_count = 0
                trail_sl = ep + trail_atr_init * atr; peak_price = ep
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
    return sum([m["total_return"] > 0, m.get("tpd", 0) >= 1.5, remove_top5(tl) > 0])


def run_hist(all_df, cfg, seed=42, windows=10, window_days=91, hist_start="2020-01-01"):
    all_df = all_df.dropna(subset=["_rsi", "_atr"])
    rng    = random.Random(seed)
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
        m, tl = res["metrics"], res["trade_log"]
        passes.append(score(m, tl)); returns.append(m["total_return"])
    n3 = sum(p == 3 for p in passes)
    return n3, (np.mean(returns) if returns else 0.0), sum(r > 0 for r in returns)


# ─── 데이터 ──────────────────────────────────────────────────────────────────

def _norm(df):
    df = df.copy()
    df.index = df.index.tz_convert(None) if df.index.tz else df.index
    return df

def load_ohlcv_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.columns = [c.lower() for c in df.columns]
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()

def load_coin_raw(coin):
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
        sym   = f"{coin.upper()}USDT"
        cands = sorted((ROOT/"data/raw").glob(f"{sym}_5m_*.csv"))
        if not cands: raise FileNotFoundError(f"{sym} 데이터 없음")
        pieces = [_norm(load_ohlcv_csv(f)) for f in cands]
    df = pd.concat(pieces).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[df["close"].notna() & (df["close"] > 0)]

HIST_START = {"btc":"2020-01-01","eth":"2021-04-01","sol":"2021-06-01","xrp":"2020-06-01"}
BASE_CFG   = dict(leverage=3, rr_base=0.10, rr_add=0.15, add_levels=3, atr_add_step=0.5,
                  trail_atr_init=0.5, trail_atr_tight=0.8,
                  dt_rsi_lo=22, dt_rsi_hi=65, ut_rsi_lo=35, ut_rsi_hi=78)


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
        print(f"\n{'█'*72}")
        print(f"  {label}/USDT — 1h BB 횡보구역 + rg_rsi 둔감화 스윕")
        print(f"  dt_rsi=22/65  ut_rsi=35/78  (고정)")
        print(f"{'█'*72}")

        try:
            raw_df = load_coin_raw(coin)
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}"); continue

        if args.mode in ("2026","both"):
            if coin == "btc":
                par  = pd.read_parquet(ROOT/"data/signals_2026/backtest_2026_signals.parquet")
                raw26 = _norm(par[par.index >= "2026-01-01"][["open","high","low","close","volume"]].copy())
            elif coin == "eth":
                raw26 = _norm(pd.read_parquet(ROOT/"data/eth/ETHUSDT_5m_2026.parquet"))
            else:
                raw26 = raw_df[raw_df.index >= "2026-01-01"].copy()

            days = (raw26.index[-1] - raw26.index[0]).days
            print(f"\n── 2026 OOS ({raw26.index[0].date()}~{raw26.index[-1].date()}, {days}일) ──")

            HDR = (f"  {'설정':<26} {'거래':>5} {'rg비율':>7} {'TPD':>5} "
                   f"{'수익률':>9} {'MDD':>5} {'PF':>6} {'Top5':>8} {'판정'}")
            print(HDR); print("  " + "─" * 74)

            def row26(label, df26, cfg, rg_lo, rg_hi):
                res = run_af(df26, **cfg, rg_rsi_lo=rg_lo, rg_rsi_hi=rg_hi)
                m, tl = res["metrics"], res["trade_log"]
                tpd = m.get("tpd", 0); r5 = remove_top5(tl)
                ok  = score(m, tl)
                mk  = "✅" if ok==3 else ("⚠️" if ok>=2 else "❌")
                # rg 비율: ranging 구간 거래 수 추정 (진입 시 tup=0,tdn=0인 것)
                print(f"  {label:<26} {m['n_trades']:>5}  {'':>6} {tpd:>4.2f} "
                      f"{m['total_return']:>+8.1f}% {m['mdd']:>4.1f}% "
                      f"{m.get('profit_factor',0):>5.3f} {r5:>+7.1f}% {mk}({ok}/3)")
                return ok

            # Baseline (기존 EMA 방식, rg=30/70)
            df26_base = add_indicators(raw26, bb_sigma=2.0)
            row26("Baseline (EMA rg=30/70)", df26_base,
                  {**BASE_CFG, "use_ema_baseline": True}, 30, 70)
            print()

            for sigma in BB_SIGMAS:
                df26_s = add_indicators(raw26, bb_sigma=sigma)
                for rg_lo, rg_hi in RG_PAIRS:
                    lbl = f"BB{sigma:.1f}σ  rg={rg_lo}/{rg_hi}"
                    row26(lbl, df26_s, {**BASE_CFG, "use_ema_baseline": False}, rg_lo, rg_hi)
                print()

        if args.mode in ("random","both"):
            print(f"\n── hist 랜덤 {args.windows}창 (91일, seed={args.seed}) ──")
            print(f"  {'설정':<26} {'통과':>6} {'평균수익':>9} {'수익양수':>8} {'판정'}")
            print(f"  {'─'*56}")

            def hrow(label, all_df_s, cfg, rg_lo, rg_hi):
                n3, avg_r, pos_cnt = run_hist(
                    all_df_s,
                    {**cfg, "rg_rsi_lo": rg_lo, "rg_rsi_hi": rg_hi},
                    seed=args.seed, windows=args.windows,
                    hist_start=HIST_START.get(coin,"2020-01-01"))
                mk = "✅" if n3 >= 6 else "❌"
                print(f"  {label:<26} {n3:>3}/{args.windows} {avg_r:>+8.1f}% "
                      f"{pos_cnt:>4}/{args.windows}  {mk}")
                return n3, avg_r

            raw_base = add_indicators(raw_df, bb_sigma=2.0)
            hrow("Baseline (EMA rg=30/70)", raw_base,
                 {**BASE_CFG, "use_ema_baseline": True}, 30, 70)
            print()

            for sigma in BB_SIGMAS:
                raw_s = add_indicators(raw_df, bb_sigma=sigma)
                for rg_lo, rg_hi in RG_PAIRS:
                    lbl = f"BB{sigma:.1f}σ  rg={rg_lo}/{rg_hi}"
                    hrow(lbl, raw_s, {**BASE_CFG, "use_ema_baseline": False}, rg_lo, rg_hi)
                print()

    print()


if __name__ == "__main__":
    main()
