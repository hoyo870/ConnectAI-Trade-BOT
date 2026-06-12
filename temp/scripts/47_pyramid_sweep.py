"""
피라미딩 횟수(add_levels) 스윕

고정: SL=S2(1.0/1.2), RSI=R0/R1/R2 각각 테스트
변수: add_levels = 0, 1, 2, 3, 4
      atr_add_step = 0.5 (고정)

Usage:
  python temp/scripts/47_pyramid_sweep.py               # 2026 OOS
  python temp/scripts/47_pyramid_sweep.py --hist         # + hist 상위 5개
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

RSI_TARGETS = {
    "R0_기준":  dict(dt_rsi_lo=22, dt_rsi_hi=65, rg_rsi_lo=30, rg_rsi_hi=70, ut_rsi_lo=35, ut_rsi_hi=78),
    "R1_안정":  dict(dt_rsi_lo=25, dt_rsi_hi=65, rg_rsi_lo=30, rg_rsi_hi=70, ut_rsi_lo=38, ut_rsi_hi=75),
    "R2_고수익": dict(dt_rsi_lo=28, dt_rsi_hi=62, rg_rsi_lo=32, rg_rsi_hi=68, ut_rsi_lo=40, ut_rsi_hi=72),
}
SL_FIXED = dict(trail_atr_init=1.0, trail_atr_tight=1.2)
PYRAMID_LEVELS = [0, 1, 2, 3, 4]

COIN_CONFIG = {
    "btc": {"hist_start": "2020-01-01"},
    "eth": {"hist_start": "2021-04-01"},
    "sol": {"hist_start": "2021-06-01"},
    "xrp": {"hist_start": "2020-06-01"},
}
COINS = ["btc", "eth", "sol", "xrp"]


def add_indicators(df):
    df = df.copy()
    close = df["close"]; high = df["high"]; low = df["low"]
    delta = close.diff()
    ag = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["_rsi"] = 100 - 100 / (1 + ag / (al + 1e-9))
    tr = pd.concat([high - low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    df["_atr"] = tr.ewm(span=14, adjust=False).mean()
    cl1h   = close.resample("1h").last().ffill()
    ema_1h = cl1h.ewm(span=20, adjust=False).mean()
    df["_trend_up"]   = (cl1h > ema_1h).reindex(df.index, method="ffill").fillna(False).astype(int)
    df["_trend_down"] = (cl1h < ema_1h).reindex(df.index, method="ffill").fillna(False).astype(int)
    return df


def _normalize_index(df):
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


def load_coin_full(coin):
    coin = coin.lower()
    if coin == "btc":
        pieces = []
        for fname in ["data/raw/BTCUSDT_5m_20200101_20251231.csv",
                      "data/raw/BTCUSDT_5m_20260101_20260520.csv"]:
            try: pieces.append(_normalize_index(load_ohlcv_csv(ROOT / fname)))
            except: pass
        if not pieces:
            par = pd.read_parquet(ROOT / "data/signals_2026/backtest_2026_signals.parquet")
            pieces.append(_normalize_index(par[["open","high","low","close","volume"]].copy()))
    elif coin == "eth":
        pieces = [
            _normalize_index(pd.read_parquet(ROOT / "data/eth/ETHUSDT_5m_history.parquet")),
            _normalize_index(pd.read_parquet(ROOT / "data/eth/ETHUSDT_5m_2026.parquet")),
        ]
    else:
        sym = f"{coin.upper()}USDT"
        candidates = sorted((ROOT / "data/raw").glob(f"{sym}_5m_*.csv"))
        if not candidates:
            raise FileNotFoundError(f"{sym} 데이터 없음")
        pieces = [_normalize_index(load_ohlcv_csv(f)) for f in candidates]
    all_df = pd.concat(pieces).sort_index()
    all_df = all_df[~all_df.index.duplicated(keep="last")]
    all_df = all_df[all_df["close"].notna() & (all_df["close"] > 0)]
    return add_indicators(all_df)


def run_antifragile(df,
    initial_capital=10_000.0,
    dt_rsi_lo=22, dt_rsi_hi=65, rg_rsi_lo=30, rg_rsi_hi=70, ut_rsi_lo=35, ut_rsi_hi=78,
    leverage=3, rr_base=0.10, rr_add=0.15,
    add_levels=3, atr_add_step=0.5,
    trail_atr_init=1.0, trail_atr_tight=1.2,
    max_hold_bars=288, cooling_bars=100, max_dd_cb=0.30,
):
    df = df.reset_index(drop=True)
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    df = df.reset_index(drop=True)
    capital = initial_capital; peak_cap = initial_capital
    pos = 0; entry_price = 0.0; current_rr = 0.0
    add_count = 0; trail_sl = 0.0; peak_price = 0.0
    entry_bar = 0; cooling_left = 0; cb_triggers = 0
    equity_curve = [capital]; trade_log = []

    for idx in range(1, len(df)):
        row   = df.iloc[idx]
        price = float(row["close"])
        rsi   = float(row["_rsi"])
        atr   = float(row["_atr"])
        tup   = int(row.get("_trend_up", 0))
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
                cp = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "forced": True, "direction": pos})
                pos = 0

        if cooling_left > 0:
            cooling_left -= 1
            if pos != 0:
                cp = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "forced": True, "direction": pos})
                pos = 0
            equity_curve.append(capital); continue

        if pos != 0:
            hold = idx - entry_bar
            if pos == 1:
                peak_price = max(peak_price, price)
                trail_mult = trail_atr_tight if add_count > 0 else trail_atr_init
                trail_sl   = max(trail_sl, peak_price - trail_mult * atr)
                hit_stop   = price <= trail_sl
            else:
                peak_price = min(peak_price, price)
                trail_mult = trail_atr_tight if add_count > 0 else trail_atr_init
                trail_sl   = min(trail_sl, peak_price + trail_mult * atr)
                hit_stop   = price >= trail_sl
            if hit_stop or hold >= max_hold_bars:
                cp = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": hold,
                                   "lev": leverage, "rr": current_rr, "forced": False, "direction": pos})
                pos = 0; add_count = 0; current_rr = 0.0
            else:
                favorable_move = pos * (price - entry_price) / (atr + 1e-9)
                next_add_level = (add_count + 1) * atr_add_step
                if add_count < add_levels and favorable_move >= next_add_level:
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
        m["tpd"] = round(len(trade_log) / (len(df) / 288), 2)
    return {"metrics": m, "trade_log": trade_log}


def remove_top_n(trades, n, base=10_000.0):
    top_idx = set(sorted(range(len(trades)), key=lambda i: trades[i]["pnl"], reverse=True)[:n])
    cap = base
    for i, t in enumerate(trades):
        if i in top_idx: continue
        cap *= 1 + t["pnl"]
    return (cap / base - 1) * 100


def run_hist(all_df, cfg, seed=42, windows=10, window_days=91, hist_start=None):
    all_df = all_df.dropna(subset=["_rsi", "_atr"])
    rng = random.Random(seed)
    min_start = hist_start or "2020-06-01"
    possible = all_df[(all_df.index >= min_start) &
                      (all_df.index <= all_df.index[-1] - pd.Timedelta(days=window_days))].index
    chosen = sorted(rng.choices(possible, k=windows))
    passes = []; returns = []
    for sd in chosen:
        ed  = sd + pd.Timedelta(days=window_days)
        seg = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
        if len(seg) < 500: continue
        res = run_antifragile(seg, **cfg)
        m   = res["metrics"]; tl = res["trade_log"]
        tpd = m.get("tpd", 0); r5 = remove_top_n(tl, 5); ret = m["total_return"]
        ok  = sum([ret > 0, tpd >= 1.5, r5 > 0])
        passes.append(ok); returns.append(ret)
    n3 = sum(p == 3 for p in passes)
    return {"pass3": n3, "total": len(passes), "avg_ret": np.mean(returns) if returns else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist", action="store_true")
    parser.add_argument("--hist-top", type=int, default=5)
    args = parser.parse_args()

    print("데이터 로드 중...", flush=True)
    coin_data = {}
    for coin in COINS:
        try:
            coin_data[coin] = load_coin_full(coin)
            print(f"  {coin.upper()} ✅", flush=True)
        except FileNotFoundError as e:
            print(f"  {coin.upper()} ⚠️ {e}")

    n_combos = len(RSI_TARGETS) * len(PYRAMID_LEVELS)
    print(f"\n총 {n_combos}개 조합 × {len(coin_data)}코인 = {n_combos*len(coin_data)}회 백테스트\n")

    # ── 2026 OOS ──
    print(f"  {'조합':<22}  {'코인':<4}  {'거래':>5}  {'TPD':>5}  {'WR':>6}  {'수익':>9}  {'MDD':>5}  {'PF':>6}  {'Top5':>8}  판정")
    print(f"  {'─'*90}")

    all_results = []  # (name, rsi_name, lvl, coin_rows)
    combo_map = {}

    for rsi_name, rsi_cfg in RSI_TARGETS.items():
        for lvl in PYRAMID_LEVELS:
            name = f"{rsi_name}+P{lvl}"
            cfg  = {**rsi_cfg, **SL_FIXED, "add_levels": lvl}
            rows = {}

            for coin in COINS:
                if coin not in coin_data: continue
                df26 = coin_data[coin][coin_data[coin].index >= "2026-01-01"].copy()
                if len(df26) < 500: continue
                res = run_antifragile(df26, **cfg)
                m   = res["metrics"]; tl = res["trade_log"]
                tpd = m.get("tpd", 0); r5 = remove_top_n(tl, 5)
                ret = m["total_return"]; mdd = m["mdd"]; pf = m.get("profit_factor", 0)
                ok  = sum([ret > 0, tpd >= 1.5, r5 > 0])
                mark = "✅" if ok == 3 else ("⚠️" if ok >= 2 else "❌")
                rows[coin] = dict(n=m["n_trades"], tpd=tpd, wr=m["win_rate"],
                                  ret=ret, mdd=mdd, pf=pf, r5=r5, ok=ok)
                print(f"  {name:<22}  {coin.upper():<4}  {m['n_trades']:>5}  {tpd:>4.2f}  "
                      f"{m['win_rate']:>5.1f}%  {ret:>+8.1f}%  {mdd:>4.1f}%  {pf:>5.3f}  "
                      f"{r5:>+7.1f}%  {mark}", flush=True)

            combo_map[name] = {"cfg": cfg, "rows": rows, "rsi_name": rsi_name, "lvl": lvl}

    # ── 랭킹 요약 ──
    print(f"\n\n{'═'*110}")
    print(f"  피라미딩 횟수별 랭킹 — SL 고정: S2(1.0/1.2)")
    print(f"{'═'*110}")

    # RSI 그룹별로 출력
    for rsi_name in RSI_TARGETS:
        print(f"\n  ── {rsi_name} ──")
        print(f"  {'add_levels':>12}  {'통과합':>6}  {'BTC수익':>9}  {'ETH수익':>9}  {'SOL수익':>9}  {'XRP수익':>9}  {'avg_MDD':>8}  {'avg_PF':>7}  {'avg_TPD':>8}")
        print(f"  {'─'*85}")

        for lvl in PYRAMID_LEVELS:
            name = f"{rsi_name}+P{lvl}"
            d    = combo_map[name]
            rows = d["rows"]
            total_pass = sum(r["ok"] for r in rows.values())
            rets = [rows[c]["ret"] for c in COINS if c in rows]
            ret_cols = "  ".join(f"{rows[c]['ret']:>+8.0f}%" if c in rows else "         " for c in COINS)
            avg_mdd  = np.mean([rows[c]["mdd"] for c in COINS if c in rows])
            avg_pf   = np.mean([rows[c]["pf"]  for c in COINS if c in rows])
            avg_tpd  = np.mean([rows[c]["tpd"] for c in COINS if c in rows])
            print(f"  add_levels={lvl:>2}  {total_pass:>4}/{len(COINS)*3}  {ret_cols}  {avg_mdd:>6.1f}%  {avg_pf:>6.3f}  {avg_tpd:>7.2f}")

    # 전체 통합 랭킹
    print(f"\n\n  {'전체 랭킹':}")
    print(f"  {'순위':<4}  {'조합':<22}  {'통과합':>6}  BTC수익   ETH수익   SOL수익   XRP수익   avg_MDD  avg_PF  avg_TPD")
    print(f"  {'─'*100}")

    ranking = []
    for name, d in combo_map.items():
        rows = d["rows"]
        total_pass = sum(r["ok"] for r in rows.values())
        avg_ret = np.mean([rows[c]["ret"] for c in COINS if c in rows])
        ranking.append((total_pass, avg_ret, name, d))
    ranking.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for rank, (total_pass, avg_ret, name, d) in enumerate(ranking, 1):
        rows = d["rows"]
        ret_cols = "  ".join(f"{rows[c]['ret']:>+8.0f}%" if c in rows else "         " for c in COINS)
        avg_mdd  = np.mean([rows[c]["mdd"] for c in COINS if c in rows])
        avg_pf   = np.mean([rows[c]["pf"]  for c in COINS if c in rows])
        avg_tpd  = np.mean([rows[c]["tpd"] for c in COINS if c in rows])
        print(f"  [{rank:02d}]  {name:<22}  {total_pass:>4}/{len(COINS)*3}  {ret_cols}  {avg_mdd:>5.1f}%  {avg_pf:>5.3f}  {avg_tpd:>6.2f}")

    # ── hist 검증 (선택) ──
    if args.hist:
        top = ranking[:args.hist_top]
        print(f"\n\n{'═'*70}")
        print(f"  상위 {args.hist_top}개 hist 검증 (10창 91일)")
        print(f"{'═'*70}")
        for rank, (_, _, name, d) in enumerate(top, 1):
            cfg = d["cfg"]
            print(f"\n  [{rank}위] {name}")
            print(f"  {'코인':<6}  {'통과':>6}  {'avg수익':>12}  판정")
            print(f"  {'─'*38}")
            for coin in COINS:
                if coin not in coin_data: continue
                hist_start = COIN_CONFIG[coin]["hist_start"]
                h = run_hist(coin_data[coin], cfg, hist_start=hist_start)
                mark = "✅" if h["pass3"] >= 7 else ("⚠️" if h["pass3"] >= 5 else "❌")
                print(f"  {coin.upper():<6}  {h['pass3']:>3}/{h['total']}  {h['avg_ret']:>+11.1f}%  {mark}")


if __name__ == "__main__":
    main()
