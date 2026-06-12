"""
51_bb_trail_sweep.py — BB0.5σ 횡보 구역 + trail 파라미터 종합 스윕

고정:  BB_sigma=0.5, dt_rsi=22/65, ut_rsi=35/78
스윕:
  rg_rsi    : (22,78) / (25,75)
  trail_init: 1.0 / 1.5 / 2.0
  trail_tight: 2.0 / 2.5 / 3.0  (tight >= init)

테스트 구간:
  ① 2026 Full  (2026-01-01 ~ 최신)
  ② 2026 Jun   (2026-06-01 ~ 최신)
  ③ hist 10창  (91일 랜덤)

Usage:
  python temp/scripts/51_bb_trail_sweep.py --coin btc --mode both
  python temp/scripts/51_bb_trail_sweep.py --coin all  --mode both
"""
import sys, argparse, random, itertools
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path
from hybrid_engine import compute_metrics

ROOT        = Path(__file__).parent.parent.parent
TRADING_FEE = 0.0005
SLIPPAGE    = 0.0002
FEE_TOTAL   = TRADING_FEE + SLIPPAGE

RG_PAIRS    = [(22, 78), (25, 75)]
TRAIL_INITS = [1.0, 1.5, 2.0]
TRAIL_TIGHTS= [2.0, 2.5, 3.0]
TRAIL_PAIRS = [(i, t) for i in TRAIL_INITS for t in TRAIL_TIGHTS if t >= i]

HIST_START  = {"btc":"2020-01-01","eth":"2021-04-01","sol":"2021-06-01","xrp":"2020-06-01"}


# ─── 지표 ─────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame, bb_sigma: float = 0.5) -> pd.DataFrame:
    df    = df.copy()
    close = df["close"]; high = df["high"]; low = df["low"]

    # RSI
    d  = close.diff()
    df["_rsi"] = 100 - 100 / (1 + d.clip(lower=0).ewm(com=13, adjust=False).mean() /
                               ((-d.clip(upper=0)).ewm(com=13, adjust=False).mean() + 1e-9))

    # ATR
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    df["_atr"] = tr.ewm(span=14, adjust=False).mean()

    # 1h EMA20 + BB (BB 구역으로 3-state 추세 판별)
    cl1h  = close.resample("1h").last().ffill()
    ema1h = cl1h.ewm(span=20, adjust=False).mean()
    std1h = cl1h.rolling(20).std()
    bb_up = ema1h + bb_sigma * std1h
    bb_lo = ema1h - bb_sigma * std1h

    df["_trend_up"]   = (cl1h > bb_up).reindex(df.index, method="ffill").fillna(False).astype(int)
    df["_trend_down"] = (cl1h < bb_lo).reindex(df.index, method="ffill").fillna(False).astype(int)

    # 기준선 (EMA only, Baseline 비교용)
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
           trail_atr_init=1.0, trail_atr_tight=2.0,
           max_hold_bars=288, cooling_bars=100, max_dd_cb=0.30,
           use_ema=False):

    df = df.reset_index(drop=True).copy()
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    df = df.reset_index(drop=True)

    up_col = "_trend_up_ema" if use_ema else "_trend_up"
    dn_col = "_trend_down_ema" if use_ema else "_trend_down"

    capital = initial_capital; peak_cap = initial_capital
    pos = 0; entry_price = 0.0; current_rr = 0.0
    add_count = 0; trail_sl = 0.0; peak_price = 0.0
    entry_bar = 0; cooling_left = 0

    equity_curve = [capital]; trade_log = []

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
                pnl = max(pos*(cp-entry_price)/(entry_price+1e-9)*leverage*current_rr, -current_rr)
                capital *= (1+pnl)
                trade_log.append({"pnl":pnl,"hold_steps":idx-entry_bar,
                                   "lev":leverage,"rr":current_rr,"forced":True,"direction":pos})
                pos = 0

        if cooling_left > 0:
            cooling_left -= 1
            if pos != 0:
                cp  = price - FEE_TOTAL * price * pos
                pnl = max(pos*(cp-entry_price)/(entry_price+1e-9)*leverage*current_rr, -current_rr)
                capital *= (1+pnl)
                trade_log.append({"pnl":pnl,"hold_steps":idx-entry_bar,
                                   "lev":leverage,"rr":current_rr,"forced":True,"direction":pos})
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
                pnl = max(pos*(cp-entry_price)/(entry_price+1e-9)*leverage*current_rr, -current_rr)
                capital *= (1+pnl)
                trade_log.append({"pnl":pnl,"hold_steps":hold,
                                   "lev":leverage,"rr":current_rr,"forced":False,"direction":pos})
                pos = 0; add_count = 0; current_rr = 0.0
            else:
                fav = pos*(price-entry_price)/(atr+1e-9)
                if add_count < add_levels and fav >= (add_count+1)*atr_add_step:
                    current_rr += rr_add; add_count += 1
                    if pos == 1: trail_sl = max(trail_sl, price - trail_atr_tight*atr)
                    else:        trail_sl = min(trail_sl, price + trail_atr_tight*atr)

        if pos == 0:
            if rsi <= rsi_lo:
                ep = price*(1+FEE_TOTAL)
                entry_price=ep; current_rr=rr_base; add_count=0
                trail_sl=ep-trail_atr_init*atr; peak_price=ep
                pos=1; entry_bar=idx
            elif rsi >= rsi_hi:
                ep = price*(1-FEE_TOTAL)
                entry_price=ep; current_rr=rr_base; add_count=0
                trail_sl=ep+trail_atr_init*atr; peak_price=ep
                pos=-1; entry_bar=idx

        equity_curve.append(capital)

    m = compute_metrics(equity_curve, trade_log)
    if trade_log:
        days = len(df)/288
        m["tpd"] = round(len(trade_log)/days, 2)
        wins  = [t["pnl"] for t in trade_log if t["pnl"]>0]
        losss = [t["pnl"] for t in trade_log if t["pnl"]<0]
        if wins and losss:
            m["pf_ratio"] = round(abs(np.mean(wins)/np.mean(losss)), 3)
    return {"metrics": m, "trade_log": trade_log}


# ─── 유틸 ────────────────────────────────────────────────────────────────────

def r5(trades, base=10_000.0):
    top = set(sorted(range(len(trades)), key=lambda i: trades[i]["pnl"], reverse=True)[:5])
    c = base
    for i, t in enumerate(trades):
        if i not in top: c *= 1+t["pnl"]
    return (c/base-1)*100

def ok3(m, tl):
    return sum([m["total_return"]>0, m.get("tpd",0)>=1.5, r5(tl)>0])

def run_hist(all_df, cfg, seed=42, windows=10, w=91, hist_start="2020-01-01"):
    all_df = all_df.dropna(subset=["_rsi","_atr"])
    rng = random.Random(seed)
    possible = all_df[(all_df.index>=hist_start) &
                      (all_df.index<=all_df.index[-1]-pd.Timedelta(days=w))].index
    chosen = sorted(rng.choices(possible, k=windows))
    passes, returns = [], []
    for sd in chosen:
        seg = all_df[(all_df.index>=sd) & (all_df.index<sd+pd.Timedelta(days=w))].copy()
        if len(seg)<500: continue
        res = run_af(seg, **cfg)
        m, tl = res["metrics"], res["trade_log"]
        passes.append(ok3(m,tl)); returns.append(m["total_return"])
    n3 = sum(p==3 for p in passes)
    return n3, (np.mean(returns) if returns else 0.0)


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def _norm(df):
    df = df.copy()
    df.index = df.index.tz_convert(None) if df.index.tz else df.index
    return df

def load_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.columns = [c.lower() for c in df.columns]
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()

def load_coin(coin: str) -> pd.DataFrame:
    coin = coin.lower()
    raw = ROOT / "data/raw"
    if coin == "btc":
        pieces = []
        for f in sorted(raw.glob("BTCUSDT_5m_*.csv")):
            try: pieces.append(_norm(load_csv(f)))
            except: pass
        if not pieces:
            par = pd.read_parquet(ROOT/"data/signals_2026/backtest_2026_signals.parquet")
            pieces.append(_norm(par[["open","high","low","close","volume"]].copy()))
    elif coin == "eth":
        pieces = [_norm(pd.read_parquet(ROOT/"data/eth/ETHUSDT_5m_history.parquet")),
                  _norm(pd.read_parquet(ROOT/"data/eth/ETHUSDT_5m_2026.parquet"))]
        for f in sorted(raw.glob("ETHUSDT_5m_*.csv")):
            try: pieces.append(_norm(load_csv(f)))
            except: pass
    else:
        sym = f"{coin.upper()}USDT"
        cands = sorted(raw.glob(f"{sym}_5m_*.csv"))
        if not cands: raise FileNotFoundError(f"{sym} 데이터 없음")
        pieces = [_norm(load_csv(f)) for f in cands]

    df = pd.concat(pieces).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[df["close"].notna() & (df["close"]>0)]


# ─── 출력 ─────────────────────────────────────────────────────────────────────

def fmt_period(res, short=False):
    """단일 구간 결과를 컴팩트 문자열로."""
    m, tl = res["metrics"], res["trade_log"]
    tpd = m.get("tpd", 0)
    ret = m["total_return"]
    mdd = m["mdd"]
    pf  = m.get("profit_factor", 0)
    r5v = r5(tl)
    ok  = ok3(m, tl)
    mk  = "✅" if ok==3 else ("⚠" if ok>=2 else "❌")
    if short:
        return f"{m['n_trades']:>4} {tpd:>4.1f} {ret:>+7.1f}% {mdd:>4.1f}% {pf:>5.2f} {mk}"
    return f"{m['n_trades']:>5} {tpd:>4.2f} {ret:>+8.1f}% {mdd:>4.1f}% {pf:>5.3f} {r5v:>+7.1f}% {mk}({ok}/3)"


def print_section(title, results_list):
    """results_list: [(label, res26, res_jun, n3_hist, avg_hist), ...]"""
    print(f"\n{'─'*100}")
    print(f"  {title}")
    print(f"  {'설정':<22}"
          f" | {'── 2026 Full ──':^40}"
          f" | {'── 2026 Jun ──':^30}"
          f" | {'── hist ──':^14}")
    print(f"  {'':22}"
          f"   {'거래':>4} {'TPD':>4} {'수익률':>8} {'MDD':>5} {'PF':>5} {'Top5':>7} {'판정':>6}"
          f"   {'거래':>4} {'TPD':>4} {'수익률':>7} {'MDD':>5} {'PF':>5} {'판정':>5}"
          f"   {'통과':>5} {'평균수익':>9}")
    print(f"  {'─'*98}")
    for label, r26, rjun, n3h, avgh in results_list:
        mk_h = "✅" if n3h >= 6 else "❌"
        line = (f"  {label:<22}"
                f"   {fmt_period(r26, short=True)}"
                f"   {fmt_period(rjun, short=True)}"
                f"   {n3h:>3}/10  {avgh:>+8.1f}% {mk_h}")
        print(line)


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
        print(f"\n{'█'*100}")
        print(f"  {label}/USDT — BB0.5σ + trail sweep")
        print(f"  trail_init: {TRAIL_INITS}  trail_tight: {TRAIL_TIGHTS}  rg_rsi: {RG_PAIRS}")
        print(f"{'█'*100}")

        try:
            raw_df = load_coin(coin)
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}"); continue

        print(f"  데이터: {raw_df.index[0].date()} ~ {raw_df.index[-1].date()}  ({len(raw_df):,}행)")

        # 지표 계산 (BB0.5σ)
        all_df = add_indicators(raw_df, bb_sigma=0.5)

        # 구간 슬라이스
        df_2026     = all_df[all_df.index >= "2026-01-01"].copy()
        df_jun      = all_df[all_df.index >= "2026-06-01"].copy()
        days_2026   = max((df_2026.index[-1] - df_2026.index[0]).days, 1) if len(df_2026)>0 else 0
        days_jun    = max((df_jun.index[-1]   - df_jun.index[0]).days,   1) if len(df_jun)>0  else 0

        print(f"  2026 Full: {df_2026.index[0].date() if len(df_2026)>0 else '-'} ~ "
              f"{df_2026.index[-1].date() if len(df_2026)>0 else '-'}  ({days_2026}일)")
        print(f"  2026 Jun:  {df_jun.index[0].date()  if len(df_jun)>0  else '-'} ~ "
              f"{df_jun.index[-1].date()  if len(df_jun)>0  else '-'}  ({days_jun}일)")

        BASE_CFG = dict(leverage=3, rr_base=0.10, rr_add=0.15,
                        add_levels=3, atr_add_step=0.5,
                        dt_rsi_lo=22, dt_rsi_hi=65,
                        ut_rsi_lo=35, ut_rsi_hi=78)

        for rg_lo, rg_hi in RG_PAIRS:
            rows = []

            # Baseline (EMA, rg=30/70, trail 1.0/1.5)
            base_cfg_e = {**BASE_CFG, "rg_rsi_lo":30, "rg_rsi_hi":70,
                          "trail_atr_init":1.0, "trail_atr_tight":1.5, "use_ema":True}
            r26_b   = run_af(df_2026, **base_cfg_e) if len(df_2026)>100 else None
            rjun_b  = run_af(df_jun,  **base_cfg_e) if len(df_jun)>100  else None
            if args.mode in ("random","both"):
                n3h_b, avg_b = run_hist(all_df, {**base_cfg_e},
                                        seed=args.seed, windows=args.windows,
                                        hist_start=HIST_START.get(coin,"2020-01-01"))
            else:
                n3h_b, avg_b = 0, 0.0
            if r26_b and rjun_b:
                rows.append(("Baseline EMA rg=30/70", r26_b, rjun_b, n3h_b, avg_b))

            # BB0.5σ 조합 스윕
            for ti, tt in TRAIL_PAIRS:
                cfg = {**BASE_CFG, "rg_rsi_lo":rg_lo, "rg_rsi_hi":rg_hi,
                       "trail_atr_init":ti, "trail_atr_tight":tt, "use_ema":False}
                r26  = run_af(df_2026, **cfg) if len(df_2026)>100 else None
                rjun = run_af(df_jun,  **cfg) if len(df_jun)>100  else None
                if args.mode in ("random","both"):
                    n3h, avgh = run_hist(all_df, cfg,
                                         seed=args.seed, windows=args.windows,
                                         hist_start=HIST_START.get(coin,"2020-01-01"))
                else:
                    n3h, avgh = 0, 0.0
                lbl = f"BB0.5 rg={rg_lo}/{rg_hi} i={ti} t={tt}"
                if r26 and rjun:
                    rows.append((lbl, r26, rjun, n3h, avgh))

            print_section(f"rg={rg_lo}/{rg_hi} 스윕 결과", rows)

        # 최적 후보 요약
        print(f"\n{'═'*100}")
        print(f"  ※ 프리셋 추천 기준: 2026 Full ✅ + Jun ✅ + hist ≥7/10")
        print(f"  ※ Standard   → rg=25/75, trail_init=1.0~1.5, trail_tight=2.0~2.5")
        print(f"  ※ Conservative → rg=22/78, trail_init=1.5~2.0, trail_tight=2.5~3.0")
        print()


if __name__ == "__main__":
    main()
