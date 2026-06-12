"""
52_preset_compare.py — 프리셋별 백테스트 비교
  Baseline (EMA) / stable / aggressive / conservative
  × BTC / ETH / SOL / XRP
  × 2026 Full / 2026 Jun / hist 10창

Usage:
  python temp/scripts/52_preset_compare.py
  python temp/scripts/52_preset_compare.py --coin btc
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

HIST_START  = {"btc":"2020-01-01","eth":"2021-04-01","sol":"2021-06-01","xrp":"2020-06-01"}

PRESETS = {
    "Baseline": dict(
        dt_rsi_lo=22, dt_rsi_hi=65, rg_rsi_lo=30, rg_rsi_hi=70,
        ut_rsi_lo=35, ut_rsi_hi=78, bb_sigma=0.0,
        trail_atr_init=1.0, trail_atr_tight=1.5,
        rr_base=0.10, rr_add=0.15, add_levels=3, atr_add_step=0.5,
        leverage=3,
    ),
    "stable": dict(
        dt_rsi_lo=25, dt_rsi_hi=65, rg_rsi_lo=25, rg_rsi_hi=75,
        ut_rsi_lo=38, ut_rsi_hi=75, bb_sigma=0.5,
        trail_atr_init=1.5, trail_atr_tight=2.0,
        rr_base=0.10, rr_add=0.15, add_levels=4, atr_add_step=0.5,
        leverage=3,
    ),
    "aggressive": dict(
        dt_rsi_lo=28, dt_rsi_hi=62, rg_rsi_lo=25, rg_rsi_hi=75,
        ut_rsi_lo=40, ut_rsi_hi=72, bb_sigma=0.5,
        trail_atr_init=1.0, trail_atr_tight=2.0,
        rr_base=0.10, rr_add=0.15, add_levels=4, atr_add_step=0.5,
        leverage=3,
    ),
    "conservative": dict(
        dt_rsi_lo=22, dt_rsi_hi=65, rg_rsi_lo=22, rg_rsi_hi=78,
        ut_rsi_lo=35, ut_rsi_hi=78, bb_sigma=0.5,
        trail_atr_init=2.0, trail_atr_tight=2.5,
        rr_base=0.10, rr_add=0.15, add_levels=3, atr_add_step=0.5,
        leverage=3,
    ),
}


# ─── 지표 ─────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame, bb_sigma: float = 0.5) -> pd.DataFrame:
    df    = df.copy()
    close = df["close"]; high = df["high"]; low = df["low"]

    d  = close.diff()
    df["_rsi"] = 100 - 100 / (1 + d.clip(lower=0).ewm(com=13, adjust=False).mean() /
                               ((-d.clip(upper=0)).ewm(com=13, adjust=False).mean() + 1e-9))

    tr = pd.concat([high - low, (high-close.shift()).abs(),
                    (low-close.shift()).abs()], axis=1).max(axis=1)
    df["_atr"] = tr.ewm(span=14, adjust=False).mean()

    cl1h  = close.resample("1h").last().ffill()
    ema1h = cl1h.ewm(span=20, adjust=False).mean()

    # BB zone (sigma=0 → EMA 기준)
    if bb_sigma > 0:
        std1h = cl1h.rolling(20).std()
        df["_trend_up"]   = (cl1h > ema1h + bb_sigma*std1h).reindex(df.index, method="ffill").fillna(False).astype(int)
        df["_trend_down"] = (cl1h < ema1h - bb_sigma*std1h).reindex(df.index, method="ffill").fillna(False).astype(int)
    else:
        df["_trend_up"]   = (cl1h > ema1h).reindex(df.index, method="ffill").fillna(False).astype(int)
        df["_trend_down"] = (cl1h < ema1h).reindex(df.index, method="ffill").fillna(False).astype(int)

    return df


# ─── 엔진 ────────────────────────────────────────────────────────────────────

def run_af(df, dt_rsi_lo=22, dt_rsi_hi=65, rg_rsi_lo=30, rg_rsi_hi=70,
           ut_rsi_lo=35, ut_rsi_hi=78, bb_sigma=0.0,
           leverage=3, rr_base=0.10, rr_add=0.15,
           add_levels=3, atr_add_step=0.5,
           trail_atr_init=1.0, trail_atr_tight=2.0,
           max_hold_bars=288, cooling_bars=100, max_dd_cb=0.30,
           initial_capital=10_000.0):

    df = add_indicators(df, bb_sigma=bb_sigma)
    df = df.reset_index(drop=True).copy()
    df.dropna(subset=["_rsi","_atr"], inplace=True)
    df = df.reset_index(drop=True)

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
        tup   = int(row.get("_trend_up",   0))
        tdn   = int(row.get("_trend_down", 0))

        rsi_lo = dt_rsi_lo if tdn else (ut_rsi_lo if tup else rg_rsi_lo)
        rsi_hi = dt_rsi_hi if tdn else (ut_rsi_hi if tup else rg_rsi_hi)

        equity = capital * (1 + pos*(price-entry_price)/(entry_price+1e-9)*leverage*current_rr) if pos else capital
        peak_cap = max(peak_cap, equity)
        dd = (peak_cap - equity) / (peak_cap + 1e-9)

        if dd > max_dd_cb and cooling_left == 0:
            cooling_left = cooling_bars
            if pos != 0:
                cp  = price - FEE_TOTAL*price*pos
                pnl = max(pos*(cp-entry_price)/(entry_price+1e-9)*leverage*current_rr, -current_rr)
                capital *= (1+pnl)
                trade_log.append({"pnl":pnl,"hold_steps":idx-entry_bar,"forced":True,"direction":pos})
                pos = 0

        if cooling_left > 0:
            cooling_left -= 1
            if pos != 0:
                cp  = price - FEE_TOTAL*price*pos
                pnl = max(pos*(cp-entry_price)/(entry_price+1e-9)*leverage*current_rr, -current_rr)
                capital *= (1+pnl)
                trade_log.append({"pnl":pnl,"hold_steps":idx-entry_bar,"forced":True,"direction":pos})
                pos = 0
            equity_curve.append(capital); continue

        if pos != 0:
            hold = idx - entry_bar
            if pos == 1:
                peak_price = max(peak_price, price)
                mult = trail_atr_tight if add_count > 0 else trail_atr_init
                trail_sl = max(trail_sl, peak_price - mult*atr)
                hit = price <= trail_sl
            else:
                peak_price = min(peak_price, price)
                mult = trail_atr_tight if add_count > 0 else trail_atr_init
                trail_sl = min(trail_sl, peak_price + mult*atr)
                hit = price >= trail_sl

            if hit or hold >= max_hold_bars:
                cp  = price - FEE_TOTAL*price*pos
                pnl = max(pos*(cp-entry_price)/(entry_price+1e-9)*leverage*current_rr, -current_rr)
                capital *= (1+pnl)
                trade_log.append({"pnl":pnl,"hold_steps":hold,"forced":False,"direction":pos})
                pos = 0; add_count = 0; current_rr = 0.0
            else:
                fav = pos*(price-entry_price)/(atr+1e-9)
                if add_count < add_levels and fav >= (add_count+1)*atr_add_step:
                    # 피라미딩 추가 진입 수수료 차감 (notional = capital × rr_add × leverage)
                    capital *= (1 - rr_add * leverage * FEE_TOTAL)
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
        days = max(len(df)/288, 1)
        m["tpd"] = round(len(trade_log)/days, 2)
        wins  = [t["pnl"] for t in trade_log if t["pnl"]>0]
        losss = [t["pnl"] for t in trade_log if t["pnl"]<0]
        m["long_cnt"]  = sum(1 for t in trade_log if t.get("direction")==1)
        m["short_cnt"] = sum(1 for t in trade_log if t.get("direction")==-1)
        if wins and losss:
            m["pf_ratio"] = round(abs(np.mean(wins)/np.mean(losss)), 3)
    return {"metrics": m, "trade_log": trade_log}


# ─── 유틸 ────────────────────────────────────────────────────────────────────

def r5(tl, base=10_000.0):
    top = set(sorted(range(len(tl)), key=lambda i: tl[i]["pnl"], reverse=True)[:5])
    c = base
    for i, t in enumerate(tl):
        if i not in top: c *= 1+t["pnl"]
    return (c/base-1)*100

def ok3(m, tl):
    return sum([m["total_return"]>0, m.get("tpd",0)>=1.5, r5(tl)>0])

def run_hist(raw_df, preset_cfg, seed=42, windows=10, w=91, hist_start="2020-01-01"):
    rng = random.Random(seed)
    raw = raw_df.copy()
    possible = raw[(raw.index>=hist_start) &
                   (raw.index<=raw.index[-1]-pd.Timedelta(days=w))].index
    chosen = sorted(rng.choices(possible, k=windows))
    passes, returns = [], []
    for sd in chosen:
        seg = raw[(raw.index>=sd)&(raw.index<sd+pd.Timedelta(days=w))].copy()
        if len(seg)<500: continue
        res = run_af(seg, **preset_cfg)
        m, tl = res["metrics"], res["trade_log"]
        passes.append(ok3(m,tl)); returns.append(m["total_return"])
    n3 = sum(p==3 for p in passes)
    avg = np.mean(returns) if returns else 0.0
    pos_cnt = sum(r>0 for r in returns)
    return n3, avg, pos_cnt


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
    raw = ROOT/"data/raw"
    if coin == "btc":
        pieces = [_norm(load_csv(f)) for f in sorted(raw.glob("BTCUSDT_5m_*.csv"))
                  if f.exists()]
        if not pieces:
            par = pd.read_parquet(ROOT/"data/signals_2026/backtest_2026_signals.parquet")
            pieces = [_norm(par[["open","high","low","close","volume"]].copy())]
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

def fmt(m, tl, days=None):
    tpd  = m.get("tpd", round(len(tl)/max(days or 1,1), 2))
    ret  = m["total_return"]
    mdd  = m["mdd"]
    pf   = m.get("profit_factor", 0)
    r5v  = r5(tl)
    wr   = m["win_rate"]
    ok   = ok3(m, tl)
    mk   = "✅" if ok==3 else ("⚠" if ok>=2 else "❌")
    return f"{m['n_trades']:>5} {tpd:>4.1f} {wr:>5.1f}% {ret:>+8.1f}% {mdd:>4.1f}% {pf:>5.2f} {r5v:>+7.1f}% {mk}"


def print_coin_table(coin, raw_df):
    label = coin.upper()
    df26  = raw_df[raw_df.index >= "2026-01-01"].copy()
    df_jun= raw_df[raw_df.index >= "2026-06-01"].copy()
    days26  = max((df26.index[-1]-df26.index[0]).days, 1) if len(df26)>0 else 0
    days_jun= max((df_jun.index[-1]-df_jun.index[0]).days, 1) if len(df_jun)>0 else 0

    d26_str = f"{df26.index[0].date()}~{df26.index[-1].date()}" if len(df26)>0 else "N/A"
    dj_str  = f"{df_jun.index[0].date()}~{df_jun.index[-1].date()}" if len(df_jun)>0 else "N/A"

    print(f"\n{'█'*108}")
    print(f"  {label}/USDT")
    print(f"  2026 Full: {d26_str} ({days26}일)   |   2026 Jun: {dj_str} ({days_jun}일)")
    print(f"{'█'*108}")

    COL = "  거래  TPD    WR       수익률   MDD    PF      Top5   판정"
    HDR = f"  {'프리셋':<14}  {COL}   |  {COL}   |   hist통과    평균수익   양수"
    SEP = "  " + "─"*106
    print(f"  {'':14}  {'── 2026 Full ──':^58}  {'── 2026 Jun ──':^52}  {'── hist ──':^28}")
    print(HDR); print(SEP)

    for pname, pcfg in PRESETS.items():
        r26  = run_af(df26,  **pcfg) if len(df26)>100  else None
        rjun = run_af(df_jun,**pcfg) if len(df_jun)>100 else None
        n3h, avgh, pos_cnt = run_hist(raw_df, pcfg,
                                       hist_start=HIST_START.get(coin,"2020-01-01"))
        mk_h = "✅" if n3h>=6 else "❌"
        s26  = fmt(r26["metrics"],  r26["trade_log"],  days26)   if r26  else "  N/A"
        sjun = fmt(rjun["metrics"], rjun["trade_log"], days_jun) if rjun else "  N/A"
        print(f"  {pname:<14}  {s26}   |  {sjun}   |  {n3h:>3}/10 {mk_h}  {avgh:>+8.1f}%  {pos_cnt}/10")

    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", default="all",
                        choices=["btc","eth","sol","xrp","both","all"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--windows", type=int, default=10)
    args = parser.parse_args()

    coin_map = {"both":["btc","eth"], "all":["btc","eth","sol","xrp"]}
    coins    = coin_map.get(args.coin, [args.coin])

    print("\n프리셋별 백테스트 비교 — Baseline / stable / aggressive / conservative")
    print("leverage=3, rr_base=0.10, rr_add=0.15  (실계좌: leverage×5 효과)")

    for coin in coins:
        try:
            raw_df = load_coin(coin)
            print(f"\n{coin.upper()} 데이터: {raw_df.index[0].date()} ~ {raw_df.index[-1].date()}  ({len(raw_df):,}행)")
            print_coin_table(coin, raw_df)
        except FileNotFoundError as e:
            print(f"\n  ⚠️  {e}")

    # 범례
    print("─"*60)
    print("Baseline:     EMA 기준, rg=30/70, trail=1.0/1.5")
    print("stable:       BB0.5σ, rg=25/75, trail=1.5/2.0, P=4")
    print("aggressive:   BB0.5σ, rg=25/75, trail=1.0/2.0, P=4")
    print("conservative: BB0.5σ, rg=22/78, trail=2.0/2.5, P=3")
    print()


if __name__ == "__main__":
    main()
