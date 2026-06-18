"""
[US-002] 강한 반대 신호 조기청산 (position flip) 백테스트 검증
- CONSECUTIVE_REVERSE_BARS=2봉 연속 반대 신호 → 즉시 청산
- Jun11-18 기간 on/off 비교
"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "src")
from scripts.backtest_antifragile import load_coin_full
from config.af_params import DEFAULT_PARAMS
import pandas as pd, numpy as np

CONSECUTIVE_REVERSE_BARS = 2
START, END = "2026-06-11", "2026-06-18 12:00"
INIT_CAP   = 270.76
PROD = dict(dt_rsi_lo=28, dt_rsi_hi=60, rg_rsi_lo=25, rg_rsi_hi=75,
            ut_rsi_lo=42, ut_rsi_hi=75, trail_atr_init=1.8, trail_atr_tight=2.0,
            leverage=7, rr_base=0.20, rr_add=0.10, add_levels=3, atr_add_step=0.5)

def run_af_with_flip(df, initial_capital=270.76, flip_bars=2, **kw):
    """run_antifragile에 reverse_flip 로직을 추가한 버전."""
    # 원본 실행 (flip 없음)은 직접 호출. flip 있는 버전은 여기서 구현.
    p = {**PROD}
    p.update(kw)

    df = df.reset_index(drop=True)
    df.dropna(subset=["_rsi","_atr"], inplace=True)
    df = df.reset_index(drop=True)

    dt_lo,dt_hi = p["dt_rsi_lo"],p["dt_rsi_hi"]
    rg_lo,rg_hi = p["rg_rsi_lo"],p["rg_rsi_hi"]
    ut_lo,ut_hi = p["ut_rsi_lo"],p["ut_rsi_hi"]
    lev         = p.get("leverage", 7)
    rr_base     = p["rr_base"]
    trail_init  = p["trail_atr_init"]
    trail_tight = p["trail_atr_tight"]
    add_step    = p["atr_add_step"]
    add_lvl     = p["add_levels"]
    max_hold    = p.get("max_hold_bars", 288)

    capital   = initial_capital
    peak_cap  = initial_capital
    pos       = 0
    entry_px  = 0.0; entry_atr = 0.0; rr = rr_base
    add_cnt   = 0;   trail_sl  = 0.0; peak_px = 0.0
    entry_bar = 0;   consec_rev = 0
    trade_log = []; equity_curve = [capital]

    def get_sig(row):
        rsi   = row["_rsi"]
        trend_up   = bool(row["_trend_up"])
        trend_down = bool(row["_trend_down"])
        if trend_down:   lo,hi = dt_lo,dt_hi
        elif trend_up:   lo,hi = ut_lo,ut_hi
        else:            lo,hi = rg_lo,rg_hi
        return rsi <= lo, rsi >= hi

    for i in range(1, len(df)):
        row   = df.iloc[i]
        price = float(row["close"])
        atr   = float(row["_atr"])
        long_ok, short_ok = get_sig(row)

        if atr < price * 0.0015:
            long_ok = short_ok = False

        if pos != 0:
            rev = (pos==1 and short_ok) or (pos==-1 and long_ok)
            consec_rev = consec_rev+1 if rev else 0
            force_flip = consec_rev >= flip_bars

            hit = (pos==1 and price<=trail_sl) or (pos==-1 and price>=trail_sl)
            timeout = (i - entry_bar) >= max_hold

            if hit or timeout or force_flip:
                reason = "trail_SL" if hit else ("timeout" if timeout else "reverse_flip")
                pnl_raw = pos*(price-entry_px)/(entry_px+1e-9)
                pnl     = max(pnl_raw*lev*rr, -rr) - 0.00222  # fee
                capital *= (1+pnl)
                peak_cap = max(peak_cap, capital)
                trade_log.append({"pnl":pnl,"direction":pos,"reason":reason,"capital":round(capital,2)})
                pos=0; entry_px=0; rr=rr_base; add_cnt=0; trail_sl=0; peak_px=0; consec_rev=0
            else:
                entry_atr_eff = max(atr, entry_atr*0.6)
                mult = trail_tight if add_cnt>0 else trail_init
                if pos==1:
                    peak_px = max(peak_px, price)
                    trail_sl= max(trail_sl, peak_px - mult*entry_atr_eff)
                    if add_cnt<add_lvl and pos*(price-entry_px)/(entry_atr+1e-9)>=(add_cnt+1)*add_step:
                        add_cnt+=1; rr+=p["rr_add"]
                        trail_sl=max(trail_sl, price-trail_tight*atr)
                else:
                    peak_px = min(peak_px, price)
                    trail_sl= min(trail_sl, peak_px + mult*entry_atr_eff)
                    if add_cnt<add_lvl and pos*(price-entry_px)/(entry_atr+1e-9)>=(add_cnt+1)*add_step:
                        add_cnt+=1; rr+=p["rr_add"]
                        trail_sl=min(trail_sl, price+trail_tight*atr)

        if pos == 0:
            d = 1 if long_ok else (-1 if short_ok else 0)
            if d != 0:
                pos=d; entry_px=price; entry_atr=atr; rr=rr_base
                peak_px=price; add_cnt=0; consec_rev=0
                trail_sl = (price-trail_init*atr) if d==1 else (price+trail_init*atr)
                entry_bar=i

        equity_curve.append(capital)

    wins   = [t for t in trade_log if t["pnl"]>0]
    losses = [t for t in trade_log if t["pnl"]<=0]
    ret    = (capital-initial_capital)/initial_capital*100
    return {"total_return":ret, "n_trades":len(trade_log), "wins":len(wins),
            "flips":[t for t in trade_log if t.get("reason")=="reverse_flip"],
            "capital":capital}

from scripts.backtest_antifragile import run_antifragile, load_coin_full
coins = ["btc","eth","sol","xrp"]

print("="*68)
print("[US-002] position flip 로직 효과 검증 — Jun11-18")
print(f"설정: CONSECUTIVE_REVERSE_BARS={CONSECUTIVE_REVERSE_BARS}")
print("="*68)

tot_off=0.0; tot_on=0.0; tot_flips=0
for coin in coins:
    df = load_coin_full(coin)
    df = df[(df.index>=START)&(df.index<END)].copy()
    if len(df)<100: continue
    days = (df.index[-1]-df.index[0]).total_seconds()/86400

    r_off = run_antifragile(df, initial_capital=INIT_CAP, **PROD)
    m_off = r_off["metrics"]

    r_on  = run_af_with_flip(df, initial_capital=INIT_CAP, flip_bars=CONSECUTIVE_REVERSE_BARS)
    flips = len(r_on["flips"])
    tot_flips += flips
    tot_off += m_off["total_return"]
    tot_on  += r_on["total_return"]

    wr_on = r_on["wins"]/r_on["n_trades"]*100 if r_on["n_trades"] else 0
    print(f"\n  [{coin.upper()}]")
    print(f"    flip OFF: 거래={m_off['n_trades']:3d}  TPD={m_off['n_trades']/days:.1f}  "
          f"수익={m_off['total_return']:+.2f}%  WR={m_off['win_rate']:.0f}%")
    print(f"    flip ON : 거래={r_on['n_trades']:3d}  TPD={r_on['n_trades']/days:.1f}  "
          f"수익={r_on['total_return']:+.2f}%  WR={wr_on:.0f}%  (flip발동={flips}건)")
    diff = r_on["total_return"]-m_off["total_return"]
    print(f"    flip 효과: {diff:+.2f}%p  거래수변화: {r_on['n_trades']-m_off['n_trades']:+d}")

print(f"\n{'='*68}")
print(f"  [합산] flip OFF={tot_off:+.2f}%  flip ON={tot_on:+.2f}%  "
      f"효과={tot_on-tot_off:+.2f}%p  총 flip발동={tot_flips}건")
