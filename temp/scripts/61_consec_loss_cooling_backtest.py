"""
[US-004] 연속 손실 방향 쿨링 효과 검증 백테스트
- CONSECUTIVE_LOSS_LIMIT=3회 연속 같은 방향 손실 → DIRECTION_COOLING_BARS=20봉 차단
- Jun11-18 기간 on/off 비교 (Jun14 pump 구간 중점)
"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "src")
from scripts.backtest_antifragile import run_antifragile, load_coin_full
import pandas as pd, numpy as np

CONSEC_LIMIT  = 3
COOLING_BARS  = 20
START, END    = "2026-06-11", "2026-06-18 12:00"
PUMP_START    = "2026-06-14 21:00"
PUMP_END      = "2026-06-15 01:00"
INIT_CAP      = 270.76
PROD = dict(dt_rsi_lo=28, dt_rsi_hi=60, rg_rsi_lo=25, rg_rsi_hi=75,
            ut_rsi_lo=42, ut_rsi_hi=75, trail_atr_init=1.8, trail_atr_tight=2.0,
            leverage=7, rr_base=0.20, rr_add=0.10, add_levels=3, atr_add_step=0.5)

def run_af_with_cooling(df, initial_capital=270.76, consec_limit=3, cooling_bars=20, **kw):
    p = {**PROD}; p.update(kw)
    dt_lo,dt_hi = p["dt_rsi_lo"],p["dt_rsi_hi"]
    rg_lo,rg_hi = p["rg_rsi_lo"],p["rg_rsi_hi"]
    ut_lo,ut_hi = p["ut_rsi_lo"],p["ut_rsi_hi"]
    lev=p.get("leverage",7); rr_base=p["rr_base"]
    trail_init=p["trail_atr_init"]; trail_tight=p["trail_atr_tight"]
    add_step=p["atr_add_step"]; add_lvl=p["add_levels"]
    max_hold=p.get("max_hold_bars",288)

    df=df.reset_index(drop=True); df.dropna(subset=["_rsi","_atr"],inplace=True)
    df=df.reset_index(drop=True)

    capital=initial_capital; peak_cap=initial_capital
    pos=0; entry_px=0.0; entry_atr=0.0; rr=rr_base
    add_cnt=0; trail_sl=0.0; peak_px=0.0; entry_bar=0
    consec_long_loss=0; consec_short_loss=0
    long_cool=0; short_cool=0
    trade_log=[]; pump_trades=[]

    def get_sig(row):
        rsi=row["_rsi"]; tu=bool(row["_trend_up"]); td=bool(row["_trend_down"])
        if td:   lo,hi=dt_lo,dt_hi
        elif tu: lo,hi=ut_lo,ut_hi
        else:    lo,hi=rg_lo,rg_hi
        return rsi<=lo, rsi>=hi

    for i in range(1, len(df)):
        row=df.iloc[i]; price=float(row["close"]); atr=float(row["_atr"])
        long_ok, short_ok = get_sig(row)
        if atr < price*0.0015: long_ok=short_ok=False

        ts = df.index[i] if hasattr(df.index[i], 'strftime') else None

        if pos != 0:
            hit = (pos==1 and price<=trail_sl) or (pos==-1 and price>=trail_sl)
            timeout = (i-entry_bar)>=max_hold
            if hit or timeout:
                reason="trail_SL" if hit else "timeout"
                pnl_raw=pos*(price-entry_px)/(entry_px+1e-9)
                pnl=max(pnl_raw*lev*rr,-rr)-0.00222
                capital*=(1+pnl); peak_cap=max(peak_cap,capital)
                # pump 구간 거래 기록
                is_pump = ts is not None and PUMP_START<=str(ts)[:16]<=PUMP_END
                rec={"pnl":pnl,"direction":pos,"reason":reason,"capital":round(capital,2),"ts":str(ts)[:16] if ts else ""}
                trade_log.append(rec)
                if is_pump: pump_trades.append(rec)
                # 쿨링 카운터 업데이트
                closed_pos=pos
                if pnl<=0:
                    if closed_pos==1: consec_long_loss+=1; consec_short_loss=0
                    else:             consec_short_loss+=1; consec_long_loss=0
                else:
                    if closed_pos==1: consec_long_loss=0
                    else:             consec_short_loss=0
                if consec_long_loss>=consec_limit:
                    long_cool=cooling_bars; consec_long_loss=0
                if consec_short_loss>=consec_limit:
                    short_cool=cooling_bars; consec_short_loss=0
                pos=0; entry_px=0; rr=rr_base; add_cnt=0; trail_sl=0; peak_px=0
            else:
                eff_atr=max(atr,entry_atr*0.6); mult=trail_tight if add_cnt>0 else trail_init
                if pos==1:
                    peak_px=max(peak_px,price); trail_sl=max(trail_sl,peak_px-mult*eff_atr)
                    if add_cnt<add_lvl and pos*(price-entry_px)/(entry_atr+1e-9)>=(add_cnt+1)*add_step:
                        add_cnt+=1; rr+=p["rr_add"]; trail_sl=max(trail_sl,price-trail_tight*atr)
                else:
                    peak_px=min(peak_px,price); trail_sl=min(trail_sl,peak_px+mult*eff_atr)
                    if add_cnt<add_lvl and pos*(price-entry_px)/(entry_atr+1e-9)>=(add_cnt+1)*add_step:
                        add_cnt+=1; rr+=p["rr_add"]; trail_sl=min(trail_sl,price+trail_tight*atr)

        if pos==0:
            if long_cool>0:  long_cool-=1;  long_ok=False
            if short_cool>0: short_cool-=1; short_ok=False
            d=1 if long_ok else (-1 if short_ok else 0)
            if d!=0:
                pos=d; entry_px=price; entry_atr=atr; rr=rr_base
                peak_px=price; add_cnt=0; entry_bar=i
                trail_sl=(price-trail_init*atr) if d==1 else (price+trail_init*atr)

    wins=[t for t in trade_log if t["pnl"]>0]
    ret=(capital-initial_capital)/initial_capital*100
    pump_losses=[t for t in pump_trades if t["pnl"]<=0]
    return {"total_return":ret,"n_trades":len(trade_log),"wins":len(wins),
            "pump_losses":len(pump_losses),"pump_trades":len(pump_trades),"capital":capital}

coins=["btc","eth","sol","xrp"]

print("="*68)
print("[US-004] 연속 손실 방향 쿨링 효과 — Jun11-18")
print(f"설정: 연속{CONSEC_LIMIT}손실 → {COOLING_BARS}봉 해당방향 차단")
print("="*68)

tot_off=0.0; tot_on=0.0; tot_pump_off=0; tot_pump_on=0
for coin in coins:
    df=load_coin_full(coin)
    df=df[(df.index>=START)&(df.index<END)].copy()
    if len(df)<100: continue
    days=(df.index[-1]-df.index[0]).total_seconds()/86400

    r_off=run_antifragile(df,initial_capital=INIT_CAP,**PROD)
    m_off=r_off["metrics"]
    tl_off=r_off["trade_log"]
    pump_off=[t for t in tl_off if PUMP_START<=t.get("time","")[:16]<=PUMP_END and t["pnl"]<=0]

    r_on=run_af_with_cooling(df,initial_capital=INIT_CAP,consec_limit=CONSEC_LIMIT,cooling_bars=COOLING_BARS)
    tot_off+=m_off["total_return"]; tot_on+=r_on["total_return"]
    tot_pump_off+=len(pump_off); tot_pump_on+=r_on["pump_losses"]

    wr_on=r_on["wins"]/r_on["n_trades"]*100 if r_on["n_trades"] else 0
    print(f"\n  [{coin.upper()}]")
    print(f"    쿨링 OFF: 거래={m_off['n_trades']:3d}  수익={m_off['total_return']:+.2f}%  WR={m_off['win_rate']:.0f}%  Jun14펌프손실={len(pump_off)}건")
    print(f"    쿨링 ON : 거래={r_on['n_trades']:3d}  수익={r_on['total_return']:+.2f}%  WR={wr_on:.0f}%  Jun14펌프손실={r_on['pump_losses']}건")
    print(f"    효과: 수익 {r_on['total_return']-m_off['total_return']:+.2f}%p  펌프구간 손실 {r_on['pump_losses']-len(pump_off):+d}건")

print(f"\n{'='*68}")
print(f"  [합산] 쿨링OFF={tot_off:+.2f}%  쿨링ON={tot_on:+.2f}%  효과={tot_on-tot_off:+.2f}%p")
print(f"  Jun14 pump 손실: OFF={tot_pump_off}건 → ON={tot_pump_on}건 ({tot_pump_on-tot_pump_off:+d}건)")
