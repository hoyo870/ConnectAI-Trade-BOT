"""
temp/scripts/keltner_reversal_2r.py
[오류 수정 완료: close_all 인자 불일치 해결]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config.loader import load_coin_raw
from keltner_super_bands import add_keltner, realize, _mdd, selftest

ENTRY_PCT = 0.50
RR = 2.0
TRAIL_ATR = 2.0

def run(df, leverage=1.0, initial_capital=1000.0, ma_type="SMA",
        rr=RR, trail_atr=TRAIL_ATR, cost_mult=1.0) -> dict:
    df = (add_keltner(df, ma_type, ma_trend_len=300)
          .dropna(subset=["basis", "rangema", "trend_ema"])
          .reset_index())
    df.rename(columns={"index": "_ts", "timestamp": "_ts"}, inplace=True)

    df['entry_long'] = np.nan; df['entry_short'] = np.nan
    df['exit_tp'] = np.nan; df['exit_sl'] = np.nan; df['sl_line'] = np.nan; df['tgt'] = np.nan

    cap = initial_capital
    pos = 0
    chunks = []
    sl = tgt = peak = trail = None
    trailing = False
    armed_long = armed_short = False
    swing_low = swing_high = None
    trades = []
    equity = [cap]

    def close_all(exit_px, ts, reason, idx):
        nonlocal cap, pos, chunks, sl, tgt, peak, trail, trailing
        if pos == 0: return
        ret = realize(chunks, exit_px, pos, cost_mult)
        cap *= 1 + ret
        trades.append({"dir": pos, "ret": ret, "reason": reason})
        
        if reason == "2R" or reason == "trail": df.loc[idx, 'exit_tp'] = exit_px
        elif reason == "SL": df.loc[idx, 'exit_sl'] = exit_px
        
        pos = 0; chunks = []; sl = tgt = peak = trail = None; trailing = False

    for i in range(1, len(df)):
        row = df.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        prev = df.iloc[i - 1]
        atr, upper, lower = float(row["rangema"]), float(row["upper"]), float(row["lower"])
        basis = float(row["basis"])
        super_upper, super_lower = float(row["super_upper"]), float(row["super_lower"])
        test_upper, test_lower = float(row["test_upper"]), float(row["test_lower"])

        if pos == 1:
            df.loc[i, 'tgt'] = tgt
            df.loc[i, 'sl_line'] = sl
            if l <= sl: close_all(o if o <= sl else sl, row["_ts"], "SL", i)
            else:
                peak = max(peak, h)
                if h >= tgt: 
                    close_all(o if o >= tgt else tgt, row["_ts"], "2R", i)
                if not trailing:
                    if h >= upper: 
                        trailing = True; 
                        #trail = peak - trail_atr * atr
                        trail = test_lower
                        df.loc[i, 'sl_line'] = trail
                if trailing and pos == 1:
                    # trail = max(trail, peak - trail_atr * atr)
                    trail = max(trail, test_lower)
                    df.loc[i, 'sl_line'] = trail
                    if l <= trail: close_all(o if o <= trail else trail, row["_ts"], "trail", i)
        elif pos == -1:
            df.loc[i, 'tgt'] = tgt
            df.loc[i, 'sl_line'] = sl
            if h >= sl: close_all(o if o >= sl else sl, row["_ts"], "SL", i)
            else:
                peak = min(peak, l)
                if l <= tgt: 
                    close_all(o if o <= tgt else tgt, row["_ts"], "2R", i)
                if not trailing:
                    if l <= lower: 
                        trailing = True; 
                        #trail = peak + trail_atr * atr
                        trail = test_upper
                        df.loc[i, 'sl_line'] = trail
                if trailing and pos == -1:
                    # trail = min(trail, peak + trail_atr * atr)
                    trail = min(trail, test_upper)
                    df.loc[i, 'sl_line'] = trail
                    if h >= trail: close_all(o if o >= trail else trail, row["_ts"], "trail", i)

        if pos == 0:
            pc, pbasis, pupper, plower = float(prev["close"]), float(prev["basis"]), float(prev["test_upper"]), float(prev["test_lower"])
            fill = o; w = ENTRY_PCT * leverage
            channel_width_pct = (pupper - plower) / pbasis
            check_width = channel_width_pct > 0.005


            if armed_long and pc > lower and swing_low is not None and swing_low < fill:
                pos = 1; chunks = [(fill, w)]; sl = swing_low
                tgt = fill + rr * (fill - sl); peak = h; trailing = False
                armed_long = False; swing_low = None
                df.loc[i, 'entry_long'] = fill
            elif armed_short and pc < upper and swing_high is not None and swing_high > fill:
                pos = -1; chunks = [(fill, w)]; sl = swing_high
                tgt = fill - rr * (sl - fill); peak = l; trailing = False
                armed_short = False; swing_high = None
                df.loc[i, 'entry_short'] = fill

            if l < plower and plower < super_lower: 
                armed_long = True; 
                swing_low = l if swing_low is None else min(swing_low, l)
            elif armed_long: 
                swing_low = min(swing_low, l)
            if h > pupper and pupper > super_upper: 
                armed_short = True; 
                swing_high = h if swing_high is None else max(swing_high, h)
            elif armed_short: 
                swing_high = max(swing_high, h)
            
            # if check_width and armed_long and pc > pbasis and swing_low is not None and swing_low < fill:
            #     pos = 1; chunks = [(fill, w)]; sl = swing_low
            #     tgt = fill + rr * (fill - sl); peak = h; trailing = False
            #     armed_long = False; swing_low = None
            #     df.loc[i, 'entry_long'] = fill
            # elif check_width and armed_short and pc < pbasis and swing_high is not None and swing_high > fill:
            #     pos = -1; chunks = [(fill, w)]; sl = swing_high
            #     tgt = fill - rr * (sl - fill); peak = l; trailing = False
            #     armed_short = False; swing_high = None
            #     df.loc[i, 'entry_short'] = fill

            # pos = 0

            # # if c < super_lower: 
            # #     armed_long = True; 
            # #     swing_low = c if swing_low is None else min(swing_low, c)
            # # elif armed_long: 
            # #     swing_low = min(swing_low, c)
            # # if c > super_upper: 
            # #     armed_short = True; 
            # #     swing_high = c if swing_high is None else max(swing_high, c)
            # # elif armed_short: 
            # #     swing_high = max(swing_high, c)
            # if l < super_lower: 
            #     armed_long = True; 
            #     swing_low = l if swing_low is None else min(swing_low, l)
            # elif armed_long: 
            #     swing_low = min(swing_low, l)
            # if h > super_upper: 
            #     armed_short = True; 
            #     swing_high = h if swing_high is None else max(swing_high, h)
            # elif armed_short: 
            #     swing_high = max(swing_high, h)
            
        equity.append(cap * (1 + realize(chunks, c, pos, cost_mult)) if pos != 0 else cap)

    if pos != 0: close_all(float(df.iloc[-1]["close"]), df.iloc[-1]["_ts"], "SL", len(df)-1)
    
    rets = [t["ret"] for t in trades]
    wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
    df["equity"] = equity
    return {"n_trades": len(trades), "total_return": (cap - initial_capital)/initial_capital*100, 
            "win_rate": (len([t for t in trades if t['ret']>0])/len(trades)*100 if trades else 0), 
            "mdd": _mdd(equity),
            "profit_factor": (abs(sum(wins) / sum(losses)) if losses and sum(losses) else float("inf")),
            "avg_win": (np.mean(wins) * 100 if wins else 0.0),
            "avg_loss": (np.mean(losses) * 100 if losses else 0.0),
            "n_long": sum(1 for t in trades if t["dir"] == 1),
            "n_short": sum(1 for t in trades if t["dir"] == -1),
            "n_2r": sum(1 for t in trades if t["reason"] == "2R"),
            "n_trail": sum(1 for t in trades if t["reason"] == "trail"),
            "n_sl": sum(1 for t in trades if t["reason"] == "SL"),
            "mdd": _mdd(equity), "profit_factor": (abs(sum([t['ret'] for t in trades if t['ret']>0])/sum([t['ret'] for t in trades if t['ret']<=0])) if any(t['ret']<=0 for t in trades) else float("inf")),
            "df": df}

def save_backtest_plot(df, coin, start):
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # 1. 봉 차트(Candlestick) 수동 구현
    width = 0.0025 # 봉의 두께
    #ax1.bar(df["_ts"], df.high - df.open, 0.0001, bottom=df.open, color=np.where(df.close >= df.open, 'lime', 'red'), alpha=0.4)
    #ax1.bar(df["_ts"], df.open - df.close, width, bottom=df.close, color=np.where(df.close >= df.open, 'lime', 'red'), alpha=0.7)
    #ax1.bar(df["_ts"], df.close - df.low, 0.0001, bottom=df.low, color=np.where(df.close >= df.open, 'lime', 'red'), alpha=0.4)
    ax1.bar(df["_ts"], df.high - df.low, width, bottom=df.low, color=np.where(df.close >= df.open, 'lime', 'red'), alpha=0.4)

    # ax1.plot(df["_ts"], df["close"], color="white", alpha=0.3, label="Price")
    ax1.plot(df["_ts"], df["basis"], color="orange", alpha=0.3, label="Basis")
    ax1.plot(df["_ts"], df["upper"], color="cyan", alpha=0.3, label="Upper")
    ax1.plot(df["_ts"], df["lower"], color="cyan", alpha=0.3, label="Lower")
    ax1.plot(df["_ts"], df["super_upper"], color="magenta", alpha=0.3, label="Super Upper")
    ax1.plot(df["_ts"], df["super_lower"], color="magenta", alpha=0.3, label="Super Lower")
    ax1.plot(df["_ts"], df["test_upper"], color="blue", alpha=0.3, label="Test Upper")
    ax1.plot(df["_ts"], df["test_lower"], color="blue", alpha=0.3, label="Test Lower")
    ax1.plot(df["_ts"], df["tgt"], color="green", linestyle="--", label="Target")
    ax1.plot(df["_ts"], df["sl_line"], color="orange", linestyle=":", label="SL Line")
    ax1.plot(df["_ts"], df["trend_ema"], color="white", linestyle="--", label="Trend EMA")
    ax1.scatter(df["_ts"], df["entry_long"], marker="o", color="lime", s=20, alpha=0.5, label="Long")
    ax1.scatter(df["_ts"], df["entry_short"], marker="o", color="magenta", s=20, alpha=0.5, label="Short")
    ax1.scatter(df["_ts"], df["exit_tp"], marker="*", color="gold", s=150, label="TP")
    ax1.scatter(df["_ts"], df["exit_sl"], marker="x", color="red", s=100, label="SL")
    ax1.legend(loc="upper left", ncol=4); ax1.grid(alpha=0.1)
    ax2.plot(df["_ts"], df["equity"], color="cyan"); ax2.grid(alpha=0.1)
    save_path = Path(__file__).parent / f"reversal_2r_{coin}_{start.replace('-','')}.png"
    plt.savefig(save_path); plt.close(); print(f"  📸 저장완료: {save_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-30")
    ap.add_argument("--end", default="2026-06-01")
    args = ap.parse_args()
    for coin in ["btc", "eth", "sol", "xrp"]:
        raw = load_coin_raw(coin)
        df = raw[(raw.index >= args.start) & (raw.index < args.end)]
        days = (df.index[-1] - df.index[0]).total_seconds() / 86400
        m = run(df)
        
        print(f"\n{'='*62}")
        print(f"  {coin.upper()} {args.start}~{args.end}")
        print(f"{'='*62}")
        print(f"  거래수:   {m['n_trades']} (롱 {m['n_long']}/숏 {m['n_short']} | 2R {m['n_2r']}/trail {m['n_trail']}/SL {m['n_sl']})  TPD {m['n_trades']/days:.2f}")
        print(f"  총수익:   {m['total_return']:+.2f}%")
        print(f"  WR:       {m['win_rate']:.1f}%   MDD: {m['mdd']:.1f}%   PF: {m['profit_factor']:.3f}")
        print(f"  avg_win:  {m['avg_win']:+.3f}%   avg_loss: {m['avg_loss']:+.3f}%")

        if "df" in m: save_backtest_plot(m["df"], coin, args.start)

if __name__ == "__main__":
    main()