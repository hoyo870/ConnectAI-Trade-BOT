"""
BTC × 4 프리셋 스윕
기간: 2026 OOS (01-01~05-31), 2026-06 (06-01~06-16), hist (91일×10창)
레버리지: 7x (현재 설정)
"""
import sys, random
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from backtest_antifragile import run_antifragile, remove_top_n, load_coin_full, run_random_validation, _normalize_index

LEVERAGE   = 7
SEED       = 42
WINDOWS    = 10
WINDOW_D   = 91
FEE        = 0.00111

PRESETS = {
    "prod":         dict(dt_rsi_lo=28, dt_rsi_hi=60, rg_rsi_lo=25, rg_rsi_hi=75,
                         ut_rsi_lo=42, ut_rsi_hi=75, trail_atr_init=1.8, trail_atr_tight=2.0, add_levels=3),
    "stable":       dict(dt_rsi_lo=30, dt_rsi_hi=60, rg_rsi_lo=25, rg_rsi_hi=75,
                         ut_rsi_lo=42, ut_rsi_hi=70, trail_atr_init=1.5, trail_atr_tight=2.0, add_levels=4),
    "aggressive":   dict(dt_rsi_lo=25, dt_rsi_hi=60, rg_rsi_lo=25, rg_rsi_hi=75,
                         ut_rsi_lo=42, ut_rsi_hi=78, trail_atr_init=0.8, trail_atr_tight=1.5, add_levels=4),
    "conservative": dict(dt_rsi_lo=28, dt_rsi_hi=70, rg_rsi_lo=25, rg_rsi_hi=75,
                         ut_rsi_lo=42, ut_rsi_hi=78, trail_atr_init=2.0, trail_atr_tight=2.5, add_levels=3),
}

OOS_START  = "2026-01-01"
OOS_END    = "2026-06-01"   # exclusive
JUNE_START = "2026-06-01"
JUNE_END   = "2026-06-17"   # exclusive

def run_period(df, start, end, cfg):
    sub = df[(df.index >= start) & (df.index < end)]
    if len(sub) < 100:
        return None
    r = run_antifragile(sub, leverage=LEVERAGE, max_dd_cb=0.99, **cfg)
    return r

def fmt_row(label, r, days):
    if r is None:
        return f"  {label:<14}  {'N/A':>6}  {'N/A':>6}  {'N/A':>10}  {'N/A':>6}  {'N/A':>5}  {'N/A':>8}  {'N/A':>8}  ?"
    m  = r["metrics"]
    tl = r["trade_log"]
    n  = m["n_trades"]
    wr = m["win_rate"]
    ret = m["total_return"]
    mdd = m["mdd"]
    tpd = round(n / max(days, 1), 1)
    avg_t = ret / n if n > 0 else 0
    r5  = remove_top_n(tl, 5)

    ok_ret = ret > 0
    ok_tpd = tpd >= 1.5
    ok_top = r5 > 0
    mark = "✅" if (ok_ret and ok_tpd and ok_top) else ("⚠️" if sum([ok_ret, ok_tpd, ok_top]) >= 2 else "❌")

    return (f"  {label:<14}  {n:>6}  {wr:>5.1f}%  {ret:>+9.1f}%  {mdd:>5.1f}%  {tpd:>5.1f}"
            f"  {avg_t:>+7.3f}%  {r5:>+7.1f}%  {mark}")

def run_hist(df, cfg):
    rng = random.Random(SEED)
    np.random.seed(SEED)
    total_days = (df.index[-1] - df.index[0]).days
    max_start  = total_days - WINDOW_D
    passes, avgs, avg_trades = [], [], []
    for _ in range(WINDOWS):
        offset = rng.randint(0, max_start)
        s = df.index[0] + pd.Timedelta(days=offset)
        e = s + pd.Timedelta(days=WINDOW_D)
        sub = df[(df.index >= s) & (df.index < e)]
        if len(sub) < 500:
            continue
        r = run_antifragile(sub, leverage=LEVERAGE, max_dd_cb=0.99, **cfg)
        m = r["metrics"]
        tl = r["trade_log"]
        n  = m["n_trades"]
        ret = m["total_return"]
        tpd = n / WINDOW_D
        r5  = remove_top_n(tl, 5)
        ok = sum([ret > 0, tpd >= 1.5, r5 > 0])
        passes.append(ok == 3)
        avgs.append(ret)
        avg_trades.append(ret / n if n > 0 else 0)
    n_pass = sum(passes)
    avg_ret = np.mean(avgs) if avgs else 0
    avg_t   = np.mean(avg_trades) if avg_trades else 0
    hist_ok = "✅" if n_pass >= 7 else ("⚠️" if n_pass >= 5 else "❌")
    return n_pass, avg_ret, avg_t, hist_ok

# ─── 데이터 로드 ───
print("BTC 데이터 로딩...")
df_btc = _normalize_index(load_coin_full("btc"))
print(f"  {df_btc.index[0].date()} ~ {df_btc.index[-1].date()}  ({len(df_btc):,}행)\n")

# ─── 헤더 ───
HDR = f"  {'프리셋':<14}  {'거래수':>6}  {'WR':>6}  {'수익률':>10}  {'MDD':>6}  {'TPD':>5}  {'avg/건':>8}  {'Top5':>8}  판정"
SEP = "  " + "─" * 90

print("=" * 95)
print(f"  BTC × 4 프리셋 스윕  |  레버리지 {LEVERAGE}x  |  FEE={FEE*100:.3f}%/side  |  max_dd_cb 비활성화")
print("=" * 95)

# ─── 2026 OOS ───
print(f"\n  [2026 OOS: {OOS_START} ~ 2026-05-31]")
print(SEP)
print(HDR)
print(SEP)
oos_days = (pd.Timestamp(OOS_END) - pd.Timestamp(OOS_START)).days
for name, cfg in PRESETS.items():
    r = run_period(df_btc, OOS_START, OOS_END, cfg)
    print(fmt_row(name, r, oos_days))

# ─── 2026-06 ───
print(f"\n  [2026-06: {JUNE_START} ~ 2026-06-16]")
print(SEP)
print(HDR)
print(SEP)
june_days = (pd.Timestamp(JUNE_END) - pd.Timestamp(JUNE_START)).days
for name, cfg in PRESETS.items():
    r = run_period(df_btc, JUNE_START, JUNE_END, cfg)
    print(fmt_row(name, r, june_days))

# ─── hist ───
print(f"\n  [hist: {WINDOW_D}일 × {WINDOWS}창, seed={SEED}]")
print(f"  {'프리셋':<14}  {'통과':>8}  {'avg수익':>10}  {'avg/건':>8}")
print(SEP)
for name, cfg in PRESETS.items():
    n_pass, avg_ret, avg_t, hist_ok = run_hist(df_btc, cfg)
    print(f"  {name:<14}  {n_pass}/{WINDOWS} {hist_ok}  {avg_ret:>+9.1f}%  {avg_t:>+7.3f}%")

print("\n" + "=" * 95)
