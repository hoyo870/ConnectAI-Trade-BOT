"""
레버리지 선형성 체크: 1x 기준으로 3/5/7/10x가 배수대로 증가/감소하는지 확인
prod 프리셋, 4종목, 2026 OOS + 2026-06 + hist
"""
import sys, random
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from backtest_antifragile import run_antifragile, remove_top_n, load_coin_full, _normalize_index

SEED      = 42
WINDOWS   = 10
WINDOW_D  = 91
LEVERAGES = [1, 3, 5, 7, 10]

PRESET = dict(
    dt_rsi_lo=28, dt_rsi_hi=60, rg_rsi_lo=25, rg_rsi_hi=75,
    ut_rsi_lo=42, ut_rsi_hi=75, trail_atr_init=1.8, trail_atr_tight=2.0, add_levels=3,
)

OOS_START  = "2026-01-01"
OOS_END    = "2026-06-01"
JUNE_START = "2026-06-01"
JUNE_END   = "2026-06-17"
COINS      = ["btc", "eth", "sol", "xrp"]

# ─── 데이터 로드 ───
print("데이터 로딩...")
dfs = {}
for coin in COINS:
    df = _normalize_index(load_coin_full(coin))
    dfs[coin] = df
    print(f"  {coin.upper()}: {df.index[0].date()} ~ {df.index[-1].date()}  ({len(df):,}행)")
print()

def run_sub(df, start, end, lev):
    sub = df[(df.index >= start) & (df.index < end)]
    if len(sub) < 100:
        return None
    return run_antifragile(sub, leverage=lev, max_dd_cb=0.99, **PRESET)

def run_hist_avg(df, lev):
    rng = random.Random(SEED)
    np.random.seed(SEED)
    total_days = (df.index[-1] - df.index[0]).days
    max_start  = total_days - WINDOW_D
    rets = []
    for _ in range(WINDOWS):
        offset = rng.randint(0, max_start)
        s = df.index[0] + pd.Timedelta(days=offset)
        e = s + pd.Timedelta(days=WINDOW_D)
        sub = df[(df.index >= s) & (df.index < e)]
        if len(sub) < 500:
            continue
        r = run_antifragile(sub, leverage=lev, max_dd_cb=0.99, **PRESET)
        rets.append(r["metrics"]["total_return"])
    return np.mean(rets) if rets else 0.0

def ratio_str(val, base):
    if base and base != 0 and val is not None:
        return f"{val/base:.2f}x"
    return "-"

# ─── 기간별 분석 ───
for period_label, start, end in [
    ("2026 OOS (01-01~05-31)", OOS_START, OOS_END),
    ("2026-06 (06-01~06-16)",  JUNE_START, JUNE_END),
]:
    # 한 번만 계산 (수익률 + MDD 동시)
    data = {}  # lev -> {coin: {ret, mdd}}
    for lev in LEVERAGES:
        data[lev] = {}
        for coin in COINS:
            r = run_sub(dfs[coin], start, end, lev)
            if r:
                data[lev][coin] = {"ret": r["metrics"]["total_return"], "mdd": r["metrics"]["mdd"]}
            else:
                data[lev][coin] = {"ret": None, "mdd": None}

    base = data[1]

    W = 100
    print("=" * W)
    print(f"  [{period_label}]")
    print("=" * W)

    # 수익률 + 배율
    print(f"\n  수익률")
    print(f"  {'레버리지':>6}  {'BTC':>9} {'(배율)':>7}  {'ETH':>9} {'(배율)':>7}  {'SOL':>9} {'(배율)':>7}  {'XRP':>9} {'(배율)':>7}")
    print("  " + "─" * 90)
    for lev in LEVERAGES:
        d = data[lev]
        cols = []
        for coin in COINS:
            v = d[coin]["ret"]
            b = base[coin]["ret"]
            v_s = f"{v:>+.1f}%" if v is not None else "  N/A "
            r_s = ratio_str(v, b) if lev > 1 else "  -   "
            cols.append(f"{v_s:>9} {r_s:>7}")
        print(f"  {lev:>5}x  {'  '.join(cols)}")

    # MDD + 배율
    print(f"\n  MDD")
    print(f"  {'레버리지':>6}  {'BTC':>9} {'(배율)':>7}  {'ETH':>9} {'(배율)':>7}  {'SOL':>9} {'(배율)':>7}  {'XRP':>9} {'(배율)':>7}")
    print("  " + "─" * 90)
    for lev in LEVERAGES:
        d = data[lev]
        cols = []
        for coin in COINS:
            v = d[coin]["mdd"]
            b = base[coin]["mdd"]
            v_s = f"{v:>+.1f}%" if v is not None else "  N/A "
            r_s = ratio_str(v, b) if lev > 1 else "  -   "
            cols.append(f"{v_s:>9} {r_s:>7}")
        print(f"  {lev:>5}x  {'  '.join(cols)}")
    print()

# ─── hist avg ───
print("=" * 100)
print(f"  [hist avg: {WINDOW_D}일 × {WINDOWS}창]  평균 수익률 + 배율")
print("=" * 100)
print(f"  {'레버리지':>6}  {'BTC':>10} {'(배율)':>7}  {'ETH':>10} {'(배율)':>7}  {'SOL':>10} {'(배율)':>7}  {'XRP':>10} {'(배율)':>7}")
print("  " + "─" * 95)

hist_data = {}
for lev in LEVERAGES:
    hist_data[lev] = {coin: run_hist_avg(dfs[coin], lev) for coin in COINS}

base_hist = hist_data[1]
for lev in LEVERAGES:
    d = hist_data[lev]
    cols = []
    for coin in COINS:
        v = d[coin]
        b = base_hist[coin]
        v_s = f"{v:>+.0f}%"
        r_s = ratio_str(v, b) if lev > 1 else "  -   "
        cols.append(f"{v_s:>10} {r_s:>7}")
    print(f"  {lev:>5}x  {'  '.join(cols)}")

print()
print("  ※ 완벽 선형이면 배율 = 레버리지값. 복리효과로 초과/미달 가능.")
