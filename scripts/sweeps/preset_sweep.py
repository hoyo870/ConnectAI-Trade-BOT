"""
종목 × 프리셋 스윕
기간: 2026 OOS (01-01~05-31), 2026-06, hist (91일×10창)
Usage:
  python scripts/sweeps/preset_sweep.py           # BTC만 (기본)
  python scripts/sweeps/preset_sweep.py --all     # 4종목 전체
"""
import sys, random, argparse
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT))

from backtest_antifragile import run_antifragile, remove_top_n, load_coin_full, _normalize_index
from config.af_params import PRESETS, get_preset, FEE_TOTAL

LEVERAGE  = 7
SEED      = 42
WINDOWS   = 10
WINDOW_D  = 91
OOS_START  = "2026-01-01"
OOS_END    = "2026-06-01"
JUNE_START = "2026-06-01"
JUNE_END   = "2026-06-17"


def run_period(df, start, end, cfg):
    sub = df[(df.index >= start) & (df.index < end)]
    if len(sub) < 100:
        return None
    return run_antifragile(sub, leverage=LEVERAGE, max_dd_cb=0.99, **cfg)


def fmt_row(label, r, days):
    if r is None:
        return f"  {label:<14}  {'N/A':>6}  {'N/A':>6}  {'N/A':>10}  {'N/A':>6}  {'N/A':>5}  {'N/A':>8}  {'N/A':>8}  ?"
    m   = r["metrics"]
    tl  = r["trade_log"]
    n   = m["n_trades"]
    ret = m["total_return"]
    mdd = m["mdd"]
    tpd = round(n / max(days, 1), 1)
    avg_t = ret / n if n > 0 else 0
    r5  = remove_top_n(tl, 5)
    ok  = [ret > 0, tpd >= 1.5, r5 > 0]
    mark = "✅" if all(ok) else ("⚠️" if sum(ok) >= 2 else "❌")
    return (f"  {label:<14}  {n:>6}  {m['win_rate']:>5.1f}%  {ret:>+9.1f}%"
            f"  {mdd:>5.1f}%  {tpd:>5.1f}  {avg_t:>+7.3f}%  {r5:>+7.1f}%  {mark}")


def run_hist(df, cfg):
    rng = random.Random(SEED)
    np.random.seed(SEED)
    total_days = (df.index[-1] - df.index[0]).days
    max_start  = total_days - WINDOW_D
    passes, avgs, avg_t_list = [], [], []
    for _ in range(WINDOWS):
        offset = rng.randint(0, max_start)
        s = df.index[0] + pd.Timedelta(days=offset)
        e = s + pd.Timedelta(days=WINDOW_D)
        sub = df[(df.index >= s) & (df.index < e)]
        if len(sub) < 500:
            continue
        r   = run_antifragile(sub, leverage=LEVERAGE, max_dd_cb=0.99, **cfg)
        m   = r["metrics"]
        tl  = r["trade_log"]
        n   = m["n_trades"]
        ret = m["total_return"]
        r5  = remove_top_n(tl, 5)
        ok  = sum([ret > 0, n / WINDOW_D >= 1.5, r5 > 0])
        passes.append(ok == 3)
        avgs.append(ret)
        avg_t_list.append(ret / n if n > 0 else 0)
    n_pass = sum(passes)
    mark = "✅" if n_pass >= 7 else ("⚠️" if n_pass >= 5 else "❌")
    return n_pass, np.mean(avgs) if avgs else 0, np.mean(avg_t_list) if avg_t_list else 0, mark


def sweep_coin(coin_key, coin_label):
    print(f"\n{'='*95}")
    print(f"  {coin_label} × 4 프리셋 스윕  |  레버리지 {LEVERAGE}x"
          f"  |  FEE={FEE_TOTAL*100:.3f}%/side  |  max_dd_cb 비활성화")
    print(f"{'='*95}")

    df = _normalize_index(load_coin_full(coin_key))

    HDR = (f"  {'프리셋':<14}  {'거래수':>6}  {'WR':>6}  {'수익률':>10}"
           f"  {'MDD':>6}  {'TPD':>5}  {'avg/건':>8}  {'Top5':>8}  판정")
    SEP = "  " + "─" * 90

    for period_label, start, end in [
        (f"2026 OOS: {OOS_START} ~ 2026-05-31", OOS_START, OOS_END),
        (f"2026-06: {JUNE_START} ~ 2026-06-16",  JUNE_START, JUNE_END),
    ]:
        days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        print(f"\n  [{period_label}]")
        print(SEP); print(HDR); print(SEP)
        for name in PRESETS:
            cfg = get_preset(name)
            r   = run_period(df, start, end, cfg)
            print(fmt_row(name, r, days))

    print(f"\n  [hist: {WINDOW_D}일 × {WINDOWS}창, seed={SEED}]")
    print(f"  {'프리셋':<14}  {'통과':>8}  {'avg수익':>10}  {'avg/건':>8}")
    print(SEP)
    for name in PRESETS:
        cfg = get_preset(name)
        n_pass, avg_ret, avg_t, mark = run_hist(df, cfg)
        print(f"  {name:<14}  {n_pass}/{WINDOWS} {mark}  {avg_ret:>+9.1f}%  {avg_t:>+7.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", default="btc",
                        choices=["btc", "eth", "sol", "xrp"],
                        help="스윕할 종목 (기본: btc)")
    parser.add_argument("--all", action="store_true", help="4종목 전체 스윕")
    args = parser.parse_args()

    targets = [("btc","BTC"),("eth","ETH"),("sol","SOL"),("xrp","XRP")] if args.all \
              else [(args.coin, args.coin.upper())]

    for coin_key, coin_label in targets:
        sweep_coin(coin_key, coin_label)
