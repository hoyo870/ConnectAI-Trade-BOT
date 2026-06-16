"""
레버리지 스윕 + 선형성 체크
1x 기준으로 레버리지별 수익률/MDD 배율 확인
prod 프리셋, 4종목, 2026 OOS + 2026-06 + hist avg

Usage:
  python scripts/sweeps/leverage_sweep.py
  python scripts/sweeps/leverage_sweep.py --leverages 1 3 5 7 10
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
from config.af_params import get_preset

SEED     = 42
WINDOWS  = 10
WINDOW_D = 91
COINS    = ["btc", "eth", "sol", "xrp"]
OOS_START  = "2026-01-01"
OOS_END    = "2026-06-01"
JUNE_START = "2026-06-01"
JUNE_END   = "2026-06-17"


def run_sub(df, start, end, lev, cfg):
    sub = df[(df.index >= start) & (df.index < end)]
    if len(sub) < 100:
        return None
    return run_antifragile(sub, leverage=lev, max_dd_cb=0.99, **cfg)


def run_hist_avg(df, lev, cfg):
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
        r = run_antifragile(sub, leverage=lev, max_dd_cb=0.99, **cfg)
        rets.append(r["metrics"]["total_return"])
    return np.mean(rets) if rets else 0.0


def ratio_str(val, base):
    if base and base != 0 and val is not None:
        return f"{val/base:.2f}x"
    return "  -   "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leverages", nargs="+", type=int,
                        default=[1, 3, 5, 7, 10])
    parser.add_argument("--preset", default="prod",
                        choices=["prod", "stable", "aggressive", "conservative"])
    args = parser.parse_args()

    leverages = args.leverages
    cfg = get_preset(args.preset)

    print(f"데이터 로딩...")
    dfs = {}
    for coin in COINS:
        df = _normalize_index(load_coin_full(coin))
        dfs[coin] = df
        print(f"  {coin.upper()}: {df.index[0].date()} ~ {df.index[-1].date()}  ({len(df):,}행)")
    print()

    # ─── 기간별 수익률 + MDD 비교 ───
    for period_label, start, end in [
        ("2026 OOS (01-01~05-31)", OOS_START, OOS_END),
        ("2026-06 (06-01~06-16)",  JUNE_START, JUNE_END),
    ]:
        data = {}
        for lev in leverages:
            data[lev] = {}
            for coin in COINS:
                r = run_sub(dfs[coin], start, end, lev, cfg)
                data[lev][coin] = {
                    "ret": r["metrics"]["total_return"] if r else None,
                    "mdd": r["metrics"]["mdd"] if r else None,
                }
        base = data[leverages[0]]

        W = 100
        print("=" * W)
        print(f"  [{period_label}]  preset={args.preset}")
        print("=" * W)

        for metric in ("ret", "mdd"):
            label = "수익률" if metric == "ret" else "MDD"
            print(f"\n  {label}")
            hdr = f"  {'레버리지':>6}"
            for coin in COINS:
                hdr += f"  {coin.upper():>9} {'(배율)':>7}"
            print(hdr)
            print("  " + "─" * 90)
            for lev in leverages:
                d = data[lev]
                row = f"  {lev:>5}x"
                for coin in COINS:
                    v = d[coin][metric]
                    b = base[coin][metric]
                    v_s = f"{v:>+.1f}%" if v is not None else "  N/A "
                    r_s = ratio_str(v, b) if lev != leverages[0] else "  -   "
                    row += f"  {v_s:>9} {r_s:>7}"
                print(row)
        print()

    # ─── hist avg ───
    print("=" * 100)
    print(f"  [hist avg: {WINDOW_D}일 × {WINDOWS}창]  평균 수익률 + 배율  preset={args.preset}")
    print("=" * 100)
    hdr = f"  {'레버리지':>6}"
    for coin in COINS:
        hdr += f"  {coin.upper():>10} {'(배율)':>7}"
    print(hdr)
    print("  " + "─" * 95)

    hist_data = {}
    for lev in leverages:
        hist_data[lev] = {coin: run_hist_avg(dfs[coin], lev, cfg) for coin in COINS}

    base_hist = hist_data[leverages[0]]
    for lev in leverages:
        d = hist_data[lev]
        row = f"  {lev:>5}x"
        for coin in COINS:
            v = d[coin]
            b = base_hist[coin]
            v_s = f"{v:>+.0f}%"
            r_s = ratio_str(v, b) if lev != leverages[0] else "  -   "
            row += f"  {v_s:>10} {r_s:>7}"
        print(row)

    print()
    print("  ※ 완벽 선형이면 배율 = 레버리지값. 복리효과로 초과/미달 가능.")


if __name__ == "__main__":
    main()
