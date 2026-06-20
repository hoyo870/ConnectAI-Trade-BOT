"""
배치 백테스트 — 4기간 × 4코인 × 4레버리지 (ML 앙상블 진입 필터 포함)

기준: backtest_af_ml.py의 run_antifragile_ml() 로직 그대로 사용.
결과 컬럼: 거래수 | WR | TPD | 일수익 | 총수익 | MDD | PF

Usage:
  python scripts/batch_backtest.py --model models/af_ensemble/saved/
  python scripts/batch_backtest.py --model models/af_ensemble/saved/ --coin btc
  python scripts/batch_backtest.py --no-ml --coin all --mode 2026
"""
import os, sys, argparse, random
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from config.af_params import DEFAULT_PARAMS
from config.loader import load_coin_raw, COIN_CONFIG
from scripts.backtest_antifragile import remove_top_n
from scripts.backtest_af_ml import run_antifragile_ml, load_coin_ml
from models.af_ensemble.ensemble import AFEnsemble

COINS     = ["btc", "eth", "sol", "xrp"]
LEVERAGES = [3, 5, 7, 10]

PERIODS = [
    ("2026년도",      "2026-01-01", "2026-06-01"),
    ("2026-06",       "2026-06-01", "2026-07-01"),
    ("2026-06-12~19", "2026-06-12", "2026-06-20"),
]
HIST_WINDOWS = 10
HIST_DAYS    = 91
HIST_SEED    = 42


def _slice(df, start, end):
    s = df[df.index >= start] if start else df
    return (s[s.index < end] if end else s).copy()


def run_period(df, ensemble, lev):
    res  = run_antifragile_ml(df, ensemble, leverage=lev)
    m    = res["metrics"]
    days = max(len(df) / 288, 1)
    return {
        "trades":    m["n_trades"],
        "wr":        round(m["win_rate"], 1),
        "tpd":       round(m.get("tpd", m["n_trades"] / days), 2),
        "daily_ret": round(m["total_return"] / days, 2),
        "total_ret": round(m["total_return"], 1),
        "mdd":       round(m["mdd"], 1),
        "pf":        round(m.get("profit_factor", 0.0), 3),
    }


def run_hist(df_all, coin, ensemble, lev):
    rng = random.Random(HIST_SEED)
    df  = df_all.dropna(subset=["_rsi", "_atr"]).copy()
    hist_start = COIN_CONFIG.get(coin, {}).get("hist_start", "2020-01-01")
    possible = df[(df.index >= hist_start) &
                  (df.index <= df.index[-1] - pd.Timedelta(days=HIST_DAYS))].index
    if len(possible) < HIST_WINDOWS:
        return None
    chosen = sorted(rng.choices(possible, k=HIST_WINDOWS))

    rows = []
    for sd in chosen:
        ed  = sd + pd.Timedelta(days=HIST_DAYS)
        seg = df[(df.index >= sd) & (df.index < ed)].copy()
        if len(seg) < 500:
            continue
        res = run_antifragile_ml(seg, ensemble, leverage=lev)
        m   = res["metrics"]
        tl  = res["trade_log"]
        tpd = m.get("tpd", m["n_trades"] / HIST_DAYS)
        r5  = remove_top_n(tl, 5) if len(tl) > 5 else sum(t["pnl"] for t in tl) * 100
        ok  = sum([m["total_return"] > 0, tpd >= 1.5, r5 > 0])
        rows.append({"pass": ok == 3, "trades": m["n_trades"], "wr": m["win_rate"],
                     "tpd": tpd, "total_ret": m["total_return"], "mdd": m["mdd"],
                     "pf": m.get("profit_factor", 0.0)})

    if not rows:
        return None
    pass_cnt = sum(r["pass"] for r in rows)
    return {
        "trades":    int(round(np.mean([r["trades"]    for r in rows]))),
        "wr":        round(np.mean([r["wr"]        for r in rows]), 1),
        "tpd":       round(np.mean([r["tpd"]       for r in rows]), 2),
        "daily_ret": round(np.mean([r["total_ret"] / HIST_DAYS for r in rows]), 2),
        "total_ret": round(np.mean([r["total_ret"] for r in rows]), 1),
        "mdd":       round(np.mean([r["mdd"]       for r in rows]), 1),
        "pf":        round(np.mean([r["pf"]        for r in rows]), 3),
        "hist_pass": f"{pass_cnt}/{len(rows)}",
    }


def _header():
    print(f"  {'구분':<22} {'거래수':>5}  {'WR':>6}  {'TPD':>5}  {'일수익':>8}  "
          f"{'총수익':<18}  {'MDD':>6}  {'PF':>6}")
    print(f"  {'-'*22} {'-'*5}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*18}  {'-'*6}  {'-'*6}")


def _row(label, d, hist_pass=None):
    tr = f"{d['total_ret']:+.1f}%"
    if hist_pass:
        tr += f" ({hist_pass})"
    print(f"  {label:<22} {d['trades']:>5}  {d['wr']:>5.1f}%  {d['tpd']:>5.2f}  "
          f"{d['daily_ret']:>+7.2f}%  {tr:<18}  {d['mdd']:>5.1f}%  {d['pf']:>6.3f}")


def main():
    parser = argparse.ArgumentParser(description="배치 백테스트 — run_antifragile_ml 기준")
    parser.add_argument("--model",    default="models/af_ensemble/saved/",
                        help="AFEnsemble 저장 디렉터리")
    parser.add_argument("--coin",     default="all",
                        choices=["btc", "eth", "sol", "xrp", "all"])
    parser.add_argument("--mode",     default="all",
                        choices=["all", "2026", "june", "hist"])
    parser.add_argument("--leverage", nargs="+", type=int, default=LEVERAGES, metavar="N")
    parser.add_argument("--no-ml",   action="store_true", help="ML 필터 비활성")
    args = parser.parse_args()

    coins = COINS if args.coin == "all" else [args.coin]
    levs  = args.leverage

    ensemble  = None
    model_tag = "ML 비활성"
    if not args.no_ml:
        try:
            ensemble  = AFEnsemble.load(args.model)
            model_tag = f"ML 활성 (theta={ensemble.threshold:.3f})"
        except Exception as e:
            print(f"⚠️  ML 모델 로드 실패: {e} → ML 없이 실행")

    print(f"\n{'█'*72}")
    print(f"  Antifragile + {model_tag}")
    print(f"  기준: run_antifragile_ml()  |  레버리지: {levs}")
    print(f"{'█'*72}")

    for coin in coins:
        label = coin.upper()
        print(f"\n{'═'*72}")
        print(f"  {label}/USDT")
        print(f"{'═'*72}")
        print(f"  데이터 로드 중...")
        df_all = load_coin_ml(coin)
        print(f"  {df_all.index[0].date()} ~ {df_all.index[-1].date()}  ({len(df_all):,}행)")

        do_2026 = args.mode in ("all", "2026")
        do_june = args.mode in ("all", "june")
        do_hist = args.mode in ("all", "hist")

        for pname, start, end in PERIODS:
            if pname == "2026년도"      and not do_2026: continue
            if pname == "2026-06"       and not do_june: continue
            if pname == "2026-06-12~19" and not do_june: continue

            seg = _slice(df_all, start, end)
            if len(seg) < 50:
                print(f"\n  [{pname}] 데이터 부족 ({len(seg)}봉)")
                continue

            days = (seg.index[-1] - seg.index[0]).days
            print(f"\n  [{pname}]  {seg.index[0].date()} ~ {seg.index[-1].date()}  ({days}일)")
            _header()
            for lev in levs:
                r = run_period(seg, ensemble, lev)
                _row(f"LEV×{lev}", r)

        if do_hist:
            print(f"\n  [hist]  random {HIST_WINDOWS}×{HIST_DAYS}일  seed={HIST_SEED}")
            _header()
            for lev in levs:
                r = run_hist(df_all, coin, ensemble, lev)
                if r:
                    _row(f"LEV×{lev}", r, hist_pass=r["hist_pass"])
                else:
                    print(f"  LEV×{lev:<2}                  데이터 부족")

    print(f"\n{'═'*72}")
    print("  완료")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
