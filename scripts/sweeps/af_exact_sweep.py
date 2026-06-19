"""
af_exact_sweep.py — AntifragileStrategy 기반 파라미터 스윕

검증 기간: 2026 OOS (01~05월), june2026 (06-01~현재), hist (91일×5창 seed=42)
종목: BTC, ETH, SOL, XRP 4종목

스윕 전략:
  Phase A — flip/cooling 스윕 (RSI·trail 고정)
  Phase B — trail 스윕 (최적 flip/cooling 고정)
  Phase C — RSI 스윕 (최적 trail·flip/cooling 고정)

Usage:
  python scripts/sweeps/af_exact_sweep.py           # 전체 스윕
  python scripts/sweeps/af_exact_sweep.py --phase A # Phase A만
  python scripts/sweeps/af_exact_sweep.py --top 10  # hist 검증 Top-N
"""
from __future__ import annotations
import sys, argparse, random, itertools
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.af_params import FEE_TOTAL, DEFAULT_PARAMS
from strategies.antifragile import AntifragileStrategy
from scripts.backtest_af_exact import add_indicators_exact, load_coin

COINS = ["btc", "eth", "sol", "xrp"]
HIST_START = {"btc": "2020-01-01", "eth": "2021-04-01",
              "sol": "2021-06-01", "xrp": "2020-06-01"}
OOS_START, OOS_END       = "2026-01-01", "2026-06-01"
JUNE_START, JUNE_END     = "2026-06-01", "2026-06-20"
HIST_WINDOW_DAYS         = 91
HIST_WINDOWS             = 5    # 속도 위해 5창 (full 검증 시 10창)
HIST_SEED                = 42
LEV                      = 7
INIT_CAP                 = 270.76


# ── 단일 파라미터 조합 평가 ─────────────────────────────────────────────────────

def _run_one(df: pd.DataFrame, p: dict) -> dict:
    strategy = AntifragileStrategy(p)
    capital  = INIT_CAP
    trade_log = []

    for idx in range(1, len(df)):
        row    = df.iloc[idx]
        price  = float(row["close"])
        result = strategy.process_tick(
            price, float(row["_atr"]), float(row["_rsi"]),
            bool(row["_trend_up"]), bool(row["_trend_down"]),
        )
        for ev in result["events"]:
            if ev["type"] == "close":
                pnl = max(ev["pnl_raw"] * LEV * ev["rr"], -ev["rr"]) - FEE_TOTAL
                capital *= (1 + pnl)
                trade_log.append(pnl)
            elif ev["type"] == "partial":
                ppnl = max(ev["pnl_raw"] * LEV * ev["rr"], -ev["rr"])
                capital *= (1 + ppnl)

    n = len(trade_log)
    ret = (capital - INIT_CAP) / INIT_CAP * 100
    days = max((df.index[-1] - df.index[0]).total_seconds() / 86400, 1)
    tpd = n / days
    top5_ok = False
    if n > 5:
        s = sorted(trade_log)
        r5 = (sum(trade_log) - sum(s[-5:])) * 100
        top5_ok = r5 > 0
    return {"ret": ret, "n": n, "tpd": tpd, "top5_ok": top5_ok, "days": days}


def score_combo(p: dict, dfs_2026: dict, dfs_june: dict) -> dict:
    """2026 OOS + june2026 점수 계산 (4코인 평균)."""
    rets_2026, rets_june, tpds = [], [], []
    for coin in COINS:
        r26 = _run_one(dfs_2026[coin], p)
        rjn = _run_one(dfs_june[coin], p)
        rets_2026.append(r26["ret"])
        rets_june.append(rjn["ret"])
        tpds.append(r26["tpd"])

    avg_2026 = np.mean(rets_2026)
    avg_june = np.mean(rets_june)
    avg_tpd  = np.mean(tpds)
    tpd_ok   = avg_tpd >= 1.5

    # 종합 점수: 2026 40% + june 20% + TPD 페널티
    score = avg_2026 * 0.40 + avg_june * 0.20
    if not tpd_ok:
        score -= 50

    return {
        "score": round(score, 2),
        "avg_2026": round(avg_2026, 1),
        "avg_june": round(avg_june, 1),
        "avg_tpd":  round(avg_tpd, 2),
        "rets_2026": [round(r, 1) for r in rets_2026],
        "rets_june": [round(r, 1) for r in rets_june],
    }


def hist_score(p: dict, all_dfs: dict) -> dict:
    """hist 5창 검증 (4코인 평균 통과율·수익률)."""
    rng = random.Random(HIST_SEED)
    passes, returns = [], []
    for coin in COINS:
        df_all  = all_dfs[coin]
        hs      = HIST_START.get(coin, "2020-01-01")
        possible = df_all[(df_all.index >= hs) &
                          (df_all.index <= df_all.index[-1] - pd.Timedelta(days=HIST_WINDOW_DAYS))].index
        chosen   = sorted(rng.choices(possible, k=HIST_WINDOWS))
        for sd in chosen:
            ed  = sd + pd.Timedelta(days=HIST_WINDOW_DAYS)
            seg = df_all[(df_all.index >= sd) & (df_all.index < ed)].copy()
            if len(seg) < 500:
                passes.append(0); returns.append(-100); continue
            r = _run_one(seg, p)
            ok = sum([r["ret"] > 0, r["tpd"] >= 1.5, r["top5_ok"]])
            passes.append(ok)
            returns.append(r["ret"])
    full_pass = sum(p == 3 for p in passes)
    return {
        "hist_pass3": full_pass,
        "hist_total": len(passes),
        "hist_avg":   round(np.mean(returns), 1),
        "hist_pos":   sum(r > 0 for r in returns),
    }


# ── 스윕 그리드 정의 ───────────────────────────────────────────────────────────

def _base_p(**overrides) -> dict:
    p = dict(DEFAULT_PARAMS)
    p.update(overrides)
    return p


def build_grid_phase_a():
    """Phase A: flip_bars × loss_limit × cooling_bars (RSI·trail 고정)"""
    combos = []
    for flip, loss_lim, cool in itertools.product([1, 2, 3], [2, 3, 4], [10, 15, 20, 25, 30]):
        combos.append(_base_p(flip_bars=flip, loss_limit=loss_lim, cooling_bars=cool))
    return combos


def build_grid_phase_b(best_a: dict):
    """Phase B: trail_init × trail_tight (최적 flip/cooling 적용)"""
    combos = []
    flip_kwargs = {k: best_a[k] for k in ("flip_bars", "loss_limit", "cooling_bars") if k in best_a}
    for ti, tt in itertools.product([1.5, 1.8, 2.0, 2.2], [2.0, 2.5, 3.0]):
        combos.append(_base_p(trail_atr_init=ti, trail_atr_tight=tt, **flip_kwargs))
    return combos


def build_grid_phase_c(best_ab: dict):
    """Phase C: RSI 파라미터 스윕 (trail + flip/cooling 고정)"""
    combos = []
    fixed = {k: best_ab[k] for k in ("trail_atr_init", "trail_atr_tight",
                                       "flip_bars", "loss_limit", "cooling_bars") if k in best_ab}
    for dt_lo, dt_hi, ut_lo, ut_hi in itertools.product(
        [25, 28, 30], [58, 60, 62], [40, 42, 44], [70, 73, 75]
    ):
        combos.append(_base_p(dt_rsi_lo=dt_lo, dt_rsi_hi=dt_hi,
                               ut_rsi_lo=ut_lo, ut_rsi_hi=ut_hi, **fixed))
    return combos


# ── 데이터 사전 로드 ───────────────────────────────────────────────────────────

def load_all_data():
    print("데이터 로딩 중...")
    dfs_2026, dfs_june, dfs_all = {}, {}, {}
    for coin in COINS:
        df_all = load_coin(coin)
        dfs_all[coin]  = df_all
        dfs_2026[coin] = df_all[(df_all.index >= OOS_START) & (df_all.index < OOS_END)].copy()
        dfs_june[coin] = df_all[(df_all.index >= JUNE_START) & (df_all.index < JUNE_END)].copy()
        print(f"  {coin.upper()}: {len(dfs_2026[coin])}봉 (2026) / {len(dfs_june[coin])}봉 (june)")
    return dfs_2026, dfs_june, dfs_all


# ── 스윕 실행 ──────────────────────────────────────────────────────────────────

def run_phase(name: str, combos: list[dict], dfs_2026: dict, dfs_june: dict,
              top_n: int = 5) -> list[dict]:
    print(f"\n{'═'*70}")
    print(f"  Phase {name} — {len(combos)}개 조합 평가")
    print(f"{'═'*70}")

    results = []
    for i, p in enumerate(combos, 1):
        sc = score_combo(p, dfs_2026, dfs_june)
        sc["params"] = p
        results.append(sc)
        if i % 20 == 0:
            print(f"  ... {i}/{len(combos)} 완료")

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n  Top-{top_n} 결과:")
    print(f"  {'#':>3}  {'score':>8}  {'2026avg':>8}  {'june':>7}  {'TPD':>5}  파라미터")
    print(f"  {'─'*70}")
    for rank, r in enumerate(results[:top_n], 1):
        p = r["params"]
        # 핵심 파라미터만 출력
        p_str = (f"trail={p.get('trail_atr_init','-')}/{p.get('trail_atr_tight','-')} "
                 f"dt={p.get('dt_rsi_lo','-')}/{p.get('dt_rsi_hi','-')} "
                 f"ut={p.get('ut_rsi_lo','-')}/{p.get('ut_rsi_hi','-')} "
                 f"flip={p.get('flip_bars','-')} loss={p.get('loss_limit','-')} cool={p.get('cooling_bars','-')}")
        print(f"  [{rank:02d}] {r['score']:>8.1f}  {r['avg_2026']:>8.1f}%  {r['avg_june']:>7.1f}%"
              f"  {r['avg_tpd']:>5.2f}  {p_str}")

    return results


def run_hist_phase(results: list[dict], dfs_all: dict, top_n: int = 10):
    print(f"\n{'═'*70}")
    print(f"  hist 검증 — Top-{top_n} 후보 ({HIST_WINDOWS}창×4코인)")
    print(f"{'═'*70}")

    candidates = results[:top_n]
    for i, r in enumerate(candidates, 1):
        h = hist_score(r["params"], dfs_all)
        r.update(h)
        p = r["params"]
        p_str = (f"trail={p.get('trail_atr_init','-')}/{p.get('trail_atr_tight','-')} "
                 f"flip={p.get('flip_bars','-')} cool={p.get('cooling_bars','-')}")
        print(f"  [{i:02d}] hist통과={h['hist_pass3']}/{h['hist_total']}  "
              f"avg={h['hist_avg']:+.1f}%  pos={h['hist_pos']}/{h['hist_total']}  {p_str}")


def print_best(best: dict):
    p = best["params"]
    print(f"\n{'█'*70}")
    print(f"  최적 파라미터 (score={best['score']:.1f})")
    print(f"{'█'*70}")
    for k, v in p.items():
        cur = DEFAULT_PARAMS.get(k, "N/A")
        marker = " ←변경" if v != cur else ""
        print(f"  {k:25s} = {v}{marker}")
    print(f"\n  2026 OOS: {best['avg_2026']:+.1f}%  (BTC={best['rets_2026'][0]:+.1f}% "
          f"ETH={best['rets_2026'][1]:+.1f}% SOL={best['rets_2026'][2]:+.1f}% XRP={best['rets_2026'][3]:+.1f}%)")
    print(f"  June2026: {best['avg_june']:+.1f}%  (BTC={best['rets_june'][0]:+.1f}% "
          f"ETH={best['rets_june'][1]:+.1f}% SOL={best['rets_june'][2]:+.1f}% XRP={best['rets_june'][3]:+.1f}%)")
    print(f"  TPD: {best['avg_tpd']:.2f}")
    if "hist_pass3" in best:
        print(f"  hist: {best['hist_pass3']}/{best['hist_total']} 통과  avg={best['hist_avg']:+.1f}%")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AntifragileStrategy 파라미터 스윕")
    parser.add_argument("--phase", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--top",   type=int, default=10, help="hist 검증 Top-N")
    args = parser.parse_args()

    dfs_2026, dfs_june, dfs_all = load_all_data()

    # 기준선: DEFAULT_PARAMS 성능 출력
    print("\n[기준선] DEFAULT_PARAMS 성능")
    base_sc = score_combo(dict(DEFAULT_PARAMS), dfs_2026, dfs_june)
    print(f"  2026={base_sc['avg_2026']:+.1f}%  june={base_sc['avg_june']:+.1f}%  TPD={base_sc['avg_tpd']:.2f}")

    best_a_params = dict(DEFAULT_PARAMS)
    best_b_params = dict(DEFAULT_PARAMS)

    if args.phase in ("A", "all"):
        combos_a   = build_grid_phase_a()
        results_a  = run_phase("A (flip/cooling)", combos_a, dfs_2026, dfs_june, top_n=args.top)
        run_hist_phase(results_a, dfs_all, top_n=args.top)
        best_a_params = results_a[0]["params"]

    if args.phase in ("B", "all"):
        combos_b   = build_grid_phase_b(best_a_params)
        results_b  = run_phase("B (trail)", combos_b, dfs_2026, dfs_june, top_n=args.top)
        run_hist_phase(results_b, dfs_all, top_n=args.top)
        best_b_params = results_b[0]["params"]

    if args.phase in ("C", "all"):
        combos_c   = build_grid_phase_c(best_b_params)
        results_c  = run_phase("C (RSI)", combos_c, dfs_2026, dfs_june, top_n=args.top)
        run_hist_phase(results_c, dfs_all, top_n=args.top)
        best_final = results_c[0]
        print_best(best_final)
    elif args.phase == "A":
        print_best(results_a[0])
    elif args.phase == "B":
        print_best(results_b[0])


if __name__ == "__main__":
    main()
