"""
Antifragile AdaptRSI Bayesian/black-box optimizer.

Optimizes the six RSI thresholds with historical 10-window validation as the
objective. 2026 OOS is reported for context only and is not used for selection.

Usage:
  python temp/scripts/37_af_rsi_optimizer.py
  python temp/scripts/37_af_rsi_optimizer.py --coin eth --n-trials 100
  python temp/scripts/37_af_rsi_optimizer.py --coin all --windows 10 --seed 42
  python temp/scripts/37_af_rsi_optimizer.py --backend scipy
"""
import argparse
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from backtest_antifragile import COIN_CONFIG, load_coin_full, remove_top_n, run_antifragile


SEARCH_SPACE = {
    "dt_lo": (12, 28),
    "dt_hi": (60, 75),   # 55 하한은 dt_hi=55 과적합 유도 → 60으로 올림
    "rg_lo": (22, 35),
    "rg_hi": (65, 80),
    "ut_lo": (30, 45),
    "ut_hi": (72, 90),
}

CURRENT_CFG = (22, 65, 30, 70, 35, 78)
COIN_MAP = {"both": ["btc", "eth"], "all": ["btc", "eth", "sol", "xrp"]}


@dataclass(frozen=True)
class EvalResult:
    score: float
    cfg: tuple
    hist_ret: float
    hist_mdd: float
    hist_pass: int
    hist_positive: int
    hist_tpd: float
    hist_pf: float
    oos_ret: float
    oos_mdd: float
    oos_tpd: float
    oos_pf: float
    oos_pass: int
    n_windows: int
    coin_rows: tuple


def _rsi_params(cfg):
    dt_lo, dt_hi, rg_lo, rg_hi, ut_lo, ut_hi = [int(x) for x in cfg]
    return {
        "dt_rsi_lo": dt_lo,
        "dt_rsi_hi": dt_hi,
        "rg_rsi_lo": rg_lo,
        "rg_rsi_hi": rg_hi,
        "ut_rsi_lo": ut_lo,
        "ut_rsi_hi": ut_hi,
    }


def _valid_cfg(cfg) -> bool:
    dt_lo, dt_hi, rg_lo, rg_hi, ut_lo, ut_hi = cfg
    return dt_lo < rg_lo < ut_lo and dt_hi < rg_hi < ut_hi


def _repair_cfg(values) -> tuple:
    """
    Convert unconstrained optimizer values to a valid integer threshold tuple.
    scipy proposes free coordinates; this keeps evaluation inside the requested
    bounds while enforcing dt < rg < ut for both low and high thresholds.
    """
    dt_lo = int(round(np.clip(values[0], *SEARCH_SPACE["dt_lo"])))
    rg_lo = int(round(np.clip(values[2], *SEARCH_SPACE["rg_lo"])))
    ut_lo = int(round(np.clip(values[4], *SEARCH_SPACE["ut_lo"])))
    lo = sorted([dt_lo, rg_lo, ut_lo])
    dt_lo = int(np.clip(lo[0], *SEARCH_SPACE["dt_lo"]))
    rg_lo = int(np.clip(max(lo[1], dt_lo + 1), *SEARCH_SPACE["rg_lo"]))
    ut_lo = int(np.clip(max(lo[2], rg_lo + 1), *SEARCH_SPACE["ut_lo"]))
    if not (dt_lo < rg_lo < ut_lo):
        rg_lo = int(np.clip(dt_lo + 1, *SEARCH_SPACE["rg_lo"]))
        ut_lo = int(np.clip(rg_lo + 1, *SEARCH_SPACE["ut_lo"]))

    dt_hi = int(round(np.clip(values[1], *SEARCH_SPACE["dt_hi"])))
    rg_hi = int(round(np.clip(values[3], *SEARCH_SPACE["rg_hi"])))
    ut_hi = int(round(np.clip(values[5], *SEARCH_SPACE["ut_hi"])))
    hi = sorted([dt_hi, rg_hi, ut_hi])
    dt_hi = int(np.clip(hi[0], *SEARCH_SPACE["dt_hi"]))
    rg_hi = int(np.clip(max(hi[1], dt_hi + 1), *SEARCH_SPACE["rg_hi"]))
    ut_hi = int(np.clip(max(hi[2], rg_hi + 1), *SEARCH_SPACE["ut_hi"]))
    if not (dt_hi < rg_hi < ut_hi):
        rg_hi = int(np.clip(dt_hi + 1, *SEARCH_SPACE["rg_hi"]))
        ut_hi = int(np.clip(rg_hi + 1, *SEARCH_SPACE["ut_hi"]))

    return (dt_lo, dt_hi, rg_lo, rg_hi, ut_lo, ut_hi)


def _sample_optuna_cfg(trial):
    dt_lo = trial.suggest_int("dt_lo", *SEARCH_SPACE["dt_lo"])
    rg_lo = trial.suggest_int("rg_lo", max(SEARCH_SPACE["rg_lo"][0], dt_lo + 1), SEARCH_SPACE["rg_lo"][1])
    ut_lo = trial.suggest_int("ut_lo", max(SEARCH_SPACE["ut_lo"][0], rg_lo + 1), SEARCH_SPACE["ut_lo"][1])

    dt_hi = trial.suggest_int("dt_hi", *SEARCH_SPACE["dt_hi"])
    rg_hi = trial.suggest_int("rg_hi", max(SEARCH_SPACE["rg_hi"][0], dt_hi + 1), SEARCH_SPACE["rg_hi"][1])
    ut_hi = trial.suggest_int("ut_hi", max(SEARCH_SPACE["ut_hi"][0], rg_hi + 1), SEARCH_SPACE["ut_hi"][1])
    return (dt_lo, dt_hi, rg_lo, rg_hi, ut_lo, ut_hi)


def _window_starts(all_df, hist_start, windows, window_days, seed):
    df = all_df.dropna(subset=["_rsi", "_atr"])
    end_limit = df.index[-1] - pd.Timedelta(days=window_days)
    possible = df[(df.index >= hist_start) & (df.index <= end_limit)].index
    if len(possible) == 0:
        raise ValueError(f"No historical windows available from {hist_start}")
    rng = random.Random(seed)
    return sorted(rng.choices(list(possible), k=windows))


def _metric_row(df, cfg):
    res = run_antifragile(df, **_rsi_params(cfg))
    m = res["metrics"]
    tl = res["trade_log"]
    days = max(len(df) / 288, 1)
    tpd = float(m.get("tpd", round(len(tl) / days, 2)))
    top5 = remove_top_n(tl, 5)
    ret = float(m.get("total_return", 0.0))
    ok = sum([ret > 0, tpd >= 1.5, top5 > 0])
    return {
        "ret": ret,
        "mdd": float(m.get("mdd", 0.0)),
        "tpd": tpd,
        "pf": float(m.get("profit_factor", 0.0)),
        "n": int(m.get("n_trades", len(tl))),
        "wr": float(m.get("win_rate", 0.0)),
        "top5": float(top5),
        "ok": int(ok),
        "long": int(m.get("long_cnt", 0)),
        "short": int(m.get("short_cnt", 0)),
    }


def load_context(coins, windows, window_days, seed):
    context = {}
    for coin in coins:
        info = COIN_CONFIG[coin]
        all_df = load_coin_full(coin).dropna(subset=["_rsi", "_atr"])
        starts = _window_starts(all_df, info["hist_start"], windows, window_days, seed)
        hist_segments = []
        for sd in starts:
            ed = sd + pd.Timedelta(days=window_days)
            seg = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
            if len(seg) >= 500:
                hist_segments.append((sd, ed, seg))
        oos = all_df[all_df.index >= "2026-01-01"].copy()
        context[coin] = {"label": info["label"], "hist": hist_segments, "oos": oos}
    return context


def evaluate_cfg(cfg, context, min_pass=7):
    if not _valid_cfg(cfg):
        return EvalResult(-1e12, cfg, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ())

    coin_rows = []
    total_score = 0.0

    for coin, c in context.items():
        hist_rows = [_metric_row(seg, cfg) for _, _, seg in c["hist"]]
        if not hist_rows:
            continue

        hist_ret = float(np.mean([r["ret"] for r in hist_rows]))
        hist_mdd = float(np.mean([r["mdd"] for r in hist_rows]))
        hist_tpd = float(np.mean([r["tpd"] for r in hist_rows]))
        hist_pf = float(np.mean([r["pf"] for r in hist_rows]))
        hist_pass = int(sum(r["ok"] == 3 for r in hist_rows))
        hist_positive = int(sum(r["ret"] > 0 for r in hist_rows))
        # 목적함수: pass_rate² × (return/MDD) × log(TPD_capped)
        # - log(min(TPD,12)): TPD 장려하되 12 이상은 보너스 없음 (과다 거래 억제)
        # - MDD ceiling: hist MDD > 7% 이면 점수 50% 페널티
        pass_ratio = hist_pass / max(len(hist_rows), 1)
        tpd_factor = np.log(max(min(hist_tpd, 12.0), 1.5))
        mdd_penalty = 0.5 if hist_mdd > 7.0 else 1.0
        score = (pass_ratio ** 2) * (hist_ret / max(hist_mdd, 1.0)) * tpd_factor * mdd_penalty

        oos_row = _metric_row(c["oos"], cfg) if len(c["oos"]) >= 500 else {
            "ret": 0.0, "mdd": 0.0, "tpd": 0.0, "pf": 0.0, "ok": 0,
            "n": 0, "wr": 0.0, "top5": 0.0, "long": 0, "short": 0,
        }

        total_score += score
        coin_rows.append({
            "coin": coin.upper(),
            "score": score,
            "hist_ret": hist_ret,
            "hist_mdd": hist_mdd,
            "hist_pass": hist_pass,
            "hist_positive": hist_positive,
            "hist_tpd": hist_tpd,
            "hist_pf": hist_pf,
            "oos_ret": oos_row["ret"],
            "oos_mdd": oos_row["mdd"],
            "oos_tpd": oos_row["tpd"],
            "oos_pf": oos_row["pf"],
            "oos_pass": oos_row["ok"],
            "n_windows": len(hist_rows),
        })

    if not coin_rows:
        return EvalResult(-1e12, cfg, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ())

    n = len(coin_rows)
    return EvalResult(
        score=float(total_score / n),
        cfg=cfg,
        hist_ret=float(np.mean([r["hist_ret"] for r in coin_rows])),
        hist_mdd=float(np.mean([r["hist_mdd"] for r in coin_rows])),
        hist_pass=int(round(np.mean([r["hist_pass"] for r in coin_rows]))),
        hist_positive=int(round(np.mean([r["hist_positive"] for r in coin_rows]))),
        hist_tpd=float(np.mean([r["hist_tpd"] for r in coin_rows])),
        hist_pf=float(np.mean([r["hist_pf"] for r in coin_rows])),
        oos_ret=float(np.mean([r["oos_ret"] for r in coin_rows])),
        oos_mdd=float(np.mean([r["oos_mdd"] for r in coin_rows])),
        oos_tpd=float(np.mean([r["oos_tpd"] for r in coin_rows])),
        oos_pf=float(np.mean([r["oos_pf"] for r in coin_rows])),
        oos_pass=int(round(np.mean([r["oos_pass"] for r in coin_rows]))),
        n_windows=int(round(np.mean([r["n_windows"] for r in coin_rows]))),
        coin_rows=tuple(coin_rows),
    )


def _dedupe_top(results, limit=5):
    best_by_cfg = {}
    for r in results:
        prev = best_by_cfg.get(r.cfg)
        if prev is None or r.score > prev.score:
            best_by_cfg[r.cfg] = r
    return sorted(best_by_cfg.values(), key=lambda x: x.score, reverse=True)[:limit]


def optimize_optuna(context, n_trials, seed, min_pass):
    import optuna

    results = []

    def objective(trial):
        cfg = _sample_optuna_cfg(trial)
        ev = evaluate_cfg(cfg, context, min_pass=min_pass)
        results.append(ev)
        trial.set_user_attr("cfg", cfg)
        trial.set_user_attr("hist_ret", ev.hist_ret)
        trial.set_user_attr("hist_mdd", ev.hist_mdd)
        trial.set_user_attr("hist_pass", ev.hist_pass)
        return ev.score

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return results


def optimize_scipy(context, n_trials, seed, min_pass):
    from scipy.optimize import dual_annealing

    results = []
    seen = {}

    bounds = [
        SEARCH_SPACE["dt_lo"], SEARCH_SPACE["dt_hi"],
        SEARCH_SPACE["rg_lo"], SEARCH_SPACE["rg_hi"],
        SEARCH_SPACE["ut_lo"], SEARCH_SPACE["ut_hi"],
    ]

    def objective(x):
        cfg = _repair_cfg(x)
        if cfg not in seen:
            seen[cfg] = evaluate_cfg(cfg, context, min_pass=min_pass)
            results.append(seen[cfg])
        return -seen[cfg].score

    dual_annealing(
        objective,
        bounds=bounds,
        seed=seed,
        maxfun=max(1, n_trials),
        no_local_search=True,
    )
    return results


def print_result_table(results, title, show_coin_rows=False):
    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")
    print(
        f"  {'rank':>4} {'score':>8} {'dt':>7} {'rg':>7} {'ut':>7} "
        f"{'hist_ret':>10} {'hist_mdd':>9} {'pass':>7} {'pos':>6} "
        f"{'hTPD':>6} {'hPF':>7} {'2026_ret':>10} {'2026_mdd':>9} {'oTPD':>6} {'oPF':>7}"
    )
    print(f"  {'-' * 126}")
    for i, r in enumerate(results, 1):
        dt = f"{r.cfg[0]}/{r.cfg[1]}"
        rg = f"{r.cfg[2]}/{r.cfg[3]}"
        ut = f"{r.cfg[4]}/{r.cfg[5]}"
        print(
            f"  {i:>4} {r.score:>8.2f} {dt:>7} {rg:>7} {ut:>7} "
            f"{r.hist_ret:>+9.1f}% {r.hist_mdd:>8.2f}% "
            f"{r.hist_pass:>2}/{r.n_windows:<2} {r.hist_positive:>2}/{r.n_windows:<2} "
            f"{r.hist_tpd:>6.2f} {r.hist_pf:>7.2f} "
            f"{r.oos_ret:>+9.1f}% {r.oos_mdd:>8.2f}% {r.oos_tpd:>6.2f} {r.oos_pf:>7.2f}"
        )
        if show_coin_rows and len(r.coin_rows) > 1:
            for cr in r.coin_rows:
                print(
                    f"       {cr['coin']:<4} {cr['score']:>8.2f} {'':>7} {'':>7} {'':>7} "
                    f"{cr['hist_ret']:>+9.1f}% {cr['hist_mdd']:>8.2f}% "
                    f"{cr['hist_pass']:>2}/{cr['n_windows']:<2} {cr['hist_positive']:>2}/{cr['n_windows']:<2} "
                    f"{cr['hist_tpd']:>6.2f} {cr['hist_pf']:>7.2f} "
                    f"{cr['oos_ret']:>+9.1f}% {cr['oos_mdd']:>8.2f}% {cr['oos_tpd']:>6.2f} {cr['oos_pf']:>7.2f}"
                )


def main():
    parser = argparse.ArgumentParser(description="Optimize Antifragile AdaptRSI thresholds")
    parser.add_argument("--coin", default="btc", choices=["btc", "eth", "sol", "xrp", "both", "all"])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--window-days", type=int, default=91)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pass", type=int, default=7)
    parser.add_argument("--backend", choices=["auto", "optuna", "scipy"], default="auto")
    parser.add_argument("--show-coin-rows", action="store_true")
    args = parser.parse_args()

    coins = COIN_MAP.get(args.coin, [args.coin])
    min_pass = min(args.min_pass, args.windows)
    show_coin_rows = args.show_coin_rows or len(coins) > 1
    print(f"\nCoins: {', '.join(c.upper() for c in coins)}")
    print(f"Hist windows: {args.windows} x {args.window_days}d, seed={args.seed}")
    print("Objective: mean(hist_avg_return / max(hist_avg_mdd, 1.0)); penalty if hist_pass < "
          f"{min_pass}/{args.windows}")

    context = load_context(coins, args.windows, args.window_days, args.seed)

    baseline = evaluate_cfg(CURRENT_CFG, context, min_pass=min_pass)
    print_result_table([baseline], "Current live config baseline", show_coin_rows=show_coin_rows)

    backend = args.backend
    if backend == "auto":
        try:
            import optuna  # noqa: F401
            backend = "optuna"
        except Exception:
            backend = "scipy"

    print(f"\nOptimizer backend: {backend}  n_trials={args.n_trials}")
    if backend == "optuna":
        results = optimize_optuna(context, args.n_trials, args.seed, min_pass)
    else:
        results = optimize_scipy(context, args.n_trials, args.seed, min_pass)

    top5 = _dedupe_top(results, limit=5)
    print_result_table(top5, "Top-5 optimized configs", show_coin_rows=show_coin_rows)

    if top5:
        best = top5[0]
        print("\nBest params:")
        for k, v in _rsi_params(best.cfg).items():
            print(f"  {k}={v}")


if __name__ == "__main__":
    main()
