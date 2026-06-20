"""
[DEPRECATED] backtest_af_ml.py — 구형 ML 백테스트 (2026-06-20 이후 사용 중단)

대체 스크립트:
  python scripts/backtest_af_exact.py --mode 2026 --coin btc --model models/af_ensemble/saved/
  python scripts/batch_backtest.py    --model models/af_ensemble/saved/

이 파일은 하위 호환성을 위해 유지되며 더 이상 수정/개선되지 않습니다.
이 파일의 run_antifragile_ml()은 AntifragileStrategy를 사용하지 않으므로
live_trader.py 실거래 로직과 다릅니다 (partial/flip/cooling/bb_sigma 불일치).

Usage (레거시):
  python scripts/backtest_af_ml.py --mode 2026 --coin btc --model models/af_ensemble/saved/
  python scripts/backtest_af_ml.py --mode random --windows 10 --coin all --model models/af_ensemble/saved/
"""
import os, sys
# LightGBM과 PyTorch OpenMP 라이브러리 충돌(세그폴트) 방지
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, "src")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hybrid_engine import compute_metrics
from config.af_params import FEE_TOTAL, PRESETS, get_preset, DEFAULT_PARAMS
from config.loader import load_coin_raw, COIN_CONFIG
from scripts.backtest_antifragile import add_indicators, print_result, remove_top_n
from models.af_ensemble.feature_extractor import add_ml_features, extract_snapshot, extract_sequence
from models.af_ensemble.ensemble import AFEnsemble


def run_antifragile_ml(
    df,
    ensemble: AFEnsemble,
    initial_capital=10_000.0,
    require_bb=False,
    leverage=7,
    max_hold_bars=288,
    cooling_bars=100,
    max_dd_cb=0.99,
    seq_len=20,
    **kwargs,
):
    """run_antifragile() 미러 + 진입 시 ML 필터 거부 훅."""
    p = {**DEFAULT_PARAMS, **kwargs}
    dt_rsi_lo = p["dt_rsi_lo"]; dt_rsi_hi = p["dt_rsi_hi"]
    rg_rsi_lo = p["rg_rsi_lo"]; rg_rsi_hi = p["rg_rsi_hi"]
    ut_rsi_lo = p["ut_rsi_lo"]; ut_rsi_hi = p["ut_rsi_hi"]
    rr_base = p["rr_base"]; rr_add = p["rr_add"]
    add_levels = p["add_levels"]; atr_add_step = p["atr_add_step"]
    trail_atr_init = p["trail_atr_init"]; trail_atr_tight = p["trail_atr_tight"]

    # 시간 피처(hour/dow)를 위해 DatetimeIndex 보존 (원본은 reset_index)
    df = df.dropna(subset=["_rsi", "_atr"]).copy()

    capital = initial_capital
    peak_cap = initial_capital
    pos = 0
    entry_price = 0.0
    entry_atr = 0.0
    current_rr = 0.0
    add_count = 0
    trail_sl = 0.0
    peak_price = 0.0
    entry_bar = 0
    cooling_left = 0
    cb_triggers = 0

    loss_streak_long = 0
    loss_streak_short = 0
    pending_dir = 0  # 현재 포지션 진입 방향 (loss_streak 업데이트용)

    equity_curve = [capital]
    trade_log = []

    def update_loss_streak(pnl, direction):
        nonlocal loss_streak_long, loss_streak_short
        if pnl <= 0:
            if direction == 1:
                loss_streak_long += 1
            else:
                loss_streak_short += 1
        else:
            if direction == 1:
                loss_streak_long = 0
            else:
                loss_streak_short = 0

    n = len(df)
    closes = df["close"].to_numpy(dtype=float)
    rsis = df["_rsi"].to_numpy(dtype=float)
    atrs = df["_atr"].to_numpy(dtype=float)
    bb_us = df["_bb_upper"].to_numpy(dtype=float) if "_bb_upper" in df.columns else None
    bb_ls = df["_bb_lower"].to_numpy(dtype=float) if "_bb_lower" in df.columns else None
    tups = df["_trend_up"].to_numpy(dtype=int) if "_trend_up" in df.columns else np.zeros(n, dtype=int)
    tdns = df["_trend_down"].to_numpy(dtype=int) if "_trend_down" in df.columns else np.zeros(n, dtype=int)

    for idx in range(1, n):
        price = closes[idx]
        rsi = rsis[idx]
        atr = atrs[idx]
        bb_u = bb_us[idx] if bb_us is not None and not np.isnan(bb_us[idx]) else price * 1.02
        bb_l = bb_ls[idx] if bb_ls is not None and not np.isnan(bb_ls[idx]) else price * 0.98
        tup = tups[idx]
        tdn = tdns[idx]

        rsi_lo = dt_rsi_lo if tdn else (ut_rsi_lo if tup else rg_rsi_lo)
        rsi_hi = dt_rsi_hi if tdn else (ut_rsi_hi if tup else rg_rsi_hi)

        if pos != 0:
            unr = pos * (price - entry_price) / (entry_price + 1e-9)
            equity = capital * (1 + unr * leverage * current_rr)
        else:
            equity = capital
        peak_cap = max(peak_cap, equity)
        dd = (peak_cap - equity) / (peak_cap + 1e-9)

        if dd > max_dd_cb and cooling_left == 0:
            cooling_left = cooling_bars; cb_triggers += 1
            if pos != 0:
                cp = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_bar,
                                   "entry_bar": entry_bar, "exit_bar": idx,
                                   "lev": leverage, "rr": current_rr, "forced": True, "direction": pos})
                update_loss_streak(pnl, pos)
                pos = 0

        if cooling_left > 0:
            cooling_left -= 1
            if pos != 0:
                cp = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_bar,
                                   "entry_bar": entry_bar, "exit_bar": idx,
                                   "lev": leverage, "rr": current_rr, "forced": True, "direction": pos})
                update_loss_streak(pnl, pos)
                pos = 0
            equity_curve.append(capital); continue

        if pos != 0:
            hold = idx - entry_bar
            effective_atr = max(atr, entry_atr * 0.6)
            if pos == 1:
                peak_price = max(peak_price, price)
                trail_mult = trail_atr_tight if add_count > 0 else trail_atr_init
                trail_sl = max(trail_sl, peak_price - trail_mult * effective_atr)
                hit_stop = price <= trail_sl
            else:
                peak_price = min(peak_price, price)
                trail_mult = trail_atr_tight if add_count > 0 else trail_atr_init
                trail_sl = min(trail_sl, peak_price + trail_mult * effective_atr)
                hit_stop = price >= trail_sl

            if hit_stop or hold >= max_hold_bars:
                cp = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": hold,
                                   "entry_bar": entry_bar, "exit_bar": idx,
                                   "lev": leverage, "rr": current_rr, "forced": False, "direction": pos})
                update_loss_streak(pnl, pos)
                pos = 0; add_count = 0; current_rr = 0.0
            else:
                favorable_move = pos * (price - entry_price) / (atr + 1e-9)
                next_add_level = (add_count + 1) * atr_add_step
                if add_count < add_levels and favorable_move >= next_add_level:
                    current_rr += rr_add; add_count += 1
                    if pos == 1:
                        trail_sl = max(trail_sl, price - trail_atr_tight * effective_atr)
                    else:
                        trail_sl = min(trail_sl, price + trail_atr_tight * effective_atr)

        # 신규 진입 + ML 필터
        if pos == 0:
            long_ok = (rsi <= rsi_lo) and ((not require_bb) or (price <= bb_l))
            short_ok = (rsi >= rsi_hi) and ((not require_bb) or (price >= bb_u))

            if (long_ok or short_ok) and atr < price * 0.0015:
                long_ok = short_ok = False

            if long_ok or short_ok:
                direction = 1 if long_ok else -1
                loss_streak = loss_streak_long if direction == 1 else loss_streak_short
                snapshot = extract_snapshot(df, idx, direction, rsi_lo, rsi_hi, loss_streak)
                sequence = extract_sequence(df, idx, seq_len)
                if not ensemble.should_enter(snapshot, sequence):
                    long_ok = short_ok = False  # ML 거부

            if long_ok:
                ep = price * (1 + FEE_TOTAL)
                entry_price = ep; entry_atr = atr; current_rr = rr_base; add_count = 0
                trail_sl = ep - trail_atr_init * atr; peak_price = ep
                pos = 1; entry_bar = idx
            elif short_ok:
                ep = price * (1 - FEE_TOTAL)
                entry_price = ep; entry_atr = atr; current_rr = rr_base; add_count = 0
                trail_sl = ep + trail_atr_init * atr; peak_price = ep
                pos = -1; entry_bar = idx

        equity_curve.append(capital)

    m = compute_metrics(equity_curve, trade_log)
    m["cb_triggers"] = cb_triggers
    if trade_log:
        days_total = len(df) / 288
        m["tpd"] = round(len(trade_log) / days_total, 2)
        m["long_cnt"] = sum(1 for t in trade_log if t.get("direction") == 1)
        m["short_cnt"] = sum(1 for t in trade_log if t.get("direction") == -1)
        wins = [t["pnl"] for t in trade_log if t["pnl"] > 0]
        losss = [t["pnl"] for t in trade_log if t["pnl"] < 0]
        if wins: m["avg_win"] = round(float(np.mean(wins)) * 100, 4)
        if losss: m["avg_loss"] = round(float(np.mean(losss)) * 100, 4)
        if wins and losss:
            m["pf_ratio"] = round(abs(np.mean(wins) / np.mean(losss)), 3)
    return {"metrics": m, "equity_curve": equity_curve, "trade_log": trade_log}


def load_coin_ml(coin: str) -> pd.DataFrame:
    """OHLCV + 지표 + ML 피처."""
    return add_ml_features(add_indicators(load_coin_raw(coin)))


def run_random_validation_ml(all_df, coin_label, ensemble, cfg, seed, windows, window_days,
                             hist_start=None):
    all_df = all_df.dropna(subset=["_rsi", "_atr"]).copy()
    rng = random.Random(seed)
    min_start = hist_start or ("2021-04-01" if "ETH" in coin_label else "2020-06-01")
    possible = all_df[(all_df.index >= min_start) &
                      (all_df.index <= all_df.index[-1] - pd.Timedelta(days=window_days))].index
    chosen = sorted(rng.choices(possible, k=windows))

    print(f"\n랜덤 {windows}회 검증 ({window_days}일 윈도우, seed={seed}) [ML 필터]")
    passes = []; returns = []
    for i, sd in enumerate(chosen):
        ed = sd + pd.Timedelta(days=window_days)
        seg = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
        if len(seg) < 500:
            continue
        res = run_antifragile_ml(seg, ensemble, **cfg)
        m = res["metrics"]; tl = res["trade_log"]
        tpd = m.get("tpd", 0); r5 = remove_top_n(tl, 5); ret = m["total_return"]
        ok = sum([ret > 0, tpd >= 1.5, r5 > 0])
        mark = "✅" if ok == 3 else ("⚠️" if ok >= 2 else "❌")
        print(f"  [{i+1:02d}] {str(sd.date())+'~'+str(ed.date()):<26}  {m['n_trades']:>5}  "
              f"{m['win_rate']:>5.1f}%  {tpd:>4.2f}  {ret:>+8.1f}%  {m['mdd']:>4.1f}%  "
              f"{m.get('profit_factor',0):>5.3f}  {r5:>+7.1f}%  {mark} ({ok}/3)")
        passes.append(ok); returns.append(ret)

    if returns:
        print(f"\n  통과(3/3): {sum(p==3 for p in passes)}/{len(passes)}")
        print(f"  수익 양수: {sum(r>0 for r in returns)}/{len(returns)}")
        print(f"  평균 수익: {np.mean(returns):+.1f}%")
    return passes, returns


def main():
    parser = argparse.ArgumentParser(description="Antifragile ML-filtered Backtest")
    parser.add_argument("--coin", default="btc", choices=["btc", "eth", "sol", "xrp", "both", "all"])
    parser.add_argument("--mode", default="2026", choices=["2026", "random", "both", "june2026"])
    parser.add_argument("--model", required=True, help="AFEnsemble saved directory")
    parser.add_argument("--preset", default=None, choices=list(PRESETS.keys()))
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-days", type=int, default=91)
    parser.add_argument("--require-bb", action="store_true")
    parser.add_argument("--trail-init", type=float, default=DEFAULT_PARAMS["trail_atr_init"])
    parser.add_argument("--trail-tight", type=float, default=None)
    parser.add_argument("--add-step", type=float, default=0.5)
    parser.add_argument("--leverage", type=int, default=5)
    args = parser.parse_args()

    ensemble = AFEnsemble.load(args.model)
    print(f"ML 앙상블 로드: {args.model}  (theta={ensemble.threshold:.3f})")

    coin_map = {"both": ["btc", "eth"], "all": ["btc", "eth", "sol", "xrp"]}
    coins = coin_map.get(args.coin, [args.coin])

    cfg = dict(require_bb=args.require_bb, trail_atr_init=args.trail_init, atr_add_step=args.add_step, leverage=args.leverage)
    if args.trail_tight is not None:
        cfg["trail_atr_tight"] = args.trail_tight
    if args.preset:
        cfg.update(get_preset(args.preset))

    for coin in coins:
        label = coin.upper()
        print(f"\n{'█'*66}")
        print(f"  Antifragile + ML 필터 — {label}/USDT 5m")
        print(f"{'█'*66}")

        try:
            all_df = load_coin_ml(coin)

            if args.mode in ("2026", "both"):
                df26 = all_df[(all_df.index >= "2026-01-01") & (all_df.index < "2026-06-01")].copy()
                days = (df26.index[-1] - df26.index[0]).days
                print(f"\n{label} 2026: {df26.index[0].date()} ~ {df26.index[-1].date()}  ({days}일)")
                result = run_antifragile_ml(df26, ensemble, **cfg)
                print_result(f"{label} 2026 [ML]", result, days)

            if args.mode == "june2026":
                df_june = all_df[(all_df.index >= "2026-06-01") & (all_df.index < "2026-07-01")]
                if len(df_june) < 100:
                    print(f"\n  ⚠️  {label} 2026-06 데이터 부족 ({len(df_june)}행)")
                else:
                    days = (df_june.index[-1] - df_june.index[0]).days
                    result = run_antifragile_ml(df_june, ensemble, **cfg)
                    print_result(f"{label} 2026-06 [ML]", result, days)

            if args.mode in ("random", "both"):
                hist_start = COIN_CONFIG.get(coin, {}).get("hist_start", "2020-01-01")
                run_random_validation_ml(all_df, label, ensemble, cfg,
                                         args.seed, args.windows, args.window_days,
                                         hist_start=hist_start)
        except FileNotFoundError as e:
            print(f"\n  ⚠️  {e}\n")
            continue


if __name__ == "__main__":
    main()
