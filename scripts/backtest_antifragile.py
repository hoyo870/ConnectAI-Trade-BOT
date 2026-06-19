"""
Antifragile Trailing Stop 전략 백테스트
temp/scripts/27, 32 검증 완료 → scripts/ 승격 (2026-06-03)

핵심:
- 진입: AdaptRSI (1h EMA 방향별 RSI 임계값 동적 조정)
- 청산: ATR trailing stop (고정 SL/TP 없음)
- 사이징: 소규모 시작(rr=0.10) → 유리방향 시 피라미딩

검증 결과:
- 2026 BTC: +132.4%, PF=8.364, MDD=2.7%, Top5=+70.3%
- hist 9/10 통과, 평균 +123.9%/3개월, 10/10 수익 양수

Usage:
  python scripts/backtest_antifragile.py --mode 2026
  python scripts/backtest_antifragile.py --mode random --windows 10 --seed 42
  python scripts/backtest_antifragile.py --mode random --require-bb
"""
import sys, argparse, random
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, "src")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from hybrid_engine import compute_metrics
from config.af_params import TRADING_FEE, SLIPPAGE, FEE_TOTAL, PRESETS, get_preset, DEFAULT_PARAMS
from config.loader import load_coin_raw, load_ohlcv_csv, _normalize_index, COIN_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# 지표 계산
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]; high = df["high"]; low = df["low"]

    delta = close.diff()
    ag = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["_rsi"] = 100 - 100 / (1 + ag / (al + 1e-9))

    mid = close.rolling(20).mean(); std = close.rolling(20).std()
    df["_bb_upper"] = mid + 2 * std
    df["_bb_lower"] = mid - 2 * std

    tr = pd.concat([high - low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    df["_atr"] = tr.ewm(span=14, adjust=False).mean()

    cl1h   = close.resample("1h").last().ffill()
    ema_1h = cl1h.ewm(span=20, adjust=False).mean()
    df["_trend_up"]   = (cl1h > ema_1h).reindex(df.index, method="ffill").fillna(False).astype(int)
    df["_trend_down"] = (cl1h < ema_1h).reindex(df.index, method="ffill").fillna(False).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Antifragile Trailing Stop 백테스트 엔진
# ─────────────────────────────────────────────────────────────────────────────

def run_antifragile(
    df,
    initial_capital = 10_000.0,
    require_bb      = False,
    leverage        = 7,
    max_hold_bars   = 288,
    cooling_bars    = 100,
    max_dd_cb       = 0.99,
    **kwargs,
):
    # DEFAULT_PARAMS 기준으로 caller kwargs 오버라이드 (파라미터 스테일 방지)
    p               = {**DEFAULT_PARAMS, **kwargs}
    dt_rsi_lo       = p["dt_rsi_lo"]
    dt_rsi_hi       = p["dt_rsi_hi"]
    rg_rsi_lo       = p["rg_rsi_lo"]
    rg_rsi_hi       = p["rg_rsi_hi"]
    ut_rsi_lo       = p["ut_rsi_lo"]
    ut_rsi_hi       = p["ut_rsi_hi"]
    rr_base         = p["rr_base"]
    rr_add          = p["rr_add"]
    add_levels      = p["add_levels"]
    atr_add_step    = p["atr_add_step"]
    trail_atr_init  = p["trail_atr_init"]
    trail_atr_tight = p["trail_atr_tight"]
    df = df.reset_index(drop=True)
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    df = df.reset_index(drop=True)

    capital    = initial_capital
    peak_cap   = initial_capital
    pos        = 0
    entry_price = 0.0
    entry_atr   = 0.0
    current_rr  = 0.0
    add_count   = 0
    trail_sl    = 0.0
    peak_price  = 0.0
    entry_bar   = 0
    cooling_left = 0
    cb_triggers  = 0

    equity_curve = [capital]
    trade_log    = []

    for idx in range(1, len(df)):
        row   = df.iloc[idx]
        price = float(row["close"])
        rsi   = float(row["_rsi"])
        atr   = float(row["_atr"])
        bb_u  = float(row.get("_bb_upper", price * 1.02))
        bb_l  = float(row.get("_bb_lower", price * 0.98))
        tup   = int(row.get("_trend_up", 0))
        tdn   = int(row.get("_trend_down", 0))

        rsi_lo = dt_rsi_lo if tdn else (ut_rsi_lo if tup else rg_rsi_lo)
        rsi_hi = dt_rsi_hi if tdn else (ut_rsi_hi if tup else rg_rsi_hi)

        # equity & CB
        if pos != 0:
            unr    = pos * (price - entry_price) / (entry_price + 1e-9)
            equity = capital * (1 + unr * leverage * current_rr)
        else:
            equity = capital
        peak_cap = max(peak_cap, equity)
        dd = (peak_cap - equity) / (peak_cap + 1e-9)

        if dd > max_dd_cb and cooling_left == 0:
            cooling_left = cooling_bars; cb_triggers += 1
            if pos != 0:
                cp  = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_bar,
                                   "entry_bar": entry_bar, "exit_bar": idx,
                                   "lev": leverage, "rr": current_rr, "forced": True, "direction": pos})
                pos = 0

        if cooling_left > 0:
            cooling_left -= 1
            if pos != 0:
                cp  = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_bar,
                                   "entry_bar": entry_bar, "exit_bar": idx,
                                   "lev": leverage, "rr": current_rr, "forced": True, "direction": pos})
                pos = 0
            equity_curve.append(capital); continue

        # 포지션 관리 (trailing stop + 피라미딩)
        if pos != 0:
            hold = idx - entry_bar
            effective_atr = max(atr, entry_atr * 0.6)  # 진입 ATR 60% 미만으로 SL 좁아지지 않도록
            if pos == 1:
                peak_price = max(peak_price, price)
                trail_mult = trail_atr_tight if add_count > 0 else trail_atr_init
                trail_sl   = max(trail_sl, peak_price - trail_mult * effective_atr)
                hit_stop   = price <= trail_sl
            else:
                peak_price = min(peak_price, price)
                trail_mult = trail_atr_tight if add_count > 0 else trail_atr_init
                trail_sl   = min(trail_sl, peak_price + trail_mult * effective_atr)
                hit_stop   = price >= trail_sl

            if hit_stop or hold >= max_hold_bars:
                cp  = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": hold,
                                   "entry_bar": entry_bar, "exit_bar": idx,
                                   "lev": leverage, "rr": current_rr, "forced": False, "direction": pos})
                pos = 0; add_count = 0; current_rr = 0.0
            else:
                favorable_move = pos * (price - entry_price) / (atr + 1e-9)
                next_add_level = (add_count + 1) * atr_add_step
                if add_count < add_levels and favorable_move >= next_add_level:
                    current_rr += rr_add; add_count += 1
                    if pos == 1: trail_sl = max(trail_sl, price - trail_atr_tight * effective_atr)
                    else:         trail_sl = min(trail_sl, price + trail_atr_tight * effective_atr)

        # 신규 진입
        if pos == 0:
            long_ok  = (rsi <= rsi_lo) and ((not require_bb) or (price <= bb_l))
            short_ok = (rsi >= rsi_hi) and ((not require_bb) or (price >= bb_u))

            if (long_ok or short_ok) and atr < price * 0.0015:
                long_ok = short_ok = False  # ATR 너무 낮음 — 횡보 구간 진입 차단

            if long_ok:
                ep          = price * (1 + FEE_TOTAL)
                entry_price = ep; entry_atr = atr; current_rr = rr_base; add_count = 0
                trail_sl    = ep - trail_atr_init * atr; peak_price = ep
                pos = 1; entry_bar = idx
            elif short_ok:
                ep          = price * (1 - FEE_TOTAL)
                entry_price = ep; entry_atr = atr; current_rr = rr_base; add_count = 0
                trail_sl    = ep + trail_atr_init * atr; peak_price = ep
                pos = -1; entry_bar = idx

        equity_curve.append(capital)

    m = compute_metrics(equity_curve, trade_log)
    m["cb_triggers"] = cb_triggers
    if trade_log:
        days_total = len(df) / 288
        m["tpd"]       = round(len(trade_log) / days_total, 2)
        m["long_cnt"]  = sum(1 for t in trade_log if t.get("direction") ==  1)
        m["short_cnt"] = sum(1 for t in trade_log if t.get("direction") == -1)
        wins  = [t["pnl"] for t in trade_log if t["pnl"] > 0]
        losss = [t["pnl"] for t in trade_log if t["pnl"] < 0]
        if wins:  m["avg_win"]  = round(float(np.mean(wins))  * 100, 4)
        if losss: m["avg_loss"] = round(float(np.mean(losss)) * 100, 4)
        if wins and losss:
            m["pf_ratio"] = round(abs(np.mean(wins) / np.mean(losss)), 3)
    return {"metrics": m, "equity_curve": equity_curve, "trade_log": trade_log}


# ─────────────────────────────────────────────────────────────────────────────
# 리포트 유틸
# ─────────────────────────────────────────────────────────────────────────────

def remove_top_n(trades, n, base=10_000.0):
    top_idx = set(sorted(range(len(trades)), key=lambda i: trades[i]["pnl"], reverse=True)[:n])
    cap = base
    for i, t in enumerate(trades):
        if i in top_idx: continue
        cap *= 1 + t["pnl"]
    return (cap / base - 1) * 100


def print_result(label, result, days):
    m  = result["metrics"]
    tl = result["trade_log"]
    tpd = m.get("tpd", round(len(tl) / max(days, 1), 2))
    r5  = remove_top_n(tl, 5)
    r3  = remove_top_n(tl, 3)
    long_t  = [t for t in tl if t.get("direction") ==  1]
    short_t = [t for t in tl if t.get("direction") == -1]
    def wr_f(ts): return sum(1 for t in ts if t["pnl"] > 0) / len(ts) * 100 if ts else 0.0

    ok_ret = m["total_return"] > 0
    ok_tpd = tpd >= 1.5
    ok_top = r5 > 0
    p = sum([ok_ret, ok_tpd, ok_top])

    print(f"\n{'='*66}")
    print(f"  {label}")
    print(f"{'='*66}")
    print(f"  거래수:    {m['n_trades']}  (롱 {m.get('long_cnt',0)} / 숏 {m.get('short_cnt',0)})")
    print(f"  WR:        {m['win_rate']:.1f}%  (롱 {wr_f(long_t):.1f}% / 숏 {wr_f(short_t):.1f}%)")
    print(f"  TPD:       {tpd:.2f}  {'✅' if ok_tpd else '❌'}")
    print(f"  수익률:    {m['total_return']:+.2f}%  {'✅' if ok_ret else '❌'}")
    print(f"  MDD:       {m['mdd']:.1f}%")
    print(f"  PF:        {m.get('profit_factor', 0):.3f}")
    print(f"  avg_win:   {m.get('avg_win', 0):+.4f}%")
    print(f"  avg_loss:  {m.get('avg_loss', 0):+.4f}%")
    print(f"  win/loss:  {m.get('pf_ratio', 0):.3f}x")
    print(f"  Top-5 제거:{r5:+.2f}%  {'✅' if ok_top else '❌'}")
    print(f"  Top-3 제거:{r3:+.2f}%")
    print(f"  판정:      {p}/3  {'✅ 통과' if p==3 else ('⚠️ 부분' if p>=2 else '❌ 탈락')}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_coin_full(coin: str) -> pd.DataFrame:
    """코인별 전체 OHLCV + 구형 EMA 지표 로드 (data/loader.py 위임, 하위 호환)."""
    return add_indicators(load_coin_raw(coin))


def load_btc_full(): return load_coin_full("btc")
def load_eth_full(): return load_coin_full("eth")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def run_random_validation(all_df, coin_label, cfg, seed, windows, window_days,
                          hist_start: str = None):
    all_df.dropna(subset=["_rsi", "_atr"], inplace=True)
    rng = random.Random(seed)
    min_start = hist_start or ("2021-04-01" if "ETH" in coin_label else "2020-06-01")
    possible = all_df[(all_df.index >= min_start) &
                      (all_df.index <= all_df.index[-1] - pd.Timedelta(days=window_days))].index
    chosen = sorted(rng.choices(possible, k=windows))

    print(f"\n랜덤 {windows}회 검증 ({window_days}일 윈도우, seed={seed})")
    print(f"\n  {'#':>3}  {'구간':<26}  {'n':>5}  {'WR':>6}  {'TPD':>5}  "
          f"{'수익':>9}  {'MDD':>5}  {'PF':>6}  {'Top5':>8}  {'판정'}")
    print(f"  {'─'*90}")

    passes = []; returns = []
    for i, sd in enumerate(chosen):
        ed  = sd + pd.Timedelta(days=window_days)
        seg = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
        if len(seg) < 500: continue

        res = run_antifragile(seg, **cfg)
        m   = res["metrics"]; tl = res["trade_log"]
        tpd = m.get("tpd", 0); r5 = remove_top_n(tl, 5); ret = m["total_return"]
        ok  = sum([ret > 0, tpd >= 1.5, r5 > 0])
        mark = "✅" if ok==3 else ("⚠️" if ok>=2 else "❌")
        print(f"  [{i+1:02d}] {str(sd.date())+'~'+str(ed.date()):<26}  {m['n_trades']:>5}  "
              f"{m['win_rate']:>5.1f}%  {tpd:>4.2f}  {ret:>+8.1f}%  {m['mdd']:>4.1f}%  "
              f"{m.get('profit_factor',0):>5.3f}  {r5:>+7.1f}%  {mark} ({ok}/3)")
        passes.append(ok); returns.append(ret)

    print(f"\n  {'─'*70}")
    print(f"  통과(3/3): {sum(p==3 for p in passes)}/{len(passes)}")
    print(f"  수익 양수: {sum(r>0 for r in returns)}/{len(returns)}")
    print(f"  평균 수익: {np.mean(returns):+.1f}%")

    n3 = sum(p==3 for p in passes)
    if n3 >= 7:   print(f"\n  ✅ 강력 통과 ({n3}/10)")
    elif n3 >= 5: print(f"\n  ✅ 통과 ({n3}/10)")
    else:         print(f"\n  ❌ 미통과 ({n3}/10)")
    return passes, returns


# 프리셋은 config/af_params.py에서 중앙 관리 (PRESETS, get_preset import됨)


def main():
    parser = argparse.ArgumentParser(description="Antifragile Trailing Stop Backtest")
    parser.add_argument("--coin",       default="btc",
                        choices=["btc", "eth", "sol", "xrp", "both", "all"])
    parser.add_argument("--mode",       default="2026", choices=["2026", "random", "both", "june2026"])
    parser.add_argument("--preset",     default=None,   choices=list(PRESETS.keys()),
                        help="파라미터 프리셋 (prod/stable/aggressive/conservative)")
    parser.add_argument("--windows",    type=int,   default=10)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--window-days",type=int,   default=91)
    parser.add_argument("--require-bb", action="store_true")
    parser.add_argument("--trail-init", type=float, default=1.8)
    parser.add_argument("--trail-tight",type=float, default=None)  # None → DEFAULT_PARAMS 사용
    parser.add_argument("--add-step",   type=float, default=0.5)
    args = parser.parse_args()

    coin_map = {
        "both": ["btc", "eth"],
        "all":  ["btc", "eth", "sol", "xrp"],
    }
    coins = coin_map.get(args.coin, [args.coin])

    cfg = dict(
        require_bb     = args.require_bb,
        trail_atr_init = args.trail_init,
        atr_add_step   = args.add_step,
    )
    if args.trail_tight is not None:
        cfg["trail_atr_tight"] = args.trail_tight
    if args.preset:
        cfg.update(get_preset(args.preset))

    preset_label = f"preset={args.preset}" if args.preset else f"trail_init={cfg['trail_atr_init']}  trail_tight={cfg.get('trail_atr_tight', 'default')}"

    for coin in coins:
        label = coin.upper()
        print(f"\n{'█'*66}")
        print(f"  Antifragile Trailing Stop — {label}/USDT 5m")
        print(f"  {preset_label}  add_step={cfg.get('atr_add_step', args.add_step)}  require_bb={args.require_bb}")
        print(f"{'█'*66}")

        try:
            all_df_cache = None  # 2026 + random 모두 사용 시 재로드 방지

            if args.mode in ("2026", "both"):
                # 4코인 모두 load_coin_full로 통일 (최신 CSV 반영, 기간 일관성)
                # 2026 OOS 기간: 2026-01-01 ~ 2026-05-31 고정
                # (6월 이후는 현재 실거래 중인 기간 → OOS 오염 방지)
                if all_df_cache is None:
                    all_df_cache = load_coin_full(coin)
                df26 = all_df_cache[
                    (all_df_cache.index >= "2026-01-01") &
                    (all_df_cache.index <  "2026-06-01")
                ].copy()

                days = (df26.index[-1] - df26.index[0]).days
                print(f"\n{label} 2026: {df26.index[0].date()} ~ {df26.index[-1].date()}  ({days}일)")
                result = run_antifragile(df26, **cfg)
                print_result(f"{label} 2026", result, days)

            if args.mode == "june2026":
                if all_df_cache is None:
                    all_df_cache = load_coin_full(coin)
                dfall = all_df_cache
                df_june = dfall[(dfall.index >= "2026-06-01") & (dfall.index < "2026-07-01")]
                if len(df_june) < 100:
                    print(f"\n  ⚠️  {label} 2026-06 데이터 부족 ({len(df_june)}행)")
                else:
                    days = (df_june.index[-1] - df_june.index[0]).days
                    print(f"\n{label} 2026-06: {df_june.index[0].date()} ~ {df_june.index[-1].date()}  ({days}일)")
                    result = run_antifragile(df_june, **cfg)
                    print_result(f"{label} 2026-06", result, days)

            if args.mode in ("random", "both"):
                if all_df_cache is None:
                    all_df_cache = load_coin_full(coin)
                hist_start = COIN_CONFIG.get(coin, {}).get("hist_start", "2020-01-01")
                run_random_validation(all_df_cache, label, cfg,
                                      args.seed, args.windows, args.window_days,
                                      hist_start=hist_start)
        except FileNotFoundError as e:
            print(f"\n  ⚠️  {e}\n")
            continue


if __name__ == "__main__":
    main()
