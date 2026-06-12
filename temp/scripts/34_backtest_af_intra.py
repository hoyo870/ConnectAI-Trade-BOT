"""
Antifragile 전략 개선 실험: reverse_RSI exit + intra-candle trail stop
기존 결과와 4가지 변형 비교.

  Baseline  : 기존 전략 (완성봉 trail_sl 체크, RSI 진입/청산 없음)
  Variant A : reverse_RSI — 숏 중 RSI ≤ rsi_lo (완성봉) 시 즉시 청산
  Variant B : intra_trail — 봉 고가/저가로 trail_sl 도달 체크 (더 현실적)
  Variant C : A + B (intra-candle RSI at LOW + intra trail)

10초 폴링 시뮬:
  - 봉 내 최저가(LOW)로 RSI를 재계산 → SHORT 조기 청산 트리거 여부 확인
  - 봉 내 고가/저가로 trail_sl 히트 여부 확인 (5분봉 말 체크보다 현실적)

Usage:
  python temp/scripts/34_backtest_af_intra.py
  python temp/scripts/34_backtest_af_intra.py --coin eth
  python temp/scripts/34_backtest_af_intra.py --coin all --mode both
"""
import sys, argparse, random
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path
from hybrid_engine import compute_metrics

# backtest_antifragile 의 데이터 로드 / 리포트 유틸 재사용
sys.path.insert(0, "scripts")
from backtest_antifragile import (
    load_coin_full, remove_top_n, COIN_CONFIG,
    load_ohlcv_csv,
)

ROOT        = Path(__file__).parent.parent.parent
TRADING_FEE = 0.0005
SLIPPAGE    = 0.0002
FEE_TOTAL   = TRADING_FEE + SLIPPAGE


# ─────────────────────────────────────────────────────────────────────────────
# 지표 확장: intra-candle RSI 추가
# ─────────────────────────────────────────────────────────────────────────────

def add_intra_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    봉 내 LOW/HIGH 가격을 마지막 close 대신 적용했을 때의 RSI 근사치 추가.
    EWM 상태를 현재 봉 적용 전으로 되돌려 LOW/HIGH 델타를 반영.

    _rsi_at_low  : 해당 봉이 LOW로 닫혔다면의 RSI  (SHORT 역신호 체크용)
    _rsi_at_high : 해당 봉이 HIGH로 닫혔다면의 RSI (LONG 역신호 체크용)
    """
    df = df.copy()
    close = df["close"]
    low   = df["low"]
    high  = df["high"]

    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))
    ag    = gain.ewm(com=13, adjust=False).mean()
    al    = loss.ewm(com=13, adjust=False).mean()

    # EWM alpha = 1/14 (com=13)
    # ag[i] = (1-a)*ag_prev + a*gain[i]  → ag_prev = (ag[i] - a*gain[i]) / (1-a)
    alpha   = 1.0 / 14.0
    ag_prev = (ag   - gain * alpha) / (1 - alpha)
    al_prev = (al   - loss * alpha) / (1 - alpha)

    prev_close = close.shift(1)

    # RSI at LOW
    g_lo = (low  - prev_close).clip(lower=0)
    l_lo = (-(low  - prev_close)).clip(lower=0)
    ag_lo = (1 - alpha) * ag_prev + alpha * g_lo
    al_lo = (1 - alpha) * al_prev + alpha * l_lo
    df["_rsi_at_low"] = 100 - 100 / (1 + ag_lo / (al_lo + 1e-9))

    # RSI at HIGH
    g_hi = (high - prev_close).clip(lower=0)
    l_hi = (-(high - prev_close)).clip(lower=0)
    ag_hi = (1 - alpha) * ag_prev + alpha * g_hi
    al_hi = (1 - alpha) * al_prev + alpha * l_hi
    df["_rsi_at_high"] = 100 - 100 / (1 + ag_hi / (al_hi + 1e-9))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 확장 백테스트 엔진
# ─────────────────────────────────────────────────────────────────────────────

def run_af_extended(
    df,
    initial_capital = 10_000.0,
    dt_rsi_lo = 22, dt_rsi_hi = 65,
    rg_rsi_lo = 30, rg_rsi_hi = 70,
    ut_rsi_lo = 35, ut_rsi_hi = 78,
    leverage        = 3,
    rr_base         = 0.10,
    rr_add          = 0.15,
    add_levels      = 3,
    atr_add_step    = 0.5,
    trail_atr_init  = 0.5,
    trail_atr_tight = 0.8,
    max_hold_bars   = 288,
    cooling_bars    = 100,
    max_dd_cb       = 0.30,
    # ── 실험 플래그 ─────────────────────────────────────────────
    reverse_rsi  = False,   # 숏(롱) 포지션 중 반대 RSI 신호 시 청산
    intra_trail  = False,   # 봉 고가/저가로 trail_sl 히트 체크
    intra_rsi    = False,   # (intra_trail 과 함께) LOW/HIGH RSI로 역신호 체크
):
    df = df.reset_index(drop=True)
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    df = df.reset_index(drop=True)

    capital     = initial_capital
    peak_cap    = initial_capital
    pos         = 0
    entry_price = 0.0
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
        low_p = float(row.get("low",  price))
        high_p= float(row.get("high", price))
        rsi   = float(row["_rsi"])
        atr   = float(row["_atr"])
        tup   = int(row.get("_trend_up",   0))
        tdn   = int(row.get("_trend_down", 0))

        # intra-candle RSI (LOW/HIGH 기반)
        rsi_at_low  = float(row.get("_rsi_at_low",  rsi))
        rsi_at_high = float(row.get("_rsi_at_high", rsi))

        rsi_lo = dt_rsi_lo if tdn else (ut_rsi_lo if tup else rg_rsi_lo)
        rsi_hi = dt_rsi_hi if tdn else (ut_rsi_hi if tup else rg_rsi_hi)

        # ── equity & circuit breaker ─────────────────────────────
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
                                   "lev": leverage, "rr": current_rr,
                                   "forced": True, "direction": pos, "reason": "CB"})
                pos = 0

        if cooling_left > 0:
            cooling_left -= 1
            if pos != 0:
                cp  = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_bar,
                                   "lev": leverage, "rr": current_rr,
                                   "forced": True, "direction": pos, "reason": "cooling"})
                pos = 0
            equity_curve.append(capital); continue

        # ── 포지션 관리 ──────────────────────────────────────────
        if pos != 0:
            hold = idx - entry_bar

            # trail_sl 업데이트
            if pos == 1:
                peak_price = max(peak_price, price)
                trail_mult = trail_atr_tight if add_count > 0 else trail_atr_init
                trail_sl   = max(trail_sl, peak_price - trail_mult * atr)
            else:
                peak_price = min(peak_price, price)
                trail_mult = trail_atr_tight if add_count > 0 else trail_atr_init
                trail_sl   = min(trail_sl, peak_price + trail_mult * atr)

            # ── 청산 조건 결정 ───────────────────────────────────
            hit_stop  = False
            exit_p    = price
            exit_reason = "close"

            # 1) intra-candle trail stop (봉 고가/저가 기준)
            if intra_trail:
                if pos == 1 and low_p <= trail_sl:
                    hit_stop = True
                    exit_p   = max(trail_sl, low_p)   # trail_sl 가격에서 청산
                    exit_reason = "trail_SL_intra"
                elif pos == -1 and high_p >= trail_sl:
                    hit_stop = True
                    exit_p   = min(trail_sl, high_p)
                    exit_reason = "trail_SL_intra"
            else:
                # 기존: 완성봉 close 기준
                if (pos == 1 and price <= trail_sl) or (pos == -1 and price >= trail_sl):
                    hit_stop = True
                    exit_reason = "trail_SL"

            # 2) reverse_RSI 청산 (trail_sl 히트보다 우선 체크)
            if not hit_stop and reverse_rsi:
                # SHORT 포지션 중 RSI 과매도 → 롱 신호 발생 → 숏 청산
                rsi_check_lo = rsi_at_low if intra_rsi else rsi
                # LONG 포지션 중 RSI 과매수 → 숏 신호 발생 → 롱 청산
                rsi_check_hi = rsi_at_high if intra_rsi else rsi

                if pos == -1 and rsi_check_lo <= rsi_lo:
                    hit_stop = True
                    exit_p   = low_p if intra_rsi else price
                    exit_reason = "reverse_RSI"
                elif pos == 1 and rsi_check_hi >= rsi_hi:
                    hit_stop = True
                    exit_p   = high_p if intra_rsi else price
                    exit_reason = "reverse_RSI"

            # 3) 최대 보유 기간
            if not hit_stop and hold >= max_hold_bars:
                hit_stop = True
                exit_reason = "timeout"

            if hit_stop:
                cp  = exit_p - FEE_TOTAL * exit_p * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": hold,
                                   "lev": leverage, "rr": current_rr,
                                   "forced": False, "direction": pos,
                                   "reason": exit_reason})
                pos = 0; add_count = 0; current_rr = 0.0
            elif not hit_stop:
                # 피라미딩 체크
                favorable = pos * (price - entry_price) / (atr + 1e-9)
                next_level = (add_count + 1) * atr_add_step
                if add_count < add_levels and favorable >= next_level:
                    current_rr += rr_add; add_count += 1
                    if pos == 1: trail_sl = max(trail_sl, price - trail_atr_tight * atr)
                    else:         trail_sl = min(trail_sl, price + trail_atr_tight * atr)

        # ── 신규 진입 ────────────────────────────────────────────
        if pos == 0:
            long_ok  = rsi <= rsi_lo
            short_ok = rsi >= rsi_hi and not long_ok

            if long_ok:
                ep = price * (1 + FEE_TOTAL)
                entry_price = ep; current_rr = rr_base; add_count = 0
                trail_sl = ep - trail_atr_init * atr; peak_price = ep
                pos = 1; entry_bar = idx
            elif short_ok:
                ep = price * (1 - FEE_TOTAL)
                entry_price = ep; current_rr = rr_base; add_count = 0
                trail_sl = ep + trail_atr_init * atr; peak_price = ep
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
        if wins:  m["avg_win"]  = round(float(np.mean(wins)) * 100, 4)
        if losss: m["avg_loss"] = round(float(np.mean(losss)) * 100, 4)
        if wins and losss:
            m["pf_ratio"] = round(abs(np.mean(wins) / np.mean(losss)), 3)
        # 청산 이유 분포
        reasons = {}
        for t in trade_log:
            r = t.get("reason", "?")
            reasons[r] = reasons.get(r, 0) + 1
        m["reasons"] = reasons
    return {"metrics": m, "equity_curve": equity_curve, "trade_log": trade_log}


# ─────────────────────────────────────────────────────────────────────────────
# 리포트
# ─────────────────────────────────────────────────────────────────────────────

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

    print(f"\n{'─'*66}")
    print(f"  {label}")
    print(f"{'─'*66}")
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
    if "reasons" in m:
        reason_str = "  ".join(f"{k}={v}" for k, v in sorted(m["reasons"].items()))
        print(f"  청산이유:  {reason_str}")
    print(f"  판정:      {p}/3  {'✅ 통과' if p==3 else ('⚠️ 부분' if p>=2 else '❌ 탈락')}")
    return p


def print_summary_table(results: dict):
    """4가지 variant 핵심 지표 한눈에 비교"""
    print(f"\n{'='*80}")
    print(f"  비교 요약")
    print(f"  {'변형':<30} {'거래수':>6} {'WR':>6} {'TPD':>5} {'수익':>8} {'MDD':>5} {'PF':>6} {'Top5':>7}")
    print(f"  {'─'*76}")
    for name, res in results.items():
        m  = res["metrics"]
        tl = res["trade_log"]
        tpd = m.get("tpd", 0)
        r5  = remove_top_n(tl, 5)
        ok  = sum([m["total_return"] > 0, tpd >= 1.5, r5 > 0])
        mark = "✅" if ok==3 else ("⚠️" if ok>=2 else "❌")
        print(f"  {name:<30} {m['n_trades']:>6} {m['win_rate']:>5.1f}% "
              f"{tpd:>4.2f} {m['total_return']:>+7.1f}% {m['mdd']:>4.1f}% "
              f"{m.get('profit_factor',0):>5.3f} {r5:>+6.1f}%  {mark}")
    print(f"{'='*80}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = {
    "Baseline (기존)":             dict(reverse_rsi=False, intra_trail=False, intra_rsi=False),
    "A: reverse_RSI (완성봉)":     dict(reverse_rsi=True,  intra_trail=False, intra_rsi=False),
    "B: intra trail (고가/저가)":  dict(reverse_rsi=False, intra_trail=True,  intra_rsi=False),
    "C: A+B (intra RSI+trail)":    dict(reverse_rsi=True,  intra_trail=True,  intra_rsi=True),
}


def run_all_variants(df, days):
    results = {}
    for name, flags in VARIANTS.items():
        res = run_af_extended(df, **flags)
        results[name] = res
        print_result(name, res, days)
    print_summary_table(results)
    return results


def run_random_validation(all_df, coin_label, seed, windows, window_days, hist_start):
    all_df.dropna(subset=["_rsi", "_atr"], inplace=True)
    rng = random.Random(seed)
    possible = all_df[(all_df.index >= hist_start) &
                      (all_df.index <= all_df.index[-1] - pd.Timedelta(days=window_days))].index
    chosen = sorted(rng.choices(possible, k=windows))

    print(f"\n랜덤 {windows}회 검증 ({window_days}일 윈도우, seed={seed}) — {coin_label}")

    # header
    header = f"  {'#':>3}  {'구간':<26}"
    for name in VARIANTS:
        short = name[:14]
        header += f"  {short:>14}"
    print(header)
    print(f"  {'─'*100}")

    all_passes = {n: [] for n in VARIANTS}
    all_returns = {n: [] for n in VARIANTS}

    for i, sd in enumerate(chosen):
        ed  = sd + pd.Timedelta(days=window_days)
        seg = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
        if len(seg) < 500:
            continue

        row_str = f"  [{i+1:02d}] {str(sd.date())+'~'+str(ed.date()):<26}"
        for name, flags in VARIANTS.items():
            res = run_af_extended(seg, **flags)
            m   = res["metrics"]
            tl  = res["trade_log"]
            tpd = m.get("tpd", 0)
            r5  = remove_top_n(tl, 5)
            ok  = sum([m["total_return"] > 0, tpd >= 1.5, r5 > 0])
            mark = "✅" if ok==3 else ("⚠️" if ok>=2 else "❌")
            row_str += f"  {m['total_return']:>+6.1f}% {mark}({ok}/3)"
            all_passes[name].append(ok)
            all_returns[name].append(m["total_return"])

        print(row_str)

    print(f"\n  {'─'*80}")
    print(f"  {'변형':<30} {'통과(3/3)':>10} {'수익양수':>10} {'평균수익':>10}")
    for name in VARIANTS:
        p3 = sum(p==3 for p in all_passes[name])
        pp = sum(r>0  for r in all_returns[name])
        av = np.mean(all_returns[name]) if all_returns[name] else 0
        print(f"  {name:<30} {p3:>4}/{len(all_passes[name]):<6} {pp:>4}/{len(all_returns[name]):<6} {av:>+8.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Antifragile intra-candle 개선 실험")
    parser.add_argument("--coin",       default="btc",
                        choices=["btc", "eth", "sol", "xrp", "both", "all"])
    parser.add_argument("--mode",       default="2026", choices=["2026", "random", "both"])
    parser.add_argument("--windows",    type=int, default=10)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--window-days",type=int, default=91)
    args = parser.parse_args()

    coin_map = {"both": ["btc","eth"], "all": ["btc","eth","sol","xrp"]}
    coins = coin_map.get(args.coin, [args.coin])

    for coin in coins:
        cfg  = COIN_CONFIG[coin]
        print(f"\n{'█'*66}")
        print(f"  {cfg['label']}/USDT 5m — reverse_RSI + intra-trail 실험")
        print(f"{'█'*66}")

        all_df = load_coin_full(coin)
        all_df = add_intra_rsi(all_df)   # _rsi_at_low / _rsi_at_high 추가

        if args.mode in ("2026", "both"):
            seg26 = all_df[all_df.index >= "2026-01-01"].copy()
            days  = len(seg26) / 288
            print(f"\n▶ 2026 OOS 구간 ({len(seg26):,}봉 / {days:.0f}일)")
            run_all_variants(seg26, days)

        if args.mode in ("random", "both"):
            run_random_validation(
                all_df, cfg["label"],
                seed=args.seed, windows=args.windows,
                window_days=args.window_days,
                hist_start=cfg["hist_start"],
            )


if __name__ == "__main__":
    main()
