"""
backtest_af_exact.py — live_trader.py process_tick_af 완벽 모방 백테스트

기존 backtest_antifragile.py와의 차이 (= 실거래와 실제로 같아지는 부분):
  1. BB σ=0.5 트렌드 판별  (EMA 단순 크로스 → BB 밴드 기반)
  2. 트렌드 안정화 1봉 지연 (첫 전환 봉 진입 차단)
  3. 절반 익절             (RSI 극단구간 포지션 50% 청산)
  4. 피라미딩 RSI 차단     (극단 RSI 구간에서 피라미딩 스킵)
  5. reverse_flip US-002   (반대 신호 2봉 연속 → 즉시 청산)
  6. 방향쿨링   US-004     (동일 방향 3연속 손실 → 20봉 차단)
  7. ATR 필터             (atr < price×0.15% 진입 차단) ← 기존 BT와 동일

수수료: config/af_params.py FEE_TOTAL = 0.111%/side (taker+slippage 실측)

Usage:
  python scripts/backtest_af_exact.py --mode 2026
  python scripts/backtest_af_exact.py --mode hist --windows 10 --seed 42
  python scripts/backtest_af_exact.py --mode jun1819
  python scripts/backtest_af_exact.py --mode 2026 --coin all
"""

import sys, argparse, random
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from config.af_params import FEE_TOTAL, DEFAULT_PARAMS

# ── 파라미터 ──────────────────────────────────────────────────────────────────
CONSECUTIVE_REVERSE_BARS = 2
CONSECUTIVE_LOSS_LIMIT   = 3
DIRECTION_COOLING_BARS   = 20
BB_SIGMA                 = 0.5   # 1h BB 횡보 구역 폭


# ── 지표 계산 (live_trader _compute_af_indicators 완벽 모방) ──────────────────
def add_indicators_exact(df: pd.DataFrame) -> pd.DataFrame:
    """
    live_trader의 _compute_af_indicators와 동일하게 각 봉의 지표를 계산.
    BB σ=0.5 기반 트렌드 (live_trader와 완전 일치).
    """
    df = df.copy()
    close = df["close"]; high = df["high"]; low = df["low"]

    # ATR 14 (EWM span=14)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["_atr"] = tr.ewm(span=14, adjust=False).mean()

    # RSI 14 (EWM com=13)
    delta = close.diff()
    ag = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["_rsi"] = 100 - 100 / (1 + ag / (al + 1e-9))

    # 1h BB σ=0.5 트렌드 (live_trader와 동일)
    cl1h   = close.resample("1h").last().ffill()
    ema_1h = cl1h.ewm(span=20, adjust=False).mean()
    std_1h = cl1h.rolling(20).std()
    bb_up  = ema_1h + BB_SIGMA * std_1h
    bb_lo  = ema_1h - BB_SIGMA * std_1h

    df["_trend_up"]   = (cl1h > bb_up).reindex(df.index, method="ffill").fillna(False).astype(int)
    df["_trend_down"] = (cl1h < bb_lo).reindex(df.index, method="ffill").fillna(False).astype(int)

    return df


# ── 핵심 백테스트 엔진 ────────────────────────────────────────────────────────
def run_af_exact(
    df: pd.DataFrame,
    initial_capital: float = 270.76,
    p: dict = None,
) -> dict:
    """
    live_trader.py process_tick_af를 완벽히 모방.
    p: DEFAULT_PARAMS 기반 dict
    """
    if p is None:
        p = dict(DEFAULT_PARAMS)

    df = df.reset_index(drop=True)
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    df = df.reset_index(drop=True)

    # 파라미터 단축
    dt_lo, dt_hi = p["dt_rsi_lo"], p["dt_rsi_hi"]
    rg_lo, rg_hi = p["rg_rsi_lo"], p["rg_rsi_hi"]
    ut_lo, ut_hi = p["ut_rsi_lo"], p["ut_rsi_hi"]
    lev          = p.get("leverage", 7)
    rr_base      = p["rr_base"]
    trail_init   = p["trail_atr_init"]
    trail_tight  = p["trail_atr_tight"]
    add_step     = p["atr_add_step"]
    add_lvl      = p["add_levels"]
    max_hold     = p.get("max_hold_bars", 288)

    # state
    capital      = initial_capital
    peak_cap     = initial_capital
    pos          = 0
    entry_price  = 0.0;  avg_entry = 0.0
    entry_atr    = 0.0;  rr        = rr_base
    add_cnt      = 0;    trail_sl  = 0.0;  peak_px = 0.0
    entry_bar    = 0;    partial   = False

    # US-002
    consec_rev   = 0
    # US-004
    cl_long = 0; cl_short = 0
    csl_long = 0; csl_short = 0

    # 트렌드 안정화
    prev_trend   = "RG"

    trade_log    = []
    equity_curve = [capital]

    def get_trend(row):
        tup = bool(row["_trend_up"]); tdn = bool(row["_trend_down"])
        return "DN" if tdn else ("UP" if tup else "RG")

    def get_thresholds(trend):
        if trend == "DN": return dt_lo, dt_hi
        if trend == "UP": return ut_lo, ut_hi
        return rg_lo, rg_hi

    for idx in range(1, len(df)):
        row   = df.iloc[idx]
        price = float(row["close"])
        atr   = float(row["_atr"])
        rsi   = float(row["_rsi"])

        trend    = get_trend(row)
        rsi_lo, rsi_hi = get_thresholds(trend)

        long_ok_now  = rsi <= rsi_lo
        short_ok_now = (rsi >= rsi_hi) and not long_ok_now

        # ── 청산 체크 ────────────────────────────────────────────────────────
        if pos != 0:
            # US-002 reverse_flip
            rev_fired = (pos == 1 and short_ok_now) or (pos == -1 and long_ok_now)
            consec_rev = consec_rev + 1 if rev_fired else 0
            force_flip = consec_rev >= CONSECUTIVE_REVERSE_BARS

            hold     = idx - entry_bar
            hit_stop = (pos == 1 and price <= trail_sl) or (pos == -1 and price >= trail_sl)
            timeout  = hold >= max_hold

            if hit_stop or timeout or force_flip:
                reason = "trail_SL" if hit_stop else ("timeout" if timeout else "reverse_flip")
                raw    = pos * (price - avg_entry) / (avg_entry + 1e-9)
                pnl    = max(raw * lev * rr, -rr) - FEE_TOTAL
                capital *= (1 + pnl)
                peak_cap = max(peak_cap, capital)
                trade_log.append({
                    "pnl": pnl, "direction": pos, "reason": reason,
                    "capital": round(capital, 2),
                    "entry": round(avg_entry, 6), "exit": round(price, 6),
                })
                closed_pos = pos
                # US-004 쿨링 카운터
                if pnl <= 0:
                    if closed_pos == 1: csl_long += 1;  csl_short = 0
                    else:               csl_short += 1; csl_long  = 0
                else:
                    if closed_pos == 1: csl_long  = 0
                    else:               csl_short = 0
                if csl_long  >= CONSECUTIVE_LOSS_LIMIT: cl_long  = DIRECTION_COOLING_BARS; csl_long  = 0
                if csl_short >= CONSECUTIVE_LOSS_LIMIT: cl_short = DIRECTION_COOLING_BARS; csl_short = 0

                pos = 0; entry_price = 0; avg_entry = 0; rr = rr_base
                add_cnt = 0; trail_sl = 0; peak_px = 0; partial = False; consec_rev = 0

            else:
                # trailing stop 업데이트
                eff_atr    = max(atr, entry_atr * 0.6)
                trail_mult = trail_tight if add_cnt > 0 else trail_init
                if pos == 1:
                    peak_px  = max(peak_px, price)
                    trail_sl = max(trail_sl, peak_px - trail_mult * eff_atr)
                else:
                    peak_px  = min(peak_px, price)
                    trail_sl = min(trail_sl, peak_px + trail_mult * eff_atr)

                # 절반 익절 (live_trader 동일)
                if not partial:
                    p_ok = (pos == 1 and rsi >= rsi_hi) or (pos == -1 and rsi <= rsi_lo)
                    if p_ok:
                        prr  = rr * 0.5
                        ppnl = max(pos * (price - avg_entry) / (avg_entry + 1e-9) * lev * prr, -prr)
                        capital *= (1 + ppnl)
                        rr     *= 0.5   # 남은 절반만 추적
                        partial = True

                # 피라미딩 (RSI 극단구간 차단)
                favorable  = pos * (price - avg_entry) / (entry_atr + 1e-9)
                next_lvl   = (add_cnt + 1) * add_step
                rsi_ok_pyr = (pos == 1 and rsi < rsi_hi) or (pos == -1 and rsi > rsi_lo)
                if add_cnt < add_lvl and favorable >= next_lvl and rsi_ok_pyr:
                    old_qty = rr / rr_base        # 가중치 비례 (수량 대신 rr 비례)
                    add_rr  = p["rr_add"]
                    new_qty = old_qty + add_rr / rr_base
                    avg_entry = (avg_entry * old_qty + price * (add_rr / rr_base)) / new_qty
                    rr     += add_rr
                    add_cnt += 1
                    if pos == 1: trail_sl = max(trail_sl, price - trail_tight * atr)
                    else:        trail_sl = min(trail_sl, price + trail_tight * atr)

        # ── 신규 진입 ────────────────────────────────────────────────────────
        if pos == 0:
            # US-004 쿨링 카운터 감소
            if cl_long  > 0: cl_long  -= 1
            if cl_short > 0: cl_short -= 1

            # 트렌드 안정화 (첫 전환 봉 차단)
            trend_stable = (prev_trend == trend)
            long_ok  = long_ok_now  and trend_stable
            short_ok = short_ok_now and trend_stable

            # US-004 쿨링 차단
            if cl_long  > 0: long_ok  = False
            if cl_short > 0: short_ok = False

            direction = 1 if long_ok else (-1 if short_ok else 0)

            # ATR 필터
            if direction != 0 and atr < price * 0.0015:
                direction = 0

            if direction != 0:
                pos        = direction
                entry_price = price
                avg_entry  = price
                entry_atr  = atr
                rr         = rr_base
                add_cnt    = 0
                partial    = False
                consec_rev = 0
                peak_px    = price
                entry_bar  = idx
                trail_sl   = (price - trail_init * atr) if pos == 1 else (price + trail_init * atr)

        prev_trend = trend
        # equity 업데이트
        if pos != 0:
            unr = pos * (price - avg_entry) / (avg_entry + 1e-9)
            eq  = capital * (1 + unr * lev * rr)
        else:
            eq  = capital
        peak_cap = max(peak_cap, eq)
        equity_curve.append(eq)

    # 미결 포지션 강제 청산
    if pos != 0 and len(df) > 0:
        price  = float(df.iloc[-1]["close"])
        raw    = pos * (price - avg_entry) / (avg_entry + 1e-9)
        pnl    = max(raw * lev * rr, -rr) - FEE_TOTAL
        capital *= (1 + pnl)
        trade_log.append({"pnl": pnl, "direction": pos, "reason": "end", "capital": round(capital,2)})

    # 메트릭 계산
    wins   = [t for t in trade_log if t["pnl"] > 0]
    losses = [t for t in trade_log if t["pnl"] <= 0]
    n      = len(trade_log)
    total_return = (capital - initial_capital) / initial_capital * 100
    wr     = len(wins) / n * 100 if n else 0
    pf     = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) \
             if losses and sum(t["pnl"] for t in losses) != 0 else float("inf")
    avg_win  = sum(t["pnl"] for t in wins)  / len(wins)  * 100 if wins   else 0
    avg_loss = sum(t["pnl"] for t in losses)/ len(losses)* 100 if losses else 0
    eq_arr   = np.array(equity_curve)
    peaks    = np.maximum.accumulate(eq_arr)
    mdd      = float(np.max((peaks - eq_arr) / (peaks + 1e-9)) * 100)
    flips    = [t for t in trade_log if t.get("reason") == "reverse_flip"]

    return {
        "capital": capital, "trade_log": trade_log,
        "metrics": {
            "total_return": total_return, "n_trades": n, "win_rate": wr,
            "profit_factor": pf, "avg_win": avg_win, "avg_loss": avg_loss, "mdd": mdd,
            "n_flips": len(flips),
        }
    }


# ── 결과 출력 ─────────────────────────────────────────────────────────────────
def print_result(label: str, res: dict, days: float):
    m   = res["metrics"]
    tl  = res["trade_log"]
    tpd = m["n_trades"] / days if days > 0 else 0
    flips = [t for t in tl if t.get("reason") == "reverse_flip"]
    ok_ret = m["total_return"] > 0
    ok_tpd = tpd >= 1.5
    # Top5 제거 후 양수 여부
    if len(tl) > 5:
        sorted_pnl = sorted(t["pnl"] for t in tl)
        top5_sum   = sum(sorted_pnl[-5:])
        r5         = (sum(t["pnl"] for t in tl) - top5_sum) * 100
    else:
        r5 = sum(t["pnl"] for t in tl) * 100
    ok_top = r5 > 0
    p = sum([ok_ret, ok_tpd, ok_top])

    print(f"\n{'='*66}")
    print(f"  {label}  [live_trader 완벽 모방]")
    print(f"{'='*66}")
    print(f"  거래수:     {m['n_trades']}  (TPD: {tpd:.2f})  {'✅' if ok_tpd else '❌'}")
    print(f"  수익률:     {m['total_return']:+.2f}%  {'✅' if ok_ret else '❌'}")
    print(f"  MDD:        {m['mdd']:.1f}%")
    print(f"  WR:         {m['win_rate']:.1f}%")
    print(f"  PF:         {m['profit_factor']:.3f}")
    print(f"  avg_win:    {m['avg_win']:+.4f}%")
    print(f"  avg_loss:   {m['avg_loss']:+.4f}%")
    print(f"  Top-5 제거: {r5:+.2f}%  {'✅' if ok_top else '❌'}")
    print(f"  flip 발동:  {m['n_flips']}건")
    print(f"  판정:       {p}/3  {'✅ 통과' if p==3 else ('⚠️ 부분' if p>=2 else '❌ 탈락')}")
    return p


# ── 데이터 로드 (backtest_antifragile.py의 load_coin_full 재사용) ──────────────
def load_coin(coin: str, start: str = None, end: str = None) -> pd.DataFrame:
    from scripts.backtest_antifragile import load_coin_full
    df = load_coin_full(coin)
    df = add_indicators_exact(df)
    if start: df = df[df.index >= start]
    if end:   df = df[df.index <  end]
    return df.copy()


# ── 랜덤 hist 검증 ─────────────────────────────────────────────────────────────
def run_hist_validation(coin: str, all_df: pd.DataFrame, seed: int, windows: int,
                        window_days: int, hist_start: str):
    rng      = random.Random(seed)
    possible = all_df[(all_df.index >= hist_start) &
                      (all_df.index <= all_df.index[-1] - pd.Timedelta(days=window_days))].index
    chosen   = sorted(rng.choices(possible, k=windows))
    p_def    = dict(DEFAULT_PARAMS)

    print(f"\n  랜덤 {windows}회 검증 ({window_days}일 창, seed={seed})")
    print(f"  {'#':>3}  {'구간':<26}  {'n':>5}  {'WR':>6}  {'TPD':>5}  {'수익':>9}  {'MDD':>5}  {'PF':>6}  {'판정'}")
    print(f"  {'─'*80}")

    passes, returns = [], []
    for i, sd in enumerate(chosen):
        ed  = sd + pd.Timedelta(days=window_days)
        seg = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
        if len(seg) < 500: continue
        res = run_af_exact(seg, **{"p": p_def})
        m   = res["metrics"]
        tpd = m["n_trades"] / window_days
        if len(res["trade_log"]) > 5:
            s5  = sorted(t["pnl"] for t in res["trade_log"])
            r5  = (sum(t["pnl"] for t in res["trade_log"]) - sum(s5[-5:])) * 100
        else:
            r5  = sum(t["pnl"] for t in res["trade_log"]) * 100
        ok  = sum([m["total_return"] > 0, tpd >= 1.5, r5 > 0])
        mark = "✅" if ok==3 else ("⚠️" if ok>=2 else "❌")
        print(f"  [{i+1:02d}] {str(sd.date())+'~'+str(ed.date()):<26}  {m['n_trades']:>5}  "
              f"{m['win_rate']:>5.1f}%  {tpd:>4.2f}  {m['total_return']:>+8.1f}%  "
              f"{m['mdd']:>4.1f}%  {m.get('profit_factor',0):>5.3f}  {mark} ({ok}/3)")
        passes.append(ok); returns.append(m["total_return"])

    print(f"\n  {'─'*60}")
    print(f"  통과(3/3): {sum(p==3 for p in passes)}/{len(passes)}")
    print(f"  수익 양수: {sum(r>0 for r in returns)}/{len(returns)}")
    print(f"  평균 수익: {np.mean(returns):+.1f}%")
    return passes, returns


# ── main ──────────────────────────────────────────────────────────────────────
HIST_START = {
    "btc": "2020-01-01", "eth": "2021-04-01",
    "sol": "2021-06-01", "xrp": "2020-06-01",
}

def main():
    parser = argparse.ArgumentParser(description="live_trader 완벽 모방 백테스트")
    parser.add_argument("--coin",    default="all", choices=["btc","eth","sol","xrp","all"])
    parser.add_argument("--mode",    default="2026",
                        choices=["2026","hist","jun1819","compare"])
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--window-days", type=int, default=91)
    args = parser.parse_args()

    coins = ["btc","eth","sol","xrp"] if args.coin == "all" else [args.coin]

    for coin in coins:
        label = coin.upper()
        print(f"\n{'█'*66}")
        print(f"  {label}/USDT — live_trader 완벽 모방 백테스트")
        print(f"  BB σ={BB_SIGMA}  flip={CONSECUTIVE_REVERSE_BARS}봉  쿨링={CONSECUTIVE_LOSS_LIMIT}연속/{DIRECTION_COOLING_BARS}봉")
        print(f"{'█'*66}")

        p_def = dict(DEFAULT_PARAMS)

        if args.mode in ("2026", "compare"):
            print(f"\n[{label}] 2026 OOS (2026-01-01 ~ 2026-05-31)")
            df26 = load_coin(coin, "2026-01-01", "2026-06-01")
            days = (df26.index[-1] - df26.index[0]).days
            res  = run_af_exact(df26, p=p_def)
            print_result(f"{label} 2026 OOS", res, days)

            if args.mode == "compare":
                # 기존 BT와 나란히 비교
                from scripts.backtest_antifragile import run_antifragile, add_indicators
                from scripts.backtest_antifragile import load_coin_full
                df26_old = load_coin_full(coin)
                df26_old = df26_old[(df26_old.index >= "2026-01-01") & (df26_old.index < "2026-06-01")].copy()
                res_old = run_antifragile(df26_old, **p_def)
                m_old   = res_old["metrics"]
                m_new   = res["metrics"]
                tpd_old = m_old["n_trades"] / days
                tpd_new = m_new["n_trades"] / days
                print(f"\n  [비교] 기존 BT vs 실거래 모방 BT")
                print(f"  {'':10} {'기존BT':>12} {'모방BT':>12} {'차이':>10}")
                for k, kn in [("total_return","수익률(%)"), ("n_trades","거래수"),
                               ("win_rate","WR(%)"), ("profit_factor","PF"), ("mdd","MDD(%)")]:
                    vo = m_old.get(k,0); vn = m_new.get(k,0)
                    if k == "n_trades": print(f"  {kn:10} {vo:>12.0f} {vn:>12.0f} {vn-vo:>+10.0f}")
                    else:               print(f"  {kn:10} {vo:>12.2f} {vn:>12.2f} {vn-vo:>+10.2f}")
                print(f"  {'TPD':10} {tpd_old:>12.2f} {tpd_new:>12.2f} {tpd_new-tpd_old:>+10.2f}")

        if args.mode in ("hist", "compare"):
            hs = HIST_START.get(coin, "2020-01-01")
            from scripts.backtest_antifragile import load_coin_full
            all_df = load_coin_full(coin)
            all_df = add_indicators_exact(all_df)
            print(f"\n[{label}] 랜덤 히스토리 검증")
            run_hist_validation(coin, all_df, args.seed, args.windows,
                                args.window_days, hs)

        if args.mode == "jun1819":
            print(f"\n[{label}] Jun 18~19 실거래 기간 (2026-06-18 03:37 ~ 2026-06-19 12:00 UTC)")
            df_jun = load_coin(coin, "2026-06-18 03:37", "2026-06-19 12:00")
            if len(df_jun) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df_jun)}봉)"); continue
            days = (df_jun.index[-1] - df_jun.index[0]).total_seconds() / 86400
            res  = run_af_exact(df_jun, p=p_def)
            print_result(f"{label} Jun18~19", res, days)
            print(f"\n  상세 거래:")
            for t in res["trade_log"]:
                dr = "롱" if t["direction"]==1 else "숏"
                flip = " ◀FLIP" if t.get("reason")=="reverse_flip" else ""
                print(f"    {dr}  pnl={t['pnl']*100:+.3f}%  {t.get('reason','')}{flip}")

if __name__ == "__main__":
    main()
