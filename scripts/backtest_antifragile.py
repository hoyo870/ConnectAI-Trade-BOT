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
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path
from hybrid_engine import compute_metrics


def load_ohlcv_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_index()

ROOT = Path(__file__).parent.parent
TRADING_FEE = 0.0005
SLIPPAGE    = 0.0002
FEE_TOTAL   = TRADING_FEE + SLIPPAGE


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
    # AdaptRSI 진입 임계값 — 실거래 prod 파라미터와 동일하게 유지
    dt_rsi_lo = 28,   # 하락추세: 롱 진입
    dt_rsi_hi = 60,   # 하락추세: 숏 진입
    rg_rsi_lo = 25,   # 횡보:     롱 진입
    rg_rsi_hi = 75,   # 횡보:     숏 진입
    ut_rsi_lo = 42,   # 상승추세: 롱 진입
    ut_rsi_hi = 75,   # 상승추세: 숏 진입
    require_bb      = False,  # BB 밴드 이탈 추가 조건 (False가 더 좋음)
    # 레버리지 — .env LEVERAGE=7과 동일
    leverage        = 7,
    # 포지션 사이징 — live_trader AF_PARAMS와 동일
    rr_base         = 0.20,   # 초기 자본 위험 비율
    rr_add          = 0.10,   # 피라미딩 추가 비율
    add_levels      = 3,      # 최대 추가 횟수
    atr_add_step    = 0.5,    # 유리방향 X×ATR마다 추가
    # Trailing Stop — prod 스윕 최적값
    trail_atr_init  = 1.8,    # 초기 trailing stop 거리 (ATR 배수)
    trail_atr_tight = 2.0,    # 피라미딩 후 tight trailing (ATR 배수)
    # 기타
    max_hold_bars   = 288,    # 최대 보유 (1일 = 288봉)
    cooling_bars    = 100,
    max_dd_cb       = 0.30,
):
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

def _normalize_index(df):
    df = df.copy()
    df.index = df.index.tz_convert(None) if df.index.tz else df.index
    return df


# 코인별 설정 (데이터 경로, 역사 시작일)
COIN_CONFIG = {
    "btc": {"label": "BTC", "hist_start": "2020-01-01"},
    "eth": {"label": "ETH", "hist_start": "2021-04-01"},
    "sol": {"label": "SOL", "hist_start": "2021-06-01"},
    "xrp": {"label": "XRP", "hist_start": "2020-06-01"},
}


def load_coin_full(coin: str) -> pd.DataFrame:
    """코인별 전체 OHLCV 로드 (BTC/ETH: 전용 경로, SOL/XRP: data/raw/ 자동 탐색)"""
    coin = coin.lower()
    label = coin.upper()
    print(f"{label} 데이터 로드 중...")

    if coin == "btc":
        pieces = []
        # data/raw/ 의 모든 BTCUSDT CSV 자동 탐색 (날짜순)
        for f in sorted((ROOT / "data/raw").glob("BTCUSDT_5m_*.csv")):
            try:
                pieces.append(_normalize_index(load_ohlcv_csv(f)))
            except Exception:
                pass
        if not pieces:
            # parquet fallback
            par = pd.read_parquet(ROOT / "data/signals_2026/backtest_2026_signals.parquet")
            pieces.append(_normalize_index(par[["open","high","low","close","volume"]].copy()))

    elif coin == "eth":
        pieces = [
            _normalize_index(pd.read_parquet(ROOT / "data/eth/ETHUSDT_5m_history.parquet")),
            _normalize_index(pd.read_parquet(ROOT / "data/eth/ETHUSDT_5m_2026.parquet")),
        ]
        # data/raw/ 의 추가 ETH CSV (최신화 파일)
        for f in sorted((ROOT / "data/raw").glob("ETHUSDT_5m_*.csv")):
            try:
                pieces.append(_normalize_index(load_ohlcv_csv(f)))
            except Exception:
                pass

    else:
        # SOL, XRP — data/raw/ 에서 패턴 탐색
        sym = f"{label}USDT"
        candidates = sorted((ROOT / "data/raw").glob(f"{sym}_5m_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"{sym} 데이터 없음. 먼저 다운로드:\n"
                f"  python src/data_fetcher.py --symbol {coin.upper()}/USDT --start 2021-01-01"
            )
        pieces = [_normalize_index(load_ohlcv_csv(f)) for f in candidates]

    all_df = pd.concat(pieces).sort_index()
    all_df = all_df[~all_df.index.duplicated(keep="last")]
    all_df = all_df[all_df["close"].notna() & (all_df["close"] > 0)]
    print(f"  {all_df.index[0].date()} ~ {all_df.index[-1].date()}  ({len(all_df):,}행)")
    return add_indicators(all_df)


# 이전 함수 이름 유지 (하위 호환)
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


# 프리셋 — live_trader.py _AF_PRESETS와 동일하게 유지 (2026-06-15 스윕 최적화)
_PRESETS = {
    "prod": dict(
        dt_rsi_lo=28, dt_rsi_hi=60, rg_rsi_lo=25, rg_rsi_hi=75,
        ut_rsi_lo=42, ut_rsi_hi=75, trail_atr_init=1.8, trail_atr_tight=2.0, add_levels=3,
    ),
    "stable": dict(
        dt_rsi_lo=30, dt_rsi_hi=60, rg_rsi_lo=25, rg_rsi_hi=75,
        ut_rsi_lo=42, ut_rsi_hi=70, trail_atr_init=1.5, trail_atr_tight=2.0, add_levels=4,
    ),
    "aggressive": dict(
        dt_rsi_lo=25, dt_rsi_hi=60, rg_rsi_lo=25, rg_rsi_hi=75,
        ut_rsi_lo=42, ut_rsi_hi=78, trail_atr_init=0.8, trail_atr_tight=1.5, add_levels=4,
    ),
    "conservative": dict(
        dt_rsi_lo=28, dt_rsi_hi=70, rg_rsi_lo=25, rg_rsi_hi=75,
        ut_rsi_lo=42, ut_rsi_hi=78, trail_atr_init=2.0, trail_atr_tight=2.5, add_levels=3,
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Antifragile Trailing Stop Backtest")
    parser.add_argument("--coin",       default="btc",
                        choices=["btc", "eth", "sol", "xrp", "both", "all"])
    parser.add_argument("--mode",       default="2026", choices=["2026", "random", "both", "june2026"])
    parser.add_argument("--preset",     default=None,   choices=list(_PRESETS.keys()),
                        help="파라미터 프리셋 (prod/stable/aggressive/conservative)")
    parser.add_argument("--windows",    type=int,   default=10)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--window-days",type=int,   default=91)
    parser.add_argument("--require-bb", action="store_true")
    parser.add_argument("--trail-init", type=float, default=1.8)
    parser.add_argument("--trail-tight",type=float, default=2.0)
    parser.add_argument("--add-step",   type=float, default=0.5)
    args = parser.parse_args()

    coin_map = {
        "both": ["btc", "eth"],
        "all":  ["btc", "eth", "sol", "xrp"],
    }
    coins = coin_map.get(args.coin, [args.coin])

    cfg = dict(
        require_bb      = args.require_bb,
        trail_atr_init  = args.trail_init,
        trail_atr_tight = args.trail_tight,
        atr_add_step    = args.add_step,
    )
    if args.preset:
        cfg.update(_PRESETS[args.preset])

    preset_label = f"preset={args.preset}" if args.preset else f"trail_init={cfg['trail_atr_init']}  trail_tight={cfg['trail_atr_tight']}"

    for coin in coins:
        label = coin.upper()
        print(f"\n{'█'*66}")
        print(f"  Antifragile Trailing Stop — {label}/USDT 5m")
        print(f"  {preset_label}  add_step={cfg.get('atr_add_step', args.add_step)}  require_bb={args.require_bb}")
        print(f"{'█'*66}")

        try:
            all_df_cache = None  # 2026 + random 모두 사용 시 재로드 방지

            if args.mode in ("2026", "both"):
                if coin == "btc":
                    par  = pd.read_parquet(ROOT / "data/signals_2026/backtest_2026_signals.parquet")
                    df26 = _normalize_index(par[par.index >= "2026-01-01"].copy())
                    df26 = add_indicators(df26)
                elif coin == "eth":
                    df26 = _normalize_index(pd.read_parquet(ROOT / "data/eth/ETHUSDT_5m_2026.parquet"))
                    df26 = add_indicators(df26)
                else:
                    # SOL/XRP: 전체 CSV에서 2026 구간 슬라이스
                    all_df_cache = load_coin_full(coin)
                    df26 = all_df_cache[all_df_cache.index >= "2026-01-01"].copy()

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
