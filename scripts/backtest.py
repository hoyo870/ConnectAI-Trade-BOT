"""
CLI 백테스트 스크립트

Usage:
  python scripts/backtest.py --coin btc --mode 2026
  python scripts/backtest.py --coin eth --mode 2026 --strategy pullback
  python scripts/backtest.py --coin both --mode random --windows 10 --seed 42
  python scripts/backtest.py --coin btc --mode custom --start 2024-01-01 --end 2024-04-01
"""
import sys, os, json, random, argparse
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import torch
import joblib

import expert_models as em
from data_pipeline import load_ohlcv_csv, add_technical_indicators, FEATURE_COLS, apply_scaler
from hybrid_engine import run_v17_backtest, compute_metrics

TRADING_FEE = 0.0005
SLIPPAGE    = 0.0002

device = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Pullback 백테스트 엔진
# ─────────────────────────────────────────────────────────────────────────────

def _fee(price):
    return price * (TRADING_FEE + SLIPPAGE)


def _get_tier(sig, tiers):
    for t in tiers:
        if sig >= t[0]:
            return t[1], t[2]
    return 0.0, 0.0


def run_pullback_backtest(
    df,
    initial_capital=10_000.0,
    tiers=None,
    price_sl=0.020,
    price_tp=0.150,
    min_hold_bars=120,
    sig_roll_window=100,
    sig_upper_thr=0.65,
    max_dd_cb=1.0,
    cooling_bars=200,
    rr_cap=0.75,
    gate_long_col=None,
    gate_short_col=None,
):
    """
    Pullback 진입:
    - 이전 봉 시그널이 tier 기준 충족 + 현재 봉 시그널이 이전보다 감소 → 진입
    - 진입 tier는 이전 봉(피크) 시그널 기준
    """
    rows = df.reset_index(drop=True)

    sig_long_roll  = rows["signal_long"].rolling(sig_roll_window).mean()
    sig_short_roll = rows["signal_short"].rolling(sig_roll_window).mean()

    capital       = initial_capital
    position      = 0
    entry_price   = 0.0
    entry_step    = 0
    entry_lev     = 1.0
    entry_rr      = 0.2
    peak_capital  = initial_capital
    cooling_left  = 0
    cb_triggers   = 0

    prev_sig_long  = 0.0
    prev_sig_short = 0.0

    equity_curve = [capital]
    trade_log    = []

    min_tier_thr = tiers[-1][0]

    for idx in range(len(rows)):
        row       = rows.iloc[idx]
        price     = float(row["close"])
        sig_long  = float(row.get("signal_long",  0.0))
        sig_short = float(row.get("signal_short", 0.0))

        roll_l = sig_long_roll.iloc[idx]
        roll_s = sig_short_roll.iloc[idx]
        roll_l = roll_l if not np.isnan(roll_l) else sig_long
        roll_s = roll_s if not np.isnan(roll_s) else sig_short
        overconfident = ((roll_l + roll_s) / 2.0) > sig_upper_thr

        # ── 미실현 equity & CB ───────────────────────────────────────────────
        if position != 0:
            pnl_raw = position * (price - entry_price) / (entry_price + 1e-9)
            equity  = capital * (1 + pnl_raw * entry_lev * entry_rr)
        else:
            equity  = capital
        peak_capital = max(peak_capital, equity)
        dd = (peak_capital - equity) / (peak_capital + 1e-9)

        if dd > max_dd_cb and cooling_left == 0:
            cooling_left = cooling_bars
            cb_triggers += 1

        if cooling_left > 0:
            cooling_left -= 1
            if position != 0:
                cp  = price - _fee(price) * position
                raw = position * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * entry_lev * entry_rr, -entry_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_step,
                                   "lev": entry_lev, "rr": entry_rr, "forced": True,
                                   "direction": position})
                position = 0; entry_price = 0.0
            equity_curve.append(capital)
            prev_sig_long  = sig_long
            prev_sig_short = sig_short
            continue

        # ── 포지션 청산 ──────────────────────────────────────────────────────
        if position != 0:
            pnl_raw   = position * (price - entry_price) / (entry_price + 1e-9)
            pnl_lev   = pnl_raw * entry_lev
            hold_bars = idx - entry_step
            sl_pnl    = price_sl * entry_lev
            tp_pnl    = price_tp * entry_lev

            reverse = hold_bars >= min_hold_bars and (
                (position ==  1 and sig_short >= tiers[-1][0]) or
                (position == -1 and sig_long  >= tiers[-1][0])
            )
            if pnl_lev <= -0.9 or pnl_lev <= -sl_pnl or pnl_lev >= tp_pnl or reverse:
                cp  = price - _fee(price) * position
                raw = position * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * entry_lev * entry_rr, -entry_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": hold_bars,
                                   "lev": entry_lev, "rr": entry_rr, "forced": False,
                                   "direction": position})
                position = 0; entry_price = 0.0

        # ── 신규 진입 (pullback 조건) ─────────────────────────────────────────
        gate_long_ok  = (gate_long_col  is None) or (float(row.get(gate_long_col,  0.0)) > 0)
        gate_short_ok = (gate_short_col is None) or (float(row.get(gate_short_col, 0.0)) > 0)

        if position == 0:
            long_pullback  = gate_long_ok  and (prev_sig_long  >= min_tier_thr) and (sig_long  < prev_sig_long)  and (sig_long  > sig_short)
            short_pullback = gate_short_ok and (prev_sig_short >= min_tier_thr) and (sig_short < prev_sig_short) and (sig_short > sig_long)

            if long_pullback and (not short_pullback or prev_sig_long >= prev_sig_short):
                lev, rr = _get_tier(prev_sig_long, tiers)
                if lev > 0:
                    if overconfident:
                        ti = next((i for i, t in enumerate(tiers) if prev_sig_long >= t[0]), None)
                        if ti is not None and ti + 1 < len(tiers):
                            _, lev, rr = tiers[ti + 1]
                        else:
                            lev, rr = lev * 0.5, rr * 0.8
                    entry_price = price + _fee(price)
                    position    = 1
                    entry_lev   = lev
                    entry_rr    = min(rr, rr_cap)
                    entry_step  = idx
            elif short_pullback:
                lev, rr = _get_tier(prev_sig_short, tiers)
                if lev > 0:
                    if overconfident:
                        ti = next((i for i, t in enumerate(tiers) if prev_sig_short >= t[0]), None)
                        if ti is not None and ti + 1 < len(tiers):
                            _, lev, rr = tiers[ti + 1]
                        else:
                            lev, rr = lev * 0.5, rr * 0.8
                    entry_price = price - _fee(price)
                    position    = -1
                    entry_lev   = lev
                    entry_rr    = min(rr, rr_cap)
                    entry_step  = idx

        equity_curve.append(capital)
        prev_sig_long  = sig_long
        prev_sig_short = sig_short

    metrics = compute_metrics(equity_curve, trade_log)
    metrics["cb_triggers"] = cb_triggers
    if trade_log:
        lev_used = [t["lev"] for t in trade_log]
        metrics["avg_lev"]        = round(float(np.mean(lev_used)), 2)
        metrics["max_lev"]        = round(float(np.max(lev_used)), 1)
        metrics["trades_per_day"] = round(len(trade_log) / (len(rows) / 288), 2)
        metrics["long_cnt"]  = sum(1 for t in trade_log if t.get("direction", 1) ==  1)
        metrics["short_cnt"] = sum(1 for t in trade_log if t.get("direction", 1) == -1)
    return {"metrics": metrics, "equity_curve": equity_curve, "trade_log": trade_log}


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 준비
# ─────────────────────────────────────────────────────────────────────────────

def load_prod_models(coin):
    n_feat = len(FEATURE_COLS)
    models = {}
    for role in ["long", "short", "context"]:
        m = em.ExpertModel(n_feat)
        m.load_state_dict(torch.load(f"models/production/{coin}_expert_{role}.pt",
                                     map_location=device))
        models[role] = m.to(device).eval()
    return models


def batch_inference(df, models, seq_len=60):
    feats = df[FEATURE_COLS].values.astype(np.float32)
    n     = len(feats)
    sigs  = {r: np.zeros(n) for r in models}
    idx_r = np.arange(seq_len, n)
    with torch.no_grad():
        for st in range(0, len(idx_r), 512):
            batch = idx_r[st:st+512]
            x = torch.tensor(
                np.stack([feats[i-seq_len:i] for i in batch]),
                dtype=torch.float32,
            ).to(device)
            for role, model in models.items():
                sigs[role][batch] = torch.sigmoid(model(x)).cpu().numpy()
    df = df.copy()
    df["signal_long"]    = sigs["long"]
    df["signal_short"]   = sigs["short"]
    df["signal_context"] = sigs["context"]
    return df


def apply_phase2_gate(df):
    close = df["_close_raw"]
    rsi   = df["_rsi_raw"]
    vol_r = df["_vol_raw"]
    ema_f = close.ewm(span=9,  adjust=False).mean()
    ema_s = close.ewm(span=21, adjust=False).mean()
    gl = (ema_f > ema_s) & (rsi >= 50) & (rsi <= 60) & (vol_r >= 0.8)
    gs = (ema_f < ema_s) & (rsi >= 30) & (rsi <= 50) & (vol_r >= 0.8)
    cl1h   = close.resample("1h").last().ffill()
    ema_1h = cl1h.ewm(span=20, adjust=False).mean()
    tup    = (cl1h > ema_1h).reindex(df.index, method="ffill").fillna(False)
    df = df.copy()
    df["gate_long"]  = (gl & tup).astype(int)
    df["gate_short"] = (gs & ~tup).astype(int)
    return df


def prepare_btc():
    models = load_prod_models("btc")
    scaler = joblib.load("models/production/btc_scaler.pkl")
    raw    = load_ohlcv_csv("data/raw/BTCUSDT_5m_20200101_20251231.csv")
    d2026  = load_ohlcv_csv("data/raw/BTCUSDT_5m_20260101_20260520.csv")
    for d in [raw, d2026]:
        if d.index.tz is not None:
            d.index = d.index.tz_convert("UTC").tz_localize(None)
    all_df = pd.concat([raw, d2026]).sort_index()
    all_df = all_df[~all_df.index.duplicated(keep="last")]
    feat   = add_technical_indicators(all_df)
    feat.dropna(subset=FEATURE_COLS, inplace=True)
    feat   = apply_scaler(feat, scaler=scaler)
    if feat.index.tz is not None:
        feat.index = feat.index.tz_convert("UTC").tz_localize(None)
    feat = batch_inference(feat, models, seq_len=60)
    print(f"  BTC 완료: {feat.index[0].date()} ~ {feat.index[-1].date()}")
    return feat


def prepare_eth():
    models = load_prod_models("eth")
    scaler = joblib.load("models/production/eth_scaler.pkl")
    hist   = pd.read_parquet("data/eth/ETHUSDT_5m_history.parquet")
    d2026  = pd.read_parquet("data/eth/ETHUSDT_5m_2026.parquet")
    for d in [hist, d2026]:
        if d.index.tz is not None:
            d.index = d.index.tz_convert("UTC").tz_localize(None)
    all_df = pd.concat([hist, d2026]).sort_index()
    all_df = all_df[~all_df.index.duplicated(keep="last")]
    feat   = add_technical_indicators(all_df)
    feat.dropna(subset=FEATURE_COLS, inplace=True)
    feat["_rsi_raw"]   = feat["rsi_14"].copy()
    feat["_vol_raw"]   = feat["vol_ratio"].copy()
    feat["_close_raw"] = feat["close"].copy()
    feat = apply_scaler(feat, scaler=scaler)
    if feat.index.tz is not None:
        feat.index = feat.index.tz_convert("UTC").tz_localize(None)
    feat = apply_phase2_gate(feat)
    feat = batch_inference(feat, models, seq_len=60)
    print(f"  ETH 완료: {feat.index[0].date()} ~ {feat.index[-1].date()}")
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 리포트 유틸
# ─────────────────────────────────────────────────────────────────────────────

def consecutive_stats(trade_log):
    if not trade_log:
        return 0, 0
    max_w = max_l = cur = 0
    prev = None
    for t in trade_log:
        win = t["pnl"] > 0
        cur = cur + 1 if win == prev else 1
        prev = win
        if win:
            max_w = max(max_w, cur)
        else:
            max_l = max(max_l, cur)
    return max_w, max_l


def build_report(result, days, label=""):
    m  = result["metrics"]
    tl = result["trade_log"]
    long_t  = [t for t in tl if t.get("direction", 1) ==  1]
    short_t = [t for t in tl if t.get("direction", 1) == -1]
    def wr(ts):   return sum(1 for t in ts if t["pnl"] > 0) / len(ts) * 100 if ts else 0.0
    def apnl(ts): return np.mean([t["pnl"] for t in ts]) * 100 if ts else 0.0
    max_cw, max_cl = consecutive_stats(tl)
    tpd       = m.get("trades_per_day", round(len(tl) / max(days, 1), 2))
    daily_ret = m["total_return"] / max(days, 1)
    return {
        "label": label, "days": days,
        "n_trades": m["n_trades"], "long_cnt": len(long_t), "short_cnt": len(short_t),
        "win_rate": m["win_rate"], "long_wr": round(wr(long_t), 1), "short_wr": round(wr(short_t), 1),
        "tpd": tpd,
        "daily_ret": round(daily_ret, 2), "monthly_ret": round(daily_ret * 30, 1),
        "total_ret": m["total_return"], "mdd": m["mdd"],
        "sharpe": m["sharpe"], "sortino": m["sortino"], "profit_factor": m["profit_factor"],
        "avg_hold_bars": m["avg_hold_bars"],
        "avg_pnl_long": round(apnl(long_t), 3), "avg_pnl_short": round(apnl(short_t), 3),
        "max_cons_win": max_cw, "max_cons_loss": max_cl,
    }


def print_report(r, strategy="instant"):
    ok_wr  = r["win_rate"]  >= 45.0
    ok_tpd = r["tpd"]       >= 1.5
    ok_day = r["daily_ret"] >= 1.0
    passed = sum([ok_wr, ok_tpd, ok_day])
    print(f"\n{'='*64}")
    print(f"  {r['label']}  [{strategy}]  ({r['days']}일)")
    print(f"{'='*64}")
    print(f"  총 거래수       : {r['n_trades']:>5}  (롱 {r['long_cnt']} / 숏 {r['short_cnt']})")
    print(f"  일일 거래수     : {r['tpd']:>5.2f}  {'✅' if ok_tpd else '❌'} (목표 ≥1.5)")
    print(f"  평균 보유봉수   : {r['avg_hold_bars']:>5.1f}봉 ({r['avg_hold_bars']*5/60:.1f}h)")
    print(f"  연속 승/패 최대 : {r['max_cons_win']}연승 / {r['max_cons_loss']}연패")
    print(f"  전체 승률       : {r['win_rate']:>5.1f}%  {'✅' if ok_wr else '❌'} (목표 ≥45%)")
    print(f"  롱 승률         : {r['long_wr']:>5.1f}%  (n={r['long_cnt']})")
    print(f"  숏 승률         : {r['short_wr']:>5.1f}%  (n={r['short_cnt']})")
    print(f"  일 평균 수익률  : {r['daily_ret']:>+6.2f}%  {'✅' if ok_day else '❌'} (목표 ≥1%)")
    print(f"  월 평균 수익률  : {r['monthly_ret']:>+6.1f}%")
    print(f"  총 수익률       : {r['total_ret']:>+7.1f}%")
    print(f"  롱 평균 PnL     : {r['avg_pnl_long']:>+6.3f}%/trade")
    print(f"  숏 평균 PnL     : {r['avg_pnl_short']:>+6.3f}%/trade")
    print(f"  MDD             : {r['mdd']:>5.1f}%")
    print(f"  Sharpe / Sortino: {r['sharpe']:.3f} / {r['sortino']:.3f}")
    print(f"  Profit Factor   : {r['profit_factor']:.3f}")
    print(f"  목표: {passed}/3  승률{'✅' if ok_wr else '❌'} | 일수익{'✅' if ok_day else '❌'} | TPD{'✅' if ok_tpd else '❌'}")
    print(f"{'='*64}")
    return passed


def print_summary_table(records, title):
    print(f"\n{'─'*94}")
    print(f"  {title}")
    print(f"{'─'*94}")
    print(f"  {'구간':<28} {'n':>4} {'WR':>5} {'롱WR':>5} {'숏WR':>5} "
          f"{'TPD':>5} {'일수익':>6} {'총수익':>7} {'MDD':>5} {'PF':>5} {'pass':>4}")
    print(f"  {'─'*92}")
    pass_cnt = 0
    for r in records:
        ok_wr  = r["win_rate"]  >= 45.0
        ok_tpd = r["tpd"]       >= 1.5
        ok_day = r["daily_ret"] >= 1.0
        p = sum([ok_wr, ok_tpd, ok_day])
        pass_cnt += (p == 3)
        mark = f"{'✅' if ok_wr else '❌'}{'✅' if ok_tpd else '❌'}{'✅' if ok_day else '❌'}"
        print(f"  {r['label']:<28} {r['n_trades']:>4} "
              f"{r['win_rate']:>4.1f}% {r['long_wr']:>4.1f}% {r['short_wr']:>4.1f}% "
              f"{r['tpd']:>5.2f} {r['daily_ret']:>+5.2f}% "
              f"{r['total_ret']:>+6.1f}% {r['mdd']:>4.1f}% {r['profit_factor']:>4.2f} {mark}")
    if records:
        avgs = {k: np.mean([r[k] for r in records])
                for k in ["win_rate", "tpd", "daily_ret", "total_ret", "mdd", "profit_factor"]}
        print(f"  {'─'*92}")
        print(f"  {'평균':<28} {'':>4} "
              f"{avgs['win_rate']:>4.1f}% {'':>5} {'':>5} "
              f"{avgs['tpd']:>5.2f} {avgs['daily_ret']:>+5.2f}% "
              f"{avgs['total_ret']:>+6.1f}% {avgs['mdd']:>4.1f}% {avgs['profit_factor']:>4.2f}")
        print(f"  통과(3/3): {pass_cnt}/{len(records)} 구간")
    print(f"{'─'*94}\n")


def generate_random_windows(n=10, seed=42, period_days=91):
    rng = random.Random(seed)
    start_bound  = pd.Timestamp("2021-06-01")
    end_bound    = pd.Timestamp("2025-09-30")
    total_days_r = (end_bound - start_bound).days - period_days
    starts = sorted([
        start_bound + pd.Timedelta(days=rng.randint(0, total_days_r))
        for _ in range(n)
    ])
    return starts


# ─────────────────────────────────────────────────────────────────────────────
# 백테스트 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_one(feat, coin, strategy, params, start, end, days, label):
    df = feat[(feat.index >= start) & (feat.index < end)].copy() if end else \
         feat[feat.index >= start].copy()

    if strategy == "pullback":
        kw = dict(
            tiers=[tuple(t) for t in params["tiers"]],
            price_sl=params["price_sl"],
            price_tp=params["price_tp"],
            min_hold_bars=params.get("min_hold_bars", 120),
            max_dd_cb=params.get("max_dd_cb", 1.0),
            cooling_bars=params.get("cooling_bars", 200),
            rr_cap=params["rr_cap"],
            sig_upper_thr=params.get("sig_upper_thr", 0.65),
        )
        if coin == "eth":
            kw["gate_long_col"]  = "gate_long"
            kw["gate_short_col"] = "gate_short"
            kw["min_hold_bars"]  = 180
        result = run_pullback_backtest(df, **kw)
    else:
        kw = dict(
            tiers=[tuple(t) for t in params["tiers"]],
            price_sl=params["price_sl"],
            price_tp=params["price_tp"],
            min_hold_bars=params.get("min_hold_bars", 120),
            max_dd_cb=params.get("max_dd_cb", 0.35),
            cooling_bars=params.get("cooling_bars", 200),
            rr_cap=params["rr_cap"],
            sig_upper_thr=params.get("sig_upper_thr", 0.65),
        )
        if coin == "eth":
            kw["gate_long_col"]  = "gate_long"
            kw["gate_short_col"] = "gate_short"
            kw["min_hold_bars"]  = 180
        result = run_v17_backtest(df, **kw)

    return build_report(result, days, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="백테스트 CLI")
    parser.add_argument("--coin",     choices=["btc", "eth", "both"], default="both")
    parser.add_argument("--mode",     choices=["2026", "random", "custom"], default="2026")
    parser.add_argument("--strategy", choices=["instant", "pullback"], default="instant")
    parser.add_argument("--windows",  type=int, default=10, help="랜덤 구간 수 (random 모드)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--start",    type=str, default=None, help="custom 모드 시작일 YYYY-MM-DD")
    parser.add_argument("--end",      type=str, default=None, help="custom 모드 종료일 YYYY-MM-DD")
    args = parser.parse_args()

    coins = ["btc", "eth"] if args.coin == "both" else [args.coin]

    with open("models/production/btc_params.json") as f:
        btc_p = json.load(f)
    with open("models/production/eth_params.json") as f:
        eth_p = json.load(f)
    params = {"btc": btc_p, "eth": eth_p}

    print(f"\n디바이스: {device}")
    print("\n" + "="*64)
    print("  데이터 로드 & 신호 추출")
    print("="*64)

    feat = {}
    if "btc" in coins:
        feat["btc"] = prepare_btc()
    if "eth" in coins:
        feat["eth"] = prepare_eth()

    strat = args.strategy

    # ── 2026 모드 ────────────────────────────────────────────────────────────
    if args.mode == "2026":
        print(f"\n\n{'█'*64}")
        print(f"  2026-01-01 기본 백테스트  [{strat}]")
        print(f"{'█'*64}")

        recs = {}
        for coin in coins:
            df_all = feat[coin]
            df26   = df_all[df_all.index >= "2026-01-01"]
            days   = (df26.index[-1] - df26.index[0]).days
            label  = f"{coin.upper()}/USDT  2026-01-01~{df26.index[-1].date()}"
            r = run_one(feat[coin], coin, strat, params[coin],
                        start="2026-01-01", end=None, days=days, label=label)
            recs[coin] = r
            print_report(r, strategy=strat)

        if len(coins) == 2:
            b, e = recs["btc"], recs["eth"]
            print(f"\n  ─ BTC+ETH 합산 평균 ─")
            print(f"    WR    : {(b['win_rate']+e['win_rate'])/2:.1f}%")
            print(f"    TPD   : {(b['tpd']+e['tpd'])/2:.2f}")
            print(f"    daily : {(b['daily_ret']+e['daily_ret'])/2:+.2f}%")
            print(f"    total : {(b['total_ret']+e['total_ret'])/2:+.1f}%")
            print(f"    MDD   : {(b['mdd']+e['mdd'])/2:.1f}%")

    # ── random 모드 ──────────────────────────────────────────────────────────
    elif args.mode == "random":
        period_days = 91
        starts = generate_random_windows(n=args.windows, seed=args.seed,
                                         period_days=period_days)

        print(f"\n\n{'█'*64}")
        print(f"  랜덤 {args.windows}회 백테스트  [{strat}]  (seed={args.seed})")
        print(f"{'█'*64}")
        print(f"\n  랜덤 시작일 (seed={args.seed}):")
        for i, s in enumerate(starts, 1):
            print(f"    {i:2d}. {s.date()} ~ {(s + pd.Timedelta(days=period_days)).date()}")

        all_recs = {coin: [] for coin in coins}

        for i, start in enumerate(starts, 1):
            end   = start + pd.Timedelta(days=period_days)
            label = f"[{i:02d}] {start.date()}~{end.date()}"

            line_parts = []
            for coin in coins:
                df_w = feat[coin][(feat[coin].index >= start) & (feat[coin].index < end)]
                if len(df_w) < 200:
                    continue
                r = run_one(feat[coin], coin, strat, params[coin],
                            start=start, end=end, days=period_days, label=label)
                all_recs[coin].append(r)
                line_parts.append(
                    f"{coin.upper()} WR={r['win_rate']:.1f}% TPD={r['tpd']:.2f} daily={r['daily_ret']:+.2f}%"
                )
            if line_parts:
                print(f"  [{i:02d}] " + "  |  ".join(line_parts))

        for coin in coins:
            print_summary_table(all_recs[coin], f"{coin.upper()} [{strat}] 랜덤 {args.windows}회")

    # ── custom 모드 ──────────────────────────────────────────────────────────
    elif args.mode == "custom":
        if not args.start:
            parser.error("--mode custom 은 --start 필수")
        start = pd.Timestamp(args.start)
        end   = pd.Timestamp(args.end) if args.end else None
        days  = (end - start).days if end else None

        print(f"\n\n{'█'*64}")
        print(f"  커스텀 구간 백테스트  [{strat}]")
        print(f"  {args.start} ~ {args.end or '끝'}")
        print(f"{'█'*64}")

        for coin in coins:
            df_w = feat[coin]
            if end:
                df_w = df_w[(df_w.index >= start) & (df_w.index < end)]
            else:
                df_w = df_w[df_w.index >= start]
            if days is None:
                days = (df_w.index[-1] - df_w.index[0]).days
            label = f"{coin.upper()}/USDT  {args.start}~{args.end or df_w.index[-1].date()}"
            r = run_one(feat[coin], coin, strat, params[coin],
                        start=start, end=end, days=days, label=label)
            print_report(r, strategy=strat)

    print("\n✅ 백테스트 완료")


if __name__ == "__main__":
    main()
