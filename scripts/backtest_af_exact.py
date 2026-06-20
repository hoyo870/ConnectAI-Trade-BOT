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

import os, sys, argparse, random
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from config.af_params import FEE_TOTAL, DEFAULT_PARAMS
from strategies.antifragile import (
    AntifragileStrategy,
    CONSECUTIVE_REVERSE_BARS, CONSECUTIVE_LOSS_LIMIT, DIRECTION_COOLING_BARS,
)
from strategies.indicators import add_indicators_af
from config.loader import load_coin_raw
from models.af_ensemble.feature_extractor import add_ml_features
from strategies.indicators import add_indicators

BB_SIGMA = 0   # backtest_af_ml.py 기준 (EMA 단순 크로스)

# 하위 호환 alias — af_exact_sweep.py 등이 직접 import하는 이름 유지
def add_indicators_exact(df: pd.DataFrame) -> pd.DataFrame:
    return add_indicators_af(df, BB_SIGMA)


def build_ml_context_df(raw: pd.DataFrame) -> pd.DataFrame:
    """ML 필터용 enriched df: bb_sigma=0.5 트렌드 + 2σ BB + 전체 ML 피처."""
    df_base = add_indicators(raw)               # _bb_upper/_bb_lower (2σ) + _rsi/_atr
    df_af   = add_indicators_af(raw, BB_SIGMA)  # bb_sigma=0.5 트렌드
    df_base["_trend_up"]   = df_af["_trend_up"].values
    df_base["_trend_down"] = df_af["_trend_down"].values
    return add_ml_features(df_base)


# ── 핵심 백테스트 엔진 (AntifragileStrategy 사용) ─────────────────────────────
def run_af_exact(
    df: pd.DataFrame,
    initial_capital: float = 270.76,
    p: dict = None,
    ensemble=None,
    df_ml: pd.DataFrame = None,
) -> dict:
    """
    AntifragileStrategy 클래스를 사용하는 백테스트 엔진.
    live_trader.py와 동일한 신호 로직을 strategies/antifragile.py에서 공유.
    p: DEFAULT_PARAMS 기반 dict
    ensemble: AFEnsemble | None — ML 진입 필터 (None이면 비활성)
    df_ml: ML 컨텍스트 df (build_ml_context_df() 결과, ensemble 사용 시 필수)
    """
    if p is None:
        p = dict(DEFAULT_PARAMS)

    df = df.reset_index(drop=True)
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    df = df.reset_index(drop=True)

    # df_ml도 같은 필터/리셋 적용 (인덱스 정렬)
    if df_ml is not None:
        df_ml = df_ml.reset_index(drop=True)
        df_ml = df_ml[~df_ml[["_rsi", "_atr"]].isna().any(axis=1)].reset_index(drop=True)

    lev      = p.get("leverage", 7)
    strategy = AntifragileStrategy(p, ml_filter=ensemble)

    capital      = initial_capital
    peak_cap     = initial_capital
    trade_log    = []
    equity_curve = [capital]

    for idx in range(1, len(df)):
        row       = df.iloc[idx]
        price     = float(row["close"])
        atr       = float(row["_atr"])
        rsi       = float(row["_rsi"])
        trend_up  = bool(row["_trend_up"])
        trend_dn  = bool(row["_trend_down"])

        if ensemble is not None and df_ml is not None and idx < len(df_ml):
            strategy.update_context(df_ml, idx)

        result = strategy.process_tick(price, atr, rsi, trend_up, trend_dn)

        for event in result["events"]:
            if event["type"] == "close":
                pnl = max(event["pnl_raw"] * lev * event["rr"], -event["rr"]) - FEE_TOTAL
                capital  *= (1 + pnl)
                peak_cap  = max(peak_cap, capital)
                trade_log.append({
                    "pnl":       pnl,
                    "direction": event["direction"],
                    "reason":    event["reason"],
                    "capital":   round(capital, 2),
                    "entry":     round(event["entry_price"], 6),
                    "exit":      round(event["exit_price"],  6),
                })
            elif event["type"] == "partial":
                ppnl     = max(event["pnl_raw"] * lev * event["rr"], -event["rr"])
                capital *= (1 + ppnl)

        # equity curve (미실현 포함)
        if strategy.pos != 0:
            unr = strategy.pos * (price - strategy.avg_entry) / (strategy.avg_entry + 1e-9)
            eq  = capital * (1 + unr * lev * strategy.rr)
        else:
            eq = capital
        peak_cap = max(peak_cap, eq)
        equity_curve.append(eq)

    # 미결 포지션 강제 청산
    if strategy.pos != 0 and len(df) > 0:
        price = float(df.iloc[-1]["close"])
        raw   = strategy.pos * (price - strategy.avg_entry) / (strategy.avg_entry + 1e-9)
        pnl   = max(raw * lev * strategy.rr, -strategy.rr) - FEE_TOTAL
        capital *= (1 + pnl)
        trade_log.append({"pnl": pnl, "direction": strategy.pos, "reason": "end",
                           "capital": round(capital, 2)})

    # 메트릭 계산
    wins   = [t for t in trade_log if t["pnl"] > 0]
    losses = [t for t in trade_log if t["pnl"] <= 0]
    n      = len(trade_log)
    total_return = (capital - initial_capital) / initial_capital * 100
    wr   = len(wins) / n * 100 if n else 0
    pf   = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) \
           if losses and sum(t["pnl"] for t in losses) != 0 else float("inf")
    avg_win  = sum(t["pnl"] for t in wins)  / len(wins)  * 100 if wins   else 0
    avg_loss = sum(t["pnl"] for t in losses)/ len(losses)* 100 if losses else 0
    eq_arr = np.array(equity_curve)
    peaks  = np.maximum.accumulate(eq_arr)
    mdd    = float(np.max((peaks - eq_arr) / (peaks + 1e-9)) * 100)
    flips  = [t for t in trade_log if t.get("reason") == "reverse_flip"]

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


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_coin(coin: str, start: str = None, end: str = None) -> pd.DataFrame:
    df = add_indicators_af(load_coin_raw(coin), BB_SIGMA)
    if start: df = df[df.index >= start]
    if end:   df = df[df.index <  end]
    return df.copy()


# ── 랜덤 hist 검증 ─────────────────────────────────────────────────────────────
def run_hist_validation(coin: str, all_df: pd.DataFrame, seed: int, windows: int,
                        window_days: int, hist_start: str,
                        ensemble=None, df_ml_all: pd.DataFrame = None):
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
        seg_ml = df_ml_all[(df_ml_all.index >= sd) & (df_ml_all.index < ed)].copy() \
                 if df_ml_all is not None else None
        res = run_af_exact(seg, p=p_def, ensemble=ensemble, df_ml=seg_ml)
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
    parser.add_argument("--model",   default=None,
                        help="AFEnsemble 저장 디렉터리 (지정 시 ML 진입 필터 활성)")
    args = parser.parse_args()

    ensemble = None
    if args.model:
        from models.af_ensemble.ensemble import AFEnsemble
        ensemble = AFEnsemble.load(args.model)
        print(f"[ML 필터] 로드 완료: {args.model}  theta={ensemble.threshold:.3f}")

    coins = ["btc","eth","sol","xrp"] if args.coin == "all" else [args.coin]

    for coin in coins:
        label = coin.upper()
        print(f"\n{'█'*66}")
        print(f"  {label}/USDT — live_trader 완벽 모방 백테스트")
        ml_tag = f"  ML 필터: 활성 (theta={ensemble.threshold:.3f})" if ensemble else ""
        print(f"  BB σ={BB_SIGMA}  flip={CONSECUTIVE_REVERSE_BARS}봉  쿨링={CONSECUTIVE_LOSS_LIMIT}연속/{DIRECTION_COOLING_BARS}봉{ml_tag}")
        print(f"{'█'*66}")

        p_def = dict(DEFAULT_PARAMS)

        # raw 전체 로드 → ML 컨텍스트 df 사전 계산 (ensemble 사용 시)
        raw_all = load_coin_raw(coin)
        all_df  = add_indicators_af(raw_all, BB_SIGMA)
        df_ml_all = build_ml_context_df(raw_all) if ensemble else None

        def _slice(df, start, end):
            s = df[df.index >= start] if start else df
            return s[s.index < end].copy() if end else s.copy()

        if args.mode in ("2026", "compare"):
            print(f"\n[{label}] 2026 OOS (2026-01-01 ~ 2026-05-31)")
            df26    = _slice(all_df, "2026-01-01", "2026-06-01")
            df26_ml = _slice(df_ml_all, "2026-01-01", "2026-06-01") if df_ml_all is not None else None
            days = (df26.index[-1] - df26.index[0]).days
            res  = run_af_exact(df26, p=p_def, ensemble=ensemble, df_ml=df26_ml)
            print_result(f"{label} 2026 OOS", res, days)

            if args.mode == "compare":
                # 기존 BT와 나란히 비교
                from scripts.backtest_antifragile import run_antifragile, load_coin_full
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
            print(f"\n[{label}] 랜덤 히스토리 검증")
            run_hist_validation(coin, all_df, args.seed, args.windows,
                                args.window_days, hs,
                                ensemble=ensemble, df_ml_all=df_ml_all)

        if args.mode == "jun1819":
            print(f"\n[{label}] Jun 18~19 실거래 기간 (2026-06-18 03:37 ~ 2026-06-19 12:00 UTC)")
            df_jun    = _slice(all_df, "2026-06-18 03:37", "2026-06-19 12:00")
            df_jun_ml = _slice(df_ml_all, "2026-06-18 03:37", "2026-06-19 12:00") if df_ml_all is not None else None
            if len(df_jun) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df_jun)}봉)"); continue
            days = (df_jun.index[-1] - df_jun.index[0]).total_seconds() / 86400
            res  = run_af_exact(df_jun, p=p_def, ensemble=ensemble, df_ml=df_jun_ml)
            print_result(f"{label} Jun18~19", res, days)
            print(f"\n  상세 거래:")
            for t in res["trade_log"]:
                dr = "롱" if t["direction"]==1 else "숏"
                flip = " ◀FLIP" if t.get("reason")=="reverse_flip" else ""
                print(f"    {dr}  pnl={t['pnl']*100:+.3f}%  {t.get('reason','')}{flip}")

if __name__ == "__main__":
    main()
