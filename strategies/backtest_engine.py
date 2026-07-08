"""
AntifragileBacktestRunner — 유일한 공식 백테스트 엔진.

live_trader.py와 동일한 AntifragileStrategy + AFEnsemble ML 필터를 사용.
반드시 ML 앙상블과 함께 실행해야 한다 (ML 없는 백테스트 미지원).

Usage:
    from strategies.backtest_engine import AntifragileBacktestRunner
    runner = AntifragileBacktestRunner.from_saved("models/af_ensemble/saved")
    df, df_ml = runner.load_coin("btc", start="2026-01-01", end="2026-06-01")
    result = runner.run(df, df_ml)
    runner.print_result("BTC 2026 OOS", result, days=151)
"""

import os
import random

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from config.af_params import FEE_TOTAL, DEFAULT_PARAMS, EMERGENCY_SL_ATR, FUNDING_RATE_8H, FUNDING_BARS_8H
from config.loader import load_coin_raw
from models.af_ensemble.ensemble import AFEnsemble
from models.af_ensemble.feature_extractor import add_ml_features
from strategies.antifragile import AntifragileStrategy
from strategies.indicators import add_indicators, add_indicators_af

_BB_SIGMA = 0  # live_trader.py와 동일: bb_sigma=0 (EMA 크로스, BB 미사용)


def _net_pnl(pnl_raw: float, rr: float, hold_bars: int, lev: float,
             cost_mult: float = 1.0) -> float:
    """비용 차감 후 자본 수익률.
    비용 = 왕복 수수료+슬리피지(2×FEE_TOTAL, 진입+청산) + 보유 funding(FUNDING_BARS_8H봉=8h당 FUNDING_RATE_8H).
    cost_mult: 비용 민감도 배수(1=기본, 2/3=보수적 스트레스). 손실은 -rr로 캡(청산)."""
    fee   = 2 * FEE_TOTAL
    fund  = (hold_bars / FUNDING_BARS_8H) * FUNDING_RATE_8H
    gross = pnl_raw - (fee + fund) * cost_mult
    return max(gross * lev * rr, -rr)


def robust_metrics(trade_log: list) -> dict:
    """아웃라이어/꼬리위험에 강건한 보조 지표 (약한 3/3 게이트 보완).
    sharpe/sortino: 거래당 손익 평균/표준편차(하방). top5_share: 양수수익 중 상위 5거래 비중
    (1에 가까울수록 소수 대박에 의존). max_consec_loss: 최대 연속손실. pct_from_top10: 전체
    순수익 중 상위 10% 거래 기여율(>100%면 나머지는 순손실 = 아웃라이어 의존)."""
    pnls = [t["pnl"] for t in trade_log]
    if not pnls:
        return {"sharpe": 0.0, "sortino": 0.0, "top5_share": 0.0,
                "max_consec_loss": 0, "pct_from_top10": 0.0}
    arr  = np.array(pnls, dtype=float)
    mean, std = arr.mean(), arr.std()
    downside  = arr[arr < 0]
    dstd = downside.std() if len(downside) else 0.0
    pos  = arr[arr > 0]
    total_pos = float(pos.sum())
    top5_share = float(np.sort(pos)[-5:].sum() / total_pos) if total_pos > 0 else 0.0
    mc = cur = 0
    for pl in pnls:
        cur = cur + 1 if pl <= 0 else 0
        mc  = max(mc, cur)
    k   = max(1, int(len(arr) * 0.1))
    tot = float(arr.sum())
    pct_from_top10 = float(np.sort(arr)[-k:].sum() / tot * 100) if tot > 0 else float("inf")
    return {
        "sharpe":  float(mean / std) if std > 0 else 0.0,
        "sortino": float(mean / dstd) if dstd > 0 else 0.0,
        "top5_share": top5_share,
        "max_consec_loss": int(mc),
        "pct_from_top10": pct_from_top10,
    }


class AntifragileBacktestRunner:
    """
    Antifragile + ML 앙상블 백테스트 엔진.

    live_trader.py의 process_tick_af()와 동일한 AntifragileStrategy를 사용.
    ML 필터는 필수이며 비활성화 불가.
    """

    def __init__(self, ensemble: AFEnsemble, params: dict = None):
        if ensemble is None:
            raise ValueError("ML 앙상블은 필수입니다. AntifragileBacktestRunner.from_saved()를 사용하세요.")
        self.ensemble = ensemble
        self.params = dict(params or DEFAULT_PARAMS)

    @classmethod
    def from_saved(cls, model_dir: str = "models/af_ensemble/saved", params: dict = None) -> "AntifragileBacktestRunner":
        """저장된 모델 경로에서 앙상블 로드 후 인스턴스 생성."""
        ensemble = AFEnsemble.load(model_dir)
        return cls(ensemble, params)

    # ── 데이터 로드 ──────────────────────────────────────────────────────────────

    def load_coin(self, coin: str, start: str = None, end: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        코인 OHLCV를 로드해 (df, df_ml) 쌍을 반환.
        df:    add_indicators_af 결과 (백테스트 루프용)
        df_ml: build_ml_context() 결과 (ML 피처용)
        """
        raw = load_coin_raw(coin)
        df    = add_indicators_af(raw, _BB_SIGMA)
        df_ml = self._build_ml_context(raw)
        if start: df = df[df.index >= start]; df_ml = df_ml[df_ml.index >= start]
        if end:   df = df[df.index <  end];   df_ml = df_ml[df_ml.index <  end]
        return df.copy(), df_ml.copy()

    @staticmethod
    def _build_ml_context(raw: pd.DataFrame) -> pd.DataFrame:
        """ML 필터용 enriched df: bb_sigma=0 트렌드 + 2σ BB + 전체 ML 피처."""
        df_base = add_indicators(raw)
        df_af   = add_indicators_af(raw, _BB_SIGMA)
        df_base["_trend_up"]   = df_af["_trend_up"].values
        df_base["_trend_down"] = df_af["_trend_down"].values
        return add_ml_features(df_base)

    # ── 핵심 백테스트 루프 ────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, df_ml: pd.DataFrame,
            initial_capital: float = 10_000.0,
            intrabar_emergency_sl: bool = True,
            cost_mult: float = 1.0) -> dict:
        """
        백테스트 실행 (기본 초기자본 $10,000).
        df:    load_coin()에서 반환된 첫 번째 요소
        df_ml: load_coin()에서 반환된 두 번째 요소
        intrabar_emergency_sl: True면 실거래 거래소 위탁 SL(EMERGENCY_SL_ATR)을 봉 high/low로
            intrabar 체결 모사. tight trailing stop은 실거래도 봉 종가 기준이라 close로 유지.
        """
        df = df.reset_index()          # datetime 인덱스 → '_ts' 컬럼으로 보존
        df.rename(columns={"timestamp": "_ts", "index": "_ts"}, inplace=True)
        df.dropna(subset=["_rsi", "_atr"], inplace=True)
        df = df.reset_index(drop=True)

        df_ml = df_ml.reset_index(drop=True)
        df_ml = df_ml[~df_ml[["_rsi", "_atr"]].isna().any(axis=1)].reset_index(drop=True)

        lev      = self.params.get("leverage", 7)
        strategy = AntifragileStrategy(self.params, ml_filter=self.ensemble)

        capital      = initial_capital
        peak_cap     = initial_capital
        trade_log    = []
        equity_curve = [capital]
        entry_ts     = None   # 현재 포지션 진입 타임스탬프 추적
        entry_idx    = 0      # funding 비용용 진입 봉 인덱스
        have_hl      = {"high", "low"}.issubset(df.columns)

        for idx in range(1, len(df)):
            row      = df.iloc[idx]
            price    = float(row["close"])
            atr      = float(row["_atr"])
            rsi      = float(row["_rsi"])
            trend_up = bool(row["_trend_up"])
            trend_dn = bool(row["_trend_down"])
            cur_ts   = df.at[idx, "_ts"] if "_ts" in df.columns else None

            # ── intrabar emergency SL: 거래소 위탁 SL(6×ATR)을 봉 wick로 체결 모사 ──
            # 실거래는 봉 사이 wick가 emergency SL에 닿으면 거래소가 즉시 체결(EX_SL).
            # 종가 기준 trailing stop만 보는 백테스트는 이 꼬리손실을 놓쳐 MDD를 과소평가함.
            emergency_closed = False
            if intrabar_emergency_sl and strategy.pos != 0 and have_hl:
                pos0    = strategy.pos
                ent_atr = strategy.entry_atr or atr
                emerg   = (strategy.avg_entry - EMERGENCY_SL_ATR * ent_atr) if pos0 == 1 \
                          else (strategy.avg_entry + EMERGENCY_SL_ATR * ent_atr)
                hit = (pos0 == 1 and float(row["low"])  <= emerg) or \
                      (pos0 == -1 and float(row["high"]) >= emerg)
                if hit:
                    rr_c    = strategy.rr
                    pnl_raw = pos0 * (emerg - strategy.avg_entry) / (strategy.avg_entry + 1e-9)
                    pnl     = _net_pnl(pnl_raw, rr_c, idx - entry_idx, lev, cost_mult)
                    capital *= (1 + pnl)
                    peak_cap = max(peak_cap, capital)
                    trade_log.append({
                        "pnl": pnl, "direction": pos0, "reason": "EX_SL",
                        "capital": round(capital, 2),
                        "entry": round(strategy.avg_entry, 6), "exit": round(emerg, 6),
                        "entry_ts": entry_ts, "exit_ts": cur_ts,
                    })
                    entry_ts = None
                    # process_tick close와 동일하게 포지션 상태 리셋
                    strategy.pos = 0; strategy.avg_entry = 0.0; strategy.entry_atr = 0.0
                    strategy.rr = strategy.p["rr_base"]; strategy.add_cnt = 0
                    strategy.trail_sl = 0.0; strategy.peak_px = 0.0
                    emergency_closed = True

            if idx < len(df_ml):
                strategy.update_context(df_ml, idx)

            # emergency 체결 시 같은 봉 재진입 차단(실거래는 다음 봉에 EX_SL 감지 후 진입 가능)
            result = strategy.process_tick(price, atr, rsi, trend_up, trend_dn,
                                           block_entry=emergency_closed)

            for event in result["events"]:
                if event["type"] == "close":
                    # 비용(왕복 수수료+슬리피지+funding) 차감 후 lev×rr 증폭. _net_pnl 참조.
                    pnl = _net_pnl(event["pnl_raw"], event["rr"], idx - entry_idx, lev, cost_mult)
                    capital  *= (1 + pnl)
                    peak_cap  = max(peak_cap, capital)
                    trade_log.append({
                        "pnl":       pnl,
                        "direction": event["direction"],
                        "reason":    event["reason"],
                        "capital":   round(capital, 2),
                        "entry":     round(event["entry_price"], 6),
                        "exit":      round(event["exit_price"],  6),
                        "entry_ts":  entry_ts,
                        "exit_ts":   cur_ts,
                    })
                    entry_ts = None  # 청산 후 초기화

            # 진입 감지: 신규 진입(prev_pos==0) 또는 flip 후 재진입(entry_ts가 None인데 포지션 있음)
            if strategy.pos != 0 and entry_ts is None:
                entry_ts  = cur_ts
                entry_idx = idx

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
            pnl   = _net_pnl(raw, strategy.rr, (len(df) - 1) - entry_idx, lev, cost_mult)
            capital *= (1 + pnl)
            trade_log.append({"pnl": pnl, "direction": strategy.pos, "reason": "end",
                               "capital": round(capital, 2)})

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
            "capital": capital,
            "trade_log": trade_log,
            "equity_curve": equity_curve,
            "metrics": {
                "total_return": total_return, "n_trades": n, "win_rate": wr,
                "profit_factor": pf, "avg_win": avg_win, "avg_loss": avg_loss,
                "mdd": mdd, "n_flips": len(flips),
                **robust_metrics(trade_log),
            },
        }

    # ── 랜덤 hist 검증 ────────────────────────────────────────────────────────────

    def run_hist_validation(self, coin: str, seed: int = 42, windows: int = 10,
                            window_days: int = 91, hist_start: str = "2020-01-01") -> tuple:
        raw    = load_coin_raw(coin)
        all_df = add_indicators_af(raw, _BB_SIGMA)
        df_ml_all = self._build_ml_context(raw)
        all_df.dropna(subset=["_rsi", "_atr"], inplace=True)

        rng      = random.Random(seed)
        possible = all_df[
            (all_df.index >= hist_start) &
            (all_df.index <= all_df.index[-1] - pd.Timedelta(days=window_days))
        ].index
        chosen = sorted(rng.choices(possible, k=windows))

        print(f"\n  랜덤 {windows}회 검증 ({window_days}일 창, seed={seed})")
        print(f"  {'#':>3}  {'구간':<26}  {'n':>5}  {'WR':>6}  {'TPD':>5}  {'수익':>9}  {'MDD':>5}  {'PF':>6}  {'판정'}")
        print(f"  {'─'*80}")

        passes, returns = [], []
        for i, sd in enumerate(chosen):
            ed     = sd + pd.Timedelta(days=window_days)
            seg    = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
            seg_ml = df_ml_all[(df_ml_all.index >= sd) & (df_ml_all.index < ed)].copy()
            if len(seg) < 500:
                continue
            res = self.run(seg, seg_ml)
            m   = res["metrics"]
            tpd = m["n_trades"] / window_days
            tl  = res["trade_log"]
            r5  = (sum(t["pnl"] for t in tl) - sum(sorted(t["pnl"] for t in tl)[-5:])) * 100 \
                  if len(tl) > 5 else sum(t["pnl"] for t in tl) * 100
            ok   = sum([m["total_return"] > 0, tpd >= 1.5, r5 > 0])
            mark = "✅" if ok == 3 else ("⚠️" if ok >= 2 else "❌")
            print(f"  [{i+1:02d}] {str(sd.date())+'~'+str(ed.date()):<26}  {m['n_trades']:>5}  "
                  f"{m['win_rate']:>5.1f}%  {tpd:>4.2f}  {m['total_return']:>+8.1f}%  "
                  f"{m['mdd']:>4.1f}%  {m.get('profit_factor',0):>5.3f}  {mark} ({ok}/3)")
            passes.append(ok); returns.append(m["total_return"])

        print(f"\n  {'─'*60}")
        print(f"  통과(3/3): {sum(p==3 for p in passes)}/{len(passes)}")
        print(f"  수익 양수: {sum(r>0 for r in returns)}/{len(returns)}")
        if returns:
            print(f"  평균 수익: {np.mean(returns):+.1f}%")
        return passes, returns

    # ── 결과 출력 ─────────────────────────────────────────────────────────────────

    @staticmethod
    def print_result(label: str, result: dict, days: float) -> int:
        m   = result["metrics"]
        tl  = result["trade_log"]
        tpd = m["n_trades"] / days if days > 0 else 0
        flips = [t for t in tl if t.get("reason") == "reverse_flip"]

        if len(tl) > 5:
            sorted_pnl = sorted(t["pnl"] for t in tl)
            r5 = (sum(t["pnl"] for t in tl) - sum(sorted_pnl[-5:])) * 100
        else:
            r5 = sum(t["pnl"] for t in tl) * 100

        ok_ret = m["total_return"] > 0
        ok_tpd = tpd >= 1.5
        ok_top = r5 > 0
        p = sum([ok_ret, ok_tpd, ok_top])

        print(f"\n{'='*66}")
        print(f"  {label}  [live_trader 완벽 모방 | ML 필터 필수]")
        print(f"{'='*66}")
        print(f"  거래수:     {m['n_trades']}  (TPD: {tpd:.2f})  {'✅' if ok_tpd else '❌'}")
        print(f"  수익률:     {m['total_return']:+.2f}%  {'✅' if ok_ret else '❌'}")
        print(f"  MDD:        {m['mdd']:.1f}%")
        print(f"  WR:         {m['win_rate']:.1f}%")
        print(f"  PF:         {m['profit_factor']:.3f}")
        print(f"  avg_win:    {m['avg_win']:+.4f}%")
        print(f"  avg_loss:   {m['avg_loss']:+.4f}%")
        print(f"  Top-5 제거: {r5:+.2f}%  {'✅' if ok_top else '❌'}")
        print(f"  flip 발동:  {len(flips)}건")
        print(f"  판정:       {p}/3  {'✅ 통과' if p==3 else ('⚠️ 부분' if p>=2 else '❌ 탈락')}")
        # ── 강건 지표 (아웃라이어/꼬리위험) ──
        sh, so = m.get("sharpe", 0.0), m.get("sortino", 0.0)
        t5, p10 = m.get("top5_share", 0.0), m.get("pct_from_top10", 0.0)
        mcl = m.get("max_consec_loss", 0)
        conc_flag = "⚠️ 아웃라이어 의존" if (p10 > 80 or t5 > 0.6) else "✅"
        print(f"  ─ 강건지표 ─")
        print(f"  Sharpe/Sortino(거래당): {sh:+.3f} / {so:+.3f}")
        print(f"  수익집중: 상위10%거래 기여 {p10:.0f}%, 양수 중 top5 비중 {t5*100:.0f}%  {conc_flag}")
        print(f"  최대 연속손실: {mcl}회")
        return p
