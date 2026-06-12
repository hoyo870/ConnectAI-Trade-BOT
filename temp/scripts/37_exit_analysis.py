"""
temp/scripts/37_exit_analysis.py
수익 최적화 케이스 분석: 놓친 수익 패턴 탐색

분석 목표:
1. 실제 paper trades CSV에서 rr/hold_bars 기준으로 미달 수익 패턴 분류
2. 2026 OOS 백테스트로 대안 청산 전략 비교
   - 기준: trail_init=0.5, trail_tight=0.8
   - A: trail_init 확대 (더 긴 줄)
   - B: Partial TP (일정 수익 시 50% 익절 + 나머지 run)
   - C: Staged trailing (초기 타이트 → 피라미딩 후 넓힘)

※ live_trader.py 수정 금지
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path
from hybrid_engine import compute_metrics

ROOT = Path(__file__).parent.parent.parent
TRADING_FEE = 0.0005
SLIPPAGE    = 0.0002
FEE_TOTAL   = TRADING_FEE + SLIPPAGE

# ─────────────────────────────────────────────────────────────────────────────
# 1. Paper Trades CSV 패턴 분석
# ─────────────────────────────────────────────────────────────────────────────

def analyze_paper_trades():
    print("=" * 60)
    print("  PART 1: Paper Trades 패턴 분석 (Jun 3-9)")
    print("=" * 60)

    coins = {"BTC": "paper_trades.csv", "ETH": "paper_trades_eth.csv",
             "SOL": "paper_trades_sol.csv", "XRP": "paper_trades_xrp.csv"}

    for coin, fname in coins.items():
        path = ROOT / "logs" / fname
        if not path.exists():
            print(f"  {coin}: 파일 없음"); continue

        df = pd.read_csv(path)
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
        df["hold_bars"] = pd.to_numeric(df["hold_bars"], errors="coerce")
        df["rr"] = pd.to_numeric(df["rr"], errors="coerce")

        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]

        print(f"\n  [{coin}] 총 {len(df)}거래 | WR={len(wins)/len(df)*100:.1f}%")
        print(f"  {'─'*55}")

        # rr 레벨별 분포 (피라미딩 단계)
        rr_map = {0.10: "rr=0.10 (0피라)", 0.25: "rr=0.25 (1피라)",
                  0.40: "rr=0.40 (2피라)", 0.55: "rr=0.55 (3피라=MAX)"}
        for rr_val, label in rr_map.items():
            sub = df[df["rr"].round(2) == rr_val]
            if len(sub) == 0: continue
            sub_wins = sub[sub["pnl"] > 0]
            avg_pnl = sub["pnl"].mean() * 100
            avg_hold = sub["hold_bars"].mean()
            win_pnl = sub_wins["pnl"].mean() * 100 if len(sub_wins) > 0 else 0
            print(f"  {label}: {len(sub)}건 | WR={len(sub_wins)/len(sub)*100:.0f}% | "
                  f"avgPnL={avg_pnl:+.3f}% | avgHold={avg_hold:.1f}봉 | winAvg={win_pnl:+.3f}%")

        # 케이스 1: 수익권이었다가 소폭 익절 (rr < 0.55 & pnl > 0 & hold_bars > 5)
        case1 = wins[(wins["rr"] < 0.55) & (wins["hold_bars"] > 5)]
        print(f"\n  [케이스A] 피라미딩 미달(rr<0.55) + 5봉 이상 보유 후 소폭 익절: {len(case1)}건")
        if len(case1) > 0:
            print(f"    평균 pnl={case1['pnl'].mean()*100:+.3f}% | 최대={case1['pnl'].max()*100:+.3f}%")
            print(f"    ⚠ 이 거래들은 추세가 있었지만 trail_SL이 너무 빨리 잡은 케이스")

        # 케이스 2: 즉시 손절 (hold_bars <= 2)
        case2 = losses[losses["hold_bars"] <= 2]
        print(f"\n  [케이스B] 2봉 이내 즉시 손절: {len(case2)}건 ({len(case2)/len(df)*100:.0f}%)")
        if len(case2) > 0:
            print(f"    평균 pnl={case2['pnl'].mean()*100:+.3f}% | 총 손실={case2['pnl'].sum()*100:+.3f}%")
            print(f"    ⚠ 초기 trail_stop이 너무 짧아 노이즈에 손절 (trail_init=0.5 ATR)")

        # 케이스 3: 큰 수익 거래 (rr=0.55) 특징
        big_wins = wins[wins["rr"] == 0.55]
        print(f"\n  [케이스C] 최대 피라미딩(rr=0.55) 대형 수익: {len(big_wins)}건")
        if len(big_wins) > 0:
            print(f"    평균 pnl={big_wins['pnl'].mean()*100:+.3f}% | 평균 보유={big_wins['hold_bars'].mean():.0f}봉")
            print(f"    ✅ 이것이 전략의 핵심 - 더 많이 만들어야 함")

        # 케이스 4: 방향별 성과
        if "direction" in df.columns:
            for dir_val, dir_name in [(1, "롱"), (-1, "숏")]:
                sub = df[df["direction"] == dir_val]
                if len(sub) == 0: continue
                sw = sub[sub["pnl"] > 0]
                print(f"  [{dir_name}] {len(sub)}건 | WR={len(sw)/len(sub)*100:.0f}% | "
                      f"avgPnL={sub['pnl'].mean()*100:+.3f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 지표 계산 (backtest_antifragile.py에서 복사)
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]; high = df["high"]; low = df["low"]
    delta = close.diff()
    ag = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["_rsi"] = 100 - 100 / (1 + ag / (al + 1e-9))
    tr = pd.concat([high - low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    df["_atr"] = tr.ewm(span=14, adjust=False).mean()
    cl1h = close.resample("1h").last().ffill()
    ema_1h = cl1h.ewm(span=20, adjust=False).mean()
    df["_trend_up"]   = (cl1h > ema_1h).reindex(df.index, method="ffill").fillna(False).astype(int)
    df["_trend_down"] = (cl1h < ema_1h).reindex(df.index, method="ffill").fillna(False).astype(int)
    return df


def load_coin_data(coin: str) -> pd.DataFrame:
    paths = {
        "BTC": ROOT / "data/raw/BTCUSDT_5m_20260101_20260520.csv",
        "SOL": ROOT / "data/raw/SOLUSDT_5m_20210101_now.csv",
        "XRP": ROOT / "data/raw/XRPUSDT_5m_20200101_now.csv",
    }
    p = paths.get(coin.upper())
    if p and p.exists():
        df = pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp")
        df.columns = [c.lower() for c in df.columns]
        for col in ["open","high","low","close","volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_index()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. 대안 청산 전략 백테스트 엔진 (MFE 트래킹 + Partial TP)
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy(
    df,
    mode,                    # "baseline" | "wide_trail" | "partial_tp" | "staged"
    trail_atr_init  = 0.5,
    trail_atr_tight = 0.8,
    # partial_tp 옵션
    partial_tp_atr  = 1.5,   # X ATR 이익 시 50% 익절
    partial_ratio   = 0.5,
    # staged 옵션
    stage_unlock_atr = 1.0,  # X ATR 달성 시 trail 확장
    stage_wide_mult  = 1.5,
    # 공통
    initial_capital = 10_000.0,
    dt_rsi_lo = 22, dt_rsi_hi = 65,
    ut_rsi_lo = 35, ut_rsi_hi = 78,
    leverage = 3,
    rr_base = 0.10, rr_add = 0.15,
    add_levels = 3, atr_add_step = 0.5,
    max_hold_bars = 288,
):
    df = df.reset_index(drop=True)
    df.dropna(subset=["_rsi","_atr"], inplace=True)
    df = df.reset_index(drop=True)

    capital = initial_capital
    peak_cap = initial_capital
    pos = 0
    entry_price = 0.0
    current_rr = 0.0
    add_count = 0
    trail_sl = 0.0
    peak_price = 0.0
    entry_bar = 0
    partial_done = False   # partial TP 여부
    stage_widened = False  # staged trail 확장 여부

    equity_curve = [capital]
    trade_log = []

    for idx in range(1, len(df)):
        row   = df.iloc[idx]
        price = float(row["close"])
        high  = float(row["high"])
        low_p = float(row["low"])
        rsi   = float(row["_rsi"])
        atr   = float(row["_atr"])
        tup   = int(row.get("_trend_up", 0))
        tdn   = int(row.get("_trend_down", 0))

        rsi_lo = dt_rsi_lo if tdn else (ut_rsi_lo if tup else 30)
        rsi_hi = dt_rsi_hi if tdn else (ut_rsi_hi if tup else 70)

        if pos != 0:
            unr    = pos * (price - entry_price) / (entry_price + 1e-9)
            equity = capital * (1 + unr * leverage * current_rr)
        else:
            equity = capital
        peak_cap = max(peak_cap, equity)

        if pos != 0:
            hold = idx - entry_bar
            favorable = pos * (price - entry_price) / (atr + 1e-9)

            # ── 청산 모드별 로직 ───────────────────────────────────────────
            if mode == "baseline":
                trail_mult = trail_atr_tight if add_count > 0 else trail_atr_init
                if pos == 1:
                    peak_price = max(peak_price, price)
                    trail_sl   = max(trail_sl, peak_price - trail_mult * atr)
                    hit_stop   = price <= trail_sl
                else:
                    peak_price = min(peak_price, price)
                    trail_sl   = min(trail_sl, peak_price + trail_mult * atr)
                    hit_stop   = price >= trail_sl

            elif mode == "wide_trail":
                # trail_init/tight 파라미터로 전달된 값 사용 (더 넓은 줄)
                trail_mult = trail_atr_tight if add_count > 0 else trail_atr_init
                if pos == 1:
                    peak_price = max(peak_price, price)
                    trail_sl   = max(trail_sl, peak_price - trail_mult * atr)
                    hit_stop   = price <= trail_sl
                else:
                    peak_price = min(peak_price, price)
                    trail_sl   = min(trail_sl, peak_price + trail_mult * atr)
                    hit_stop   = price >= trail_sl

            elif mode == "partial_tp":
                # 기본 trailing
                trail_mult = trail_atr_tight if add_count > 0 else trail_atr_init
                if pos == 1:
                    peak_price = max(peak_price, price)
                    trail_sl   = max(trail_sl, peak_price - trail_mult * atr)
                    hit_stop   = price <= trail_sl
                else:
                    peak_price = min(peak_price, price)
                    trail_sl   = min(trail_sl, peak_price + trail_mult * atr)
                    hit_stop   = price >= trail_sl

                # Partial TP: favorable >= partial_tp_atr ATR 이익 시 절반 익절
                if not partial_done and favorable >= partial_tp_atr:
                    partial_done = True
                    # 50% 포지션 익절
                    cp  = price - FEE_TOTAL * price * pos
                    raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                    pnl_partial = max(raw * leverage * (current_rr * partial_ratio),
                                      -(current_rr * partial_ratio))
                    capital *= (1 + pnl_partial)
                    current_rr *= (1 - partial_ratio)  # 남은 포지션
                    # 남은 포지션은 trail_sl을 진입가로 올림 (BEP 보호)
                    if pos == 1:
                        trail_sl = max(trail_sl, entry_price)
                    else:
                        trail_sl = min(trail_sl, entry_price)
                    trade_log.append({
                        "pnl": pnl_partial, "hold_steps": hold,
                        "rr": current_rr * partial_ratio, "forced": False,
                        "direction": pos, "partial": True
                    })

            elif mode == "staged":
                # 초기: 타이트(trail_atr_init), 이익 달성 후: 넓힘(stage_wide_mult)
                if not stage_widened and favorable >= stage_unlock_atr:
                    stage_widened = True

                trail_mult = stage_wide_mult if stage_widened else trail_atr_init
                if pos == 1:
                    peak_price = max(peak_price, price)
                    trail_sl   = max(trail_sl, peak_price - trail_mult * atr)
                    hit_stop   = price <= trail_sl
                else:
                    peak_price = min(peak_price, price)
                    trail_sl   = min(trail_sl, peak_price + trail_mult * atr)
                    hit_stop   = price >= trail_sl

            if hit_stop or hold >= max_hold_bars:
                cp  = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({
                    "pnl": pnl, "hold_steps": hold, "rr": current_rr,
                    "forced": False, "direction": pos, "partial": False
                })
                pos = 0; add_count = 0; current_rr = 0.0
                partial_done = False; stage_widened = False
            else:
                # 피라미딩
                next_add_level = (add_count + 1) * atr_add_step
                if add_count < add_levels and favorable >= next_add_level:
                    current_rr += rr_add; add_count += 1
                    if pos == 1: trail_sl = max(trail_sl, price - trail_atr_tight * atr)
                    else:        trail_sl = min(trail_sl, price + trail_atr_tight * atr)

        # 신규 진입
        if pos == 0:
            long_ok  = rsi <= rsi_lo
            short_ok = rsi >= rsi_hi
            if long_ok:
                ep = price * (1 + FEE_TOTAL)
                entry_price = ep; current_rr = rr_base; add_count = 0
                trail_sl    = ep - trail_atr_init * atr; peak_price = ep
                pos = 1; entry_bar = idx
                partial_done = False; stage_widened = False
            elif short_ok:
                ep = price * (1 - FEE_TOTAL)
                entry_price = ep; current_rr = rr_base; add_count = 0
                trail_sl    = ep + trail_atr_init * atr; peak_price = ep
                pos = -1; entry_bar = idx
                partial_done = False; stage_widened = False

        equity_curve.append(capital)

    m = compute_metrics(equity_curve, trade_log)
    real_trades = [t for t in trade_log if not t.get("partial", False)]
    if real_trades:
        days = len(df) / 288
        m["tpd"] = round(len(real_trades) / days, 2)
        wins  = [t["pnl"] for t in real_trades if t["pnl"] > 0]
        losss = [t["pnl"] for t in real_trades if t["pnl"] < 0]
        if wins:  m["avg_win"]  = round(float(np.mean(wins)) * 100, 4)
        if losss: m["avg_loss"] = round(float(np.mean(losss)) * 100, 4)
        if wins and losss:
            m["pf_ratio"] = round(abs(np.mean(wins) / np.mean(losss)), 3)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# 4. 코인별 전략 비교 실행
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = [
    {"name": "기준 (trail=0.5/0.8)",    "mode": "baseline", "trail_atr_init": 0.5, "trail_atr_tight": 0.8},
    {"name": "넓은줄 (trail=0.7/1.0)",  "mode": "wide_trail","trail_atr_init": 0.7, "trail_atr_tight": 1.0},
    {"name": "넓은줄 (trail=1.0/1.5)",  "mode": "wide_trail","trail_atr_init": 1.0, "trail_atr_tight": 1.5},
    {"name": "넓은줄 (trail=1.5/2.0)",  "mode": "wide_trail","trail_atr_init": 1.5, "trail_atr_tight": 2.0},
    {"name": "분할익절 (1.5ATR→50%)",   "mode": "partial_tp","trail_atr_init": 0.5, "trail_atr_tight": 0.8,
     "partial_tp_atr": 1.5, "partial_ratio": 0.5},
    {"name": "분할익절 (1.0ATR→50%)",   "mode": "partial_tp","trail_atr_init": 0.5, "trail_atr_tight": 0.8,
     "partial_tp_atr": 1.0, "partial_ratio": 0.5},
    {"name": "Staged (0.3→1.5 @1ATR)", "mode": "staged",    "trail_atr_init": 0.3, "trail_atr_tight": 0.8,
     "stage_unlock_atr": 1.0, "stage_wide_mult": 1.5},
    {"name": "Staged (0.3→2.0 @1ATR)", "mode": "staged",    "trail_atr_init": 0.3, "trail_atr_tight": 0.8,
     "stage_unlock_atr": 1.0, "stage_wide_mult": 2.0},
]

OOS_START = "2026-01-01"
OOS_END   = "2026-05-20"
COINS     = ["BTC", "SOL", "XRP"]


def run_comparison():
    print("\n\n" + "=" * 70)
    print("  PART 2: 대안 청산 전략 비교 (OOS: 2026-01-01 ~ 2026-05-20)")
    print("=" * 70)

    for coin in COINS:
        raw = load_coin_data(coin)
        if raw is None:
            print(f"  {coin}: 데이터 없음"); continue

        df = add_indicators(raw)
        # OOS 구간만
        mask = (df.index >= OOS_START) & (df.index <= OOS_END)
        df_oos = df[mask].copy()
        if len(df_oos) < 500:
            print(f"  {coin}: OOS 데이터 부족"); continue

        print(f"\n  [{coin}] OOS {OOS_START}~{OOS_END}  ({len(df_oos)}봉)")
        print(f"  {'전략':<30} {'수익률':>8} {'TPD':>5} {'PF':>7} {'MDD':>6} {'avgWin':>8} {'avgLoss':>9}")
        print(f"  {'─'*70}")

        for s in STRATEGIES:
            kwargs = {k: v for k, v in s.items() if k not in ("name", "mode")}
            m = run_strategy(df_oos, mode=s["mode"], **kwargs)
            ret  = m.get("total_return", 0) * 100
            tpd  = m.get("tpd", 0)
            pf   = m.get("pf_ratio", 0)
            mdd  = m.get("max_dd", 0) * 100
            aw   = m.get("avg_win", 0)
            al   = m.get("avg_loss", 0)
            flag = " ✅" if (ret > 0 and pf > 5 and mdd < 5) else ""
            print(f"  {s['name']:<30} {ret:>+7.1f}% {tpd:>5.2f} {pf:>7.2f} {mdd:>5.2f}% {aw:>+7.3f}% {al:>+8.3f}%{flag}")


if __name__ == "__main__":
    import os
    os.chdir(ROOT)

    analyze_paper_trades()
    run_comparison()

    print("\n\n" + "=" * 70)
    print("  분석 완료")
    print("  핵심 케이스 요약:")
    print("  A. rr<0.55 + 5봉이상 소폭익절 → trail_init 확대로 더 달릴 수 있음")
    print("  B. 2봉이내 즉시손절 → trail_init=0.3 staged로 노이즈 감소")
    print("  C. 분할익절은 fat-tail 전략에서 대형 수익을 cap함 → 신중")
    print("=" * 70)
