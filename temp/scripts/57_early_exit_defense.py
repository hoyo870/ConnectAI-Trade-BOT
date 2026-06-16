"""
조기 청산 방어 백테스트
trail_sl 청산 + 같은 방향 시그널 + N봉 이내 → 청산 대신 피라미딩

비교:
  - baseline: 기존 로직 (N=0, 항상 청산)
  - N=3  (15분 이내)
  - N=6  (30분 이내)
  - N=12 (1시간 이내)

Usage:
  python temp/scripts/57_early_exit_defense.py
"""
import sys, random
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from hybrid_engine import compute_metrics
from backtest_antifragile import add_indicators, load_coin_full, remove_top_n, _normalize_index
from config.af_params import get_preset, FEE_TOTAL

LEVERAGE   = 7
SEED       = 42
WINDOWS    = 10
WINDOW_D   = 91
OOS_START  = "2026-01-01"
OOS_END    = "2026-06-01"
JUNE_START = "2026-06-01"
JUNE_END   = "2026-06-17"
COINS      = ["btc", "eth", "sol", "xrp"]

CFG = get_preset("prod")


def run_with_defense(df, leverage=7, early_exit_bars=0, max_dd_cb=0.99, **cfg):
    """
    early_exit_bars=0 → 기존 로직 (방어 없음)
    early_exit_bars=N → N봉 이내 trail_sl 청산 + 같은방향 시그널 → 피라미딩
    """
    dt_rsi_lo       = cfg.get("dt_rsi_lo", 28)
    dt_rsi_hi       = cfg.get("dt_rsi_hi", 60)
    rg_rsi_lo       = cfg.get("rg_rsi_lo", 25)
    rg_rsi_hi       = cfg.get("rg_rsi_hi", 75)
    ut_rsi_lo       = cfg.get("ut_rsi_lo", 42)
    ut_rsi_hi       = cfg.get("ut_rsi_hi", 75)
    rr_base         = cfg.get("rr_base", 0.20)
    rr_add          = cfg.get("rr_add", 0.10)
    add_levels      = cfg.get("add_levels", 3)
    atr_add_step    = cfg.get("atr_add_step", 0.5)
    trail_atr_init  = cfg.get("trail_atr_init", 1.8)
    trail_atr_tight = cfg.get("trail_atr_tight", 2.0)
    max_hold_bars   = cfg.get("max_hold_bars", 288)

    df = add_indicators(df.copy())
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    df = df.reset_index(drop=True)

    capital      = 10_000.0
    peak_cap     = 10_000.0
    pos          = 0
    entry_price  = 0.0
    entry_atr    = 0.0
    current_rr   = 0.0
    add_count    = 0
    trail_sl     = 0.0
    peak_price   = 0.0
    entry_bar    = 0
    cooling_left = 0
    cb_triggers  = 0
    defense_count = 0  # 방어 발동 횟수 추적

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

        # 시그널 사전 계산 (포지션 보유 중에도 필요)
        atr_ok    = atr >= price * 0.0015
        sig_long  = (rsi <= rsi_lo) and atr_ok
        sig_short = (rsi >= rsi_hi) and atr_ok

        # equity & CB
        if pos != 0:
            unr    = pos * (price - entry_price) / (entry_price + 1e-9)
            equity = capital * (1 + unr * leverage * current_rr)
        else:
            equity = capital
        peak_cap = max(peak_cap, equity)
        dd = (peak_cap - equity) / (peak_cap + 1e-9)

        if dd > max_dd_cb and cooling_left == 0:
            cooling_left = 20; cb_triggers += 1
            if pos != 0:
                cp  = price - FEE_TOTAL * price * pos
                raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * leverage * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_bar,
                                   "entry_bar": entry_bar, "exit_bar": idx,
                                   "lev": leverage, "rr": current_rr,
                                   "forced": True, "direction": pos, "defense": False})
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
                                   "lev": leverage, "rr": current_rr,
                                   "forced": True, "direction": pos, "defense": False})
                pos = 0
            equity_curve.append(capital); continue

        # 포지션 관리
        if pos != 0:
            hold = idx - entry_bar
            effective_atr = max(atr, entry_atr * 0.6)
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
                # ── 조기 청산 방어 조건 ──────────────────────────────────────
                same_dir = (pos == 1 and sig_long) or (pos == -1 and sig_short)
                can_defend = (
                    early_exit_bars > 0          # 방어 모드 활성화
                    and hit_stop                 # trail_sl 청산 (max_hold 아님)
                    and hold <= early_exit_bars  # N봉 이내
                    and same_dir                 # 같은 방향 시그널
                    and add_count < add_levels   # 피라미딩 여유 있음
                )
                if can_defend:
                    # 청산 대신 피라미딩 추가 + trail_sl 현재가 기준 재설정
                    current_rr += rr_add
                    add_count  += 1
                    defense_count += 1
                    if pos == 1:
                        trail_sl = price - trail_atr_tight * effective_atr
                    else:
                        trail_sl = price + trail_atr_tight * effective_atr
                else:
                    # 정상 청산
                    cp  = price - FEE_TOTAL * price * pos
                    raw = pos * (cp - entry_price) / (entry_price + 1e-9)
                    pnl = max(raw * leverage * current_rr, -current_rr)
                    capital *= (1 + pnl)
                    trade_log.append({"pnl": pnl, "hold_steps": hold,
                                       "entry_bar": entry_bar, "exit_bar": idx,
                                       "lev": leverage, "rr": current_rr,
                                       "forced": False, "direction": pos, "defense": False})
                    pos = 0; add_count = 0; current_rr = 0.0
            else:
                # 피라미딩 (유리방향 추가)
                favorable_move = pos * (price - entry_price) / (atr + 1e-9)
                next_add_level = (add_count + 1) * atr_add_step
                if add_count < add_levels and favorable_move >= next_add_level:
                    current_rr += rr_add; add_count += 1
                    if pos == 1: trail_sl = max(trail_sl, price - trail_atr_tight * effective_atr)
                    else:         trail_sl = min(trail_sl, price + trail_atr_tight * effective_atr)

        # 신규 진입
        if pos == 0:
            if sig_long:
                ep          = price * (1 + FEE_TOTAL)
                entry_price = ep; entry_atr = atr; current_rr = rr_base; add_count = 0
                trail_sl    = ep - trail_atr_init * atr; peak_price = ep
                pos = 1; entry_bar = idx
            elif sig_short:
                ep          = price * (1 - FEE_TOTAL)
                entry_price = ep; entry_atr = atr; current_rr = rr_base; add_count = 0
                trail_sl    = ep + trail_atr_init * atr; peak_price = ep
                pos = -1; entry_bar = idx

        equity_curve.append(capital)

    m = compute_metrics(equity_curve, trade_log)
    if trade_log:
        days_total = len(df) / 288
        m["tpd"]      = round(len(trade_log) / days_total, 2)
        m["long_cnt"] = sum(1 for t in trade_log if t.get("direction") ==  1)
        m["short_cnt"]= sum(1 for t in trade_log if t.get("direction") == -1)
        wins  = [t["pnl"] for t in trade_log if t["pnl"] > 0]
        losss = [t["pnl"] for t in trade_log if t["pnl"] < 0]
        if wins:  m["avg_win"]  = round(float(np.mean(wins))  * 100, 4)
        if losss: m["avg_loss"] = round(float(np.mean(losss)) * 100, 4)
    m["defense_count"] = defense_count
    return {"metrics": m, "trade_log": trade_log}


def run_period(df, start, end, n):
    sub = df[(df.index >= start) & (df.index < end)]
    if len(sub) < 100:
        return None
    return run_with_defense(sub, leverage=LEVERAGE, early_exit_bars=n, **CFG)


def run_hist(df, n):
    rng = random.Random(SEED)
    np.random.seed(SEED)
    total_days = (df.index[-1] - df.index[0]).days
    max_start  = total_days - WINDOW_D
    passes, rets, avg_ts = [], [], []
    for _ in range(WINDOWS):
        offset = rng.randint(0, max_start)
        s = df.index[0] + pd.Timedelta(days=offset)
        e = s + pd.Timedelta(days=WINDOW_D)
        sub = df[(df.index >= s) & (df.index < e)]
        if len(sub) < 500:
            continue
        r   = run_with_defense(sub, leverage=LEVERAGE, early_exit_bars=n, **CFG)
        m   = r["metrics"]
        tl  = r["trade_log"]
        ret = m["total_return"]
        r5  = remove_top_n(tl, 5)
        tpd = m.get("tpd", 0)
        ok  = sum([ret > 0, tpd >= 1.5, r5 > 0])
        passes.append(ok == 3)
        rets.append(ret)
        avg_ts.append(ret / len(tl) if tl else 0)
    n_pass = sum(passes)
    mark   = "✅" if n_pass >= 7 else ("⚠️" if n_pass >= 5 else "❌")
    return n_pass, np.mean(rets) if rets else 0, np.mean(avg_ts) if avg_ts else 0, mark


# ─── 실행 ───
print("데이터 로딩...")
dfs = {}
for coin in COINS:
    dfs[coin] = _normalize_index(load_coin_full(coin))
    print(f"  {coin.upper()}: {len(dfs[coin]):,}행")
print()

N_VALUES = [0, 3, 6, 12]
N_LABELS = {0: "기존(N=0)", 3: "N=3(15분)", 6: "N=6(30분)", 12: "N=12(1h)"}

for period_label, start, end in [
    ("2026 OOS (01-01~05-31)", OOS_START, OOS_END),
    ("2026-06 (06-01~06-16)",  JUNE_START, JUNE_END),
]:
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    print("=" * 100)
    print(f"  [{period_label}]")
    print("=" * 100)
    print(f"  {'설정':<12}  {'코인':<5}  {'거래수':>6}  {'WR':>6}  {'수익률':>9}  "
          f"{'MDD':>6}  {'TPD':>5}  {'avg/건':>7}  {'Top5':>7}  {'방어':>5}  판정")
    print("  " + "─" * 92)

    for n in N_VALUES:
        for coin in COINS:
            r = run_period(dfs[coin], start, end, n)
            if r is None:
                continue
            m  = r["metrics"]
            tl = r["trade_log"]
            nt = m["n_trades"]
            wr = m["win_rate"]
            ret = m["total_return"]
            mdd = m["mdd"]
            tpd = round(nt / max(days, 1), 1)
            avg_t = ret / nt if nt > 0 else 0
            r5  = remove_top_n(tl, 5)
            def_cnt = m.get("defense_count", 0)
            ok  = [ret > 0, tpd >= 1.5, r5 > 0]
            mark = "✅" if all(ok) else ("⚠️" if sum(ok) >= 2 else "❌")
            print(f"  {N_LABELS[n]:<12}  {coin.upper():<5}  {nt:>6}  {wr:>5.1f}%  {ret:>+8.1f}%  "
                  f"{mdd:>5.1f}%  {tpd:>5.1f}  {avg_t:>+6.3f}%  {r5:>+6.1f}%  {def_cnt:>5}  {mark}")
        print("  " + "─" * 92)
    print()

print("=" * 100)
print(f"  [hist: {WINDOW_D}일 × {WINDOWS}창, seed={SEED}]")
print("=" * 100)
print(f"  {'설정':<12}  {'코인':<5}  {'통과':>7}  {'avg수익':>9}  {'avg/건':>8}")
print("  " + "─" * 52)
for n in N_VALUES:
    for coin in COINS:
        n_pass, avg_ret, avg_t, mark = run_hist(dfs[coin], n)
        print(f"  {N_LABELS[n]:<12}  {coin.upper():<5}  {n_pass}/{WINDOWS} {mark}  "
              f"{avg_ret:>+8.1f}%  {avg_t:>+7.3f}%")
    print("  " + "─" * 52)
