"""
temp/scripts/53_param_sweep_post_emergency_sl.py

Emergency SL 변경 후 파라미터 최적화 스윕
- Phase 1: trail_init × trail_tight 그리드 (RSI는 prod 고정)
- Phase 2: ut_rsi_hi × rg_rsi 조합 (trail은 Phase1 최적값 고정)

점수 기준:
  score = log(2026_avg_ret + 1) × hist_pass_ratio × mdd_penalty
  hist_pass_ratio = 통과창수 / 40 (4코인 × 10창)
  mdd_penalty = 1.0 if max_mdd <= 0.07 else 0.7 if <= 0.10 else 0.3

Usage:
  python temp/scripts/53_param_sweep_post_emergency_sl.py --phase 1
  python temp/scripts/53_param_sweep_post_emergency_sl.py --phase 2
  python temp/scripts/53_param_sweep_post_emergency_sl.py --phase all
"""
import sys, argparse, random, math, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
from pathlib import Path
from backtest_antifragile import run_antifragile, add_indicators, load_coin_full, remove_top_n

ROOT = Path(__file__).parent.parent.parent
COINS = ["btc", "eth", "sol", "xrp"]
SEED  = 42
WINDOWS = 10
WINDOW_DAYS = 91

# ── 기준 RSI 파라미터 (prod) ────────────────────────────────────────────────
BASE_RSI = dict(
    dt_rsi_lo=22, dt_rsi_hi=65,
    rg_rsi_lo=25, rg_rsi_hi=75,
    ut_rsi_lo=40, ut_rsi_hi=85,
)

# Phase 1 탐색 범위
TRAIL_INIT_VALS  = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
TRAIL_TIGHT_VALS = [1.5, 2.0, 2.5, 3.0]

# Phase 2 탐색 범위
UT_HI_VALS   = [72, 75, 78, 80, 82, 85]
RG_PAIRS     = [(22,78),(25,75),(28,72),(30,70)]  # (rg_lo, rg_hi)

# Phase 1 최적 trail (Phase 2에서 사용) — Phase1 실행 후 자동 업데이트
BEST_TRAIL = dict(trail_atr_init=1.5, trail_atr_tight=2.0)


# ─────────────────────────────────────────────────────────────────────────────

def _normalize(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df

def load_all():
    data = {}
    for c in COINS:
        data[c] = load_coin_full(c)
    return data

def run_2026(df, coin, cfg):
    start = "2026-01-01"
    sub = df[df.index >= start]
    if len(sub) < 500:
        return None
    r = run_antifragile(sub, **cfg)
    m = r["metrics"]
    days = (sub.index[-1] - sub.index[0]).days
    ret  = m.get("total_return", 0)          # 이미 퍼센트값
    mdd  = m.get("mdd", 100) / 100           # 퍼센트 → 소수 변환
    tpd  = m.get("tpd", 0)
    top5 = remove_top_n(r["trade_log"], 5)   # trade_log에서 직접 계산
    passed = ret > 0 and tpd >= 1.5 and top5 > 0
    return dict(ret=ret, mdd=mdd, tpd=tpd, top5=top5, passed=passed, days=days)

def run_hist(df, coin, cfg):
    hist_start_map = {"btc":"2020-01-01","eth":"2021-04-01","sol":"2021-06-01","xrp":"2020-06-01"}
    hist_start = hist_start_map.get(coin, "2020-01-01")
    sub = df[df.index >= hist_start]
    # index 기반으로 timezone 문제 회피
    min_start = sub.index[0]
    max_start = sub.index[-1] - pd.Timedelta(days=WINDOW_DAYS)
    rng = random.Random(SEED)
    passes = []; rets = []
    for _ in range(WINDOWS):
        days_range = int((max_start - min_start).days)
        if days_range <= 0:
            break
        offset = rng.randint(0, days_range)
        ws = min_start + pd.Timedelta(days=offset)
        we = ws + pd.Timedelta(days=WINDOW_DAYS)
        w  = sub[(sub.index >= ws) & (sub.index < we)]
        if len(w) < 200:
            passes.append(0); rets.append(0); continue
        r  = run_antifragile(w, **cfg)
        m  = r["metrics"]
        ret  = m.get("total_return", 0)          # 이미 퍼센트값
        tpd  = m.get("tpd", 0)
        top5 = remove_top_n(r["trade_log"], 5)   # trade_log에서 직접 계산
        p = int(ret > 0) + int(tpd >= 1.5) + int(top5 > 0)
        passes.append(p); rets.append(ret)
    n3 = sum(1 for p in passes if p == 3)
    avg_ret = np.mean(rets) if rets else 0
    return dict(pass3=n3, total=len(passes), avg_ret=avg_ret)

def score_combo(results_2026, results_hist):
    """4코인 통합 점수 계산."""
    all_rets  = [r["ret"]  for r in results_2026.values() if r]
    all_pass  = [r["passed"] for r in results_2026.values() if r]
    all_mdd   = [r["mdd"]  for r in results_2026.values() if r]
    all_h3    = [r["pass3"] for r in results_hist.values() if r]
    all_htot  = [r["total"] for r in results_hist.values() if r]

    if not all_rets:
        return -999

    avg_ret   = np.mean([math.log(max(r, 1) + 1) for r in all_rets])
    pass_2026 = sum(all_pass) / len(all_pass)
    max_mdd   = max(all_mdd)
    hist_pass = sum(all_h3) / max(sum(all_htot), 1)

    mdd_pen = 1.0 if max_mdd <= 0.07 else (0.7 if max_mdd <= 0.10 else 0.3)
    return avg_ret * pass_2026 * hist_pass * mdd_pen

def run_combo(data, cfg, label):
    res2026 = {}; reshist = {}
    for c in COINS:
        df = data[c]
        res2026[c] = run_2026(df, c, cfg)
        reshist[c] = run_hist(df, c, cfg)
    sc = score_combo(res2026, reshist)

    hist_pass_total = sum(r["pass3"] for r in reshist.values() if r)
    hist_total      = sum(r["total"] for r in reshist.values() if r)
    avg_2026 = np.mean([r["ret"] for r in res2026.values() if r])
    max_mdd  = max((r["mdd"] for r in res2026.values() if r), default=0)
    avg_tpd  = np.mean([r["tpd"] for r in res2026.values() if r])

    print(f"  {label:45}  2026={avg_2026:+8.1f}%  MDD={max_mdd*100:.1f}%  "
          f"TPD={avg_tpd:.1f}  hist={hist_pass_total}/{hist_total}  score={sc:.4f}")
    return sc, dict(label=label, score=sc, avg_2026=avg_2026, max_mdd=max_mdd,
                    avg_tpd=avg_tpd, hist_pass=hist_pass_total, hist_total=hist_total,
                    **{f"ret_{c}": res2026[c]["ret"] if res2026[c] else 0 for c in COINS})

def phase1(data):
    print(f"\n{'='*80}")
    print(f"  PHASE 1: trail_init × trail_tight 그리드 (RSI=prod 고정)")
    print(f"  {len(TRAIL_INIT_VALS)} × {len(TRAIL_TIGHT_VALS)} = {len(TRAIL_INIT_VALS)*len(TRAIL_TIGHT_VALS)} 조합 × 4코인")
    print(f"{'='*80}")

    rows = []
    for ti in TRAIL_INIT_VALS:
        for tt in TRAIL_TIGHT_VALS:
            if tt <= ti:
                continue  # tight은 init보다 커야 함
            cfg = {**BASE_RSI, "trail_atr_init": ti, "trail_atr_tight": tt}
            label = f"trail={ti:.1f}/{tt:.1f}"
            sc, row = run_combo(data, cfg, label)
            row["trail_init"] = ti; row["trail_tight"] = tt
            rows.append(row)

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(f"\n{'─'*80}")
    print("  PHASE 1 TOP 10 결과 (score 내림차순)")
    print(f"{'─'*80}")
    print(f"  {'trail init/tight':20}  {'avg_2026':>10}  {'MDD':>6}  {'hist':>8}  {'score':>8}")
    for _, r in df.head(10).iterrows():
        print(f"  {r['label']:20}  {r['avg_2026']:>+10.1f}%  {r['max_mdd']*100:>5.1f}%"
              f"  {r['hist_pass']}/{r['hist_total']:>3}  {r['score']:>8.4f}")

    best = df.iloc[0]
    print(f"\n  ★ 최적: trail_init={best['trail_init']:.1f}  trail_tight={best['trail_tight']:.1f}")
    return best["trail_init"], best["trail_tight"], df

def phase2(data, trail_init, trail_tight):
    print(f"\n{'='*80}")
    print(f"  PHASE 2: ut_rsi_hi × rg_rsi 그리드 (trail={trail_init:.1f}/{trail_tight:.1f} 고정)")
    print(f"  {len(UT_HI_VALS)} × {len(RG_PAIRS)} = {len(UT_HI_VALS)*len(RG_PAIRS)} 조합 × 4코인")
    print(f"{'='*80}")

    rows = []
    for ut_hi in UT_HI_VALS:
        for (rg_lo, rg_hi) in RG_PAIRS:
            cfg = {
                **BASE_RSI,
                "rg_rsi_lo": rg_lo, "rg_rsi_hi": rg_hi,
                "ut_rsi_hi": ut_hi,
                "trail_atr_init": trail_init, "trail_atr_tight": trail_tight,
            }
            label = f"ut_hi={ut_hi}  rg={rg_lo}/{rg_hi}"
            sc, row = run_combo(data, cfg, label)
            row["ut_hi"] = ut_hi; row["rg_lo"] = rg_lo; row["rg_hi"] = rg_hi
            rows.append(row)

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(f"\n{'─'*80}")
    print("  PHASE 2 TOP 10 결과 (score 내림차순)")
    print(f"{'─'*80}")
    print(f"  {'RSI 설정':30}  {'avg_2026':>10}  {'MDD':>6}  {'hist':>8}  {'score':>8}")
    for _, r in df.head(10).iterrows():
        print(f"  {r['label']:30}  {r['avg_2026']:>+10.1f}%  {r['max_mdd']*100:>5.1f}%"
              f"  {r['hist_pass']}/{r['hist_total']:>3}  {r['score']:>8.4f}")

    best = df.iloc[0]
    print(f"\n  ★ 최적: ut_rsi_hi={best['ut_hi']}  rg_rsi={best['rg_lo']}/{best['rg_hi']}")
    return best, df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["1","2","all"])
    args = parser.parse_args()

    print("데이터 로딩 중...")
    data = load_all()
    print("로딩 완료.\n")

    if args.phase in ("1", "all"):
        best_ti, best_tt, _ = phase1(data)
    else:
        best_ti, best_tt = BEST_TRAIL["trail_atr_init"], BEST_TRAIL["trail_atr_tight"]

    if args.phase in ("2", "all"):
        phase2(data, best_ti, best_tt)

if __name__ == "__main__":
    main()
