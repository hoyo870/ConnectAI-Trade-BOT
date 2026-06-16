"""
temp/scripts/54_rsi_sweep_full.py

전축 RSI 스윕 + 프리셋별 파라미터 최적화
- Phase 2a: dt_rsi_lo × dt_rsi_hi 독립 스윕
- Phase 2b: ut_rsi_lo × ut_rsi_hi 독립 스윕
- Phase 2c: rg_rsi_lo × rg_rsi_hi 독립 스윕
- Phase 3:  프리셋(prod/stable/aggressive/conservative) × 특성 trail 검증

Usage:
  python scripts/sweeps/rsi_sweep.py --phase all
  python scripts/sweeps/rsi_sweep.py --phase 2a
  python scripts/sweeps/rsi_sweep.py --phase 3
"""
import sys, argparse, random, math, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from backtest_antifragile import run_antifragile, load_coin_full, remove_top_n

COINS   = ["btc", "eth", "sol", "xrp"]
SEED    = 42
WINDOWS = 10
WINDOW_DAYS = 91

# Phase 1 최적 trail
BEST_TRAIL = dict(trail_atr_init=1.8, trail_atr_tight=2.0)

# Phase 1 이전 현재 prod RSI 기준값
BASE_PROD = dict(
    dt_rsi_lo=22, dt_rsi_hi=65,
    rg_rsi_lo=25, rg_rsi_hi=75,
    ut_rsi_lo=40, ut_rsi_hi=85,
)

# Phase 2a: dt_rsi 탐색 범위
DT_LO_VALS = [18, 20, 22, 25, 28]
DT_HI_VALS = [60, 62, 65, 68, 70]

# Phase 2b: ut_rsi 탐색 범위
UT_LO_VALS = [35, 38, 40, 42, 45]
UT_HI_VALS = [72, 75, 78, 80, 82, 85]

# Phase 2c: rg_rsi 탐색 범위
RG_LO_VALS = [20, 22, 25, 28, 30]
RG_HI_VALS = [68, 70, 72, 75, 78]

# Phase 3: 프리셋별 trail 특성
PRESET_TRAILS = {
    "prod":         dict(trail_atr_init=1.8, trail_atr_tight=2.0),
    "stable":       dict(trail_atr_init=1.5, trail_atr_tight=2.0),
    "aggressive":   dict(trail_atr_init=0.8, trail_atr_tight=1.5),
    "conservative": dict(trail_atr_init=2.0, trail_atr_tight=2.5),
}

# Phase 3: 프리셋별 RSI 성향 조정 범위
PRESET_RSI_VARIANTS = {
    "prod": [
        # 스윕 최적값 적용 (Phase 2 완료 후 자동 업데이트)
    ],
    "stable": [
        # 안정: 더 극단적인 조건 (높은 lo, 낮은 hi → 덜 진입)
        dict(dt_rsi_lo=25, dt_rsi_hi=62, ut_rsi_lo=42, ut_rsi_hi=80, rg_rsi_lo=25, rg_rsi_hi=75),
        dict(dt_rsi_lo=25, dt_rsi_hi=62, ut_rsi_lo=42, ut_rsi_hi=78, rg_rsi_lo=25, rg_rsi_hi=75),
        dict(dt_rsi_lo=25, dt_rsi_hi=60, ut_rsi_lo=42, ut_rsi_hi=78, rg_rsi_lo=28, rg_rsi_hi=72),
    ],
    "aggressive": [
        # 공격: 더 완화된 조건 (낮은 lo, 높은 hi → 더 많이 진입)
        dict(dt_rsi_lo=20, dt_rsi_hi=68, ut_rsi_lo=38, ut_rsi_hi=82, rg_rsi_lo=22, rg_rsi_hi=78),
        dict(dt_rsi_lo=18, dt_rsi_hi=68, ut_rsi_lo=35, ut_rsi_hi=82, rg_rsi_lo=22, rg_rsi_hi=78),
        dict(dt_rsi_lo=20, dt_rsi_hi=70, ut_rsi_lo=38, ut_rsi_hi=80, rg_rsi_lo=20, rg_rsi_hi=80),
    ],
    "conservative": [
        # 보수: 더 극단적 + 넓은 trail (긴 보유)
        dict(dt_rsi_lo=28, dt_rsi_hi=60, ut_rsi_lo=45, ut_rsi_hi=75, rg_rsi_lo=28, rg_rsi_hi=72),
        dict(dt_rsi_lo=25, dt_rsi_hi=60, ut_rsi_lo=42, ut_rsi_hi=75, rg_rsi_lo=25, rg_rsi_hi=75),
        dict(dt_rsi_lo=28, dt_rsi_hi=62, ut_rsi_lo=45, ut_rsi_hi=78, rg_rsi_lo=28, rg_rsi_hi=72),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────

def load_all():
    data = {}
    for c in COINS:
        data[c] = load_coin_full(c)
    return data


def run_2026(df, cfg):
    sub = df[df.index >= "2026-01-01"]
    if len(sub) < 500:
        return None
    r   = run_antifragile(sub, **cfg)
    m   = r["metrics"]
    ret  = m.get("total_return", 0)
    mdd  = m.get("mdd", 100) / 100
    tpd  = m.get("tpd", 0)
    top5 = remove_top_n(r["trade_log"], 5)
    passed = ret > 0 and tpd >= 1.5 and top5 > 0
    return dict(ret=ret, mdd=mdd, tpd=tpd, top5=top5, passed=passed)


def run_hist(df, coin, cfg):
    hist_start_map = {
        "btc": "2020-01-01", "eth": "2021-04-01",
        "sol": "2021-06-01", "xrp": "2020-06-01",
    }
    hist_start = hist_start_map.get(coin, "2020-01-01")
    sub = df[df.index >= hist_start]
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
        ret  = m.get("total_return", 0)
        tpd  = m.get("tpd", 0)
        top5 = remove_top_n(r["trade_log"], 5)
        p = int(ret > 0) + int(tpd >= 1.5) + int(top5 > 0)
        passes.append(p); rets.append(ret)
    n3 = sum(1 for p in passes if p == 3)
    return dict(pass3=n3, total=len(passes))


def score_combo(results_2026, results_hist):
    all_rets  = [r["ret"]    for r in results_2026.values() if r]
    all_pass  = [r["passed"] for r in results_2026.values() if r]
    all_mdd   = [r["mdd"]   for r in results_2026.values() if r]
    all_h3    = [r["pass3"]  for r in results_hist.values()  if r]
    all_htot  = [r["total"]  for r in results_hist.values()  if r]
    if not all_rets:
        return -999
    avg_ret   = np.mean([math.log(max(r, 1) + 1) for r in all_rets])
    pass_2026 = sum(all_pass) / len(all_pass)
    max_mdd   = max(all_mdd)
    hist_pass = sum(all_h3) / max(sum(all_htot), 1)
    mdd_pen   = 1.0 if max_mdd <= 0.07 else (0.7 if max_mdd <= 0.10 else 0.3)
    return avg_ret * pass_2026 * hist_pass * mdd_pen


def run_combo(data, cfg, label):
    res2026 = {}; reshist = {}
    for c in COINS:
        df = data[c]
        res2026[c] = run_2026(df, cfg)
        reshist[c] = run_hist(df, c, cfg)
    sc = score_combo(res2026, reshist)
    avg_2026 = np.mean([r["ret"]  for r in res2026.values() if r])
    max_mdd  = max((r["mdd"]      for r in res2026.values() if r), default=0)
    avg_tpd  = np.mean([r["tpd"]  for r in res2026.values() if r])
    h_pass   = sum(r["pass3"] for r in reshist.values() if r)
    h_total  = sum(r["total"] for r in reshist.values() if r)
    print(f"  {label:50}  2026={avg_2026:+8.1f}%  MDD={max_mdd*100:.1f}%"
          f"  TPD={avg_tpd:.1f}  hist={h_pass}/{h_total}  score={sc:.4f}")
    return sc, dict(label=label, score=sc, avg_2026=avg_2026, max_mdd=max_mdd,
                    avg_tpd=avg_tpd, hist_pass=h_pass, hist_total=h_total)


def print_top(df_rows, n=10, col_label="설정"):
    df = pd.DataFrame(df_rows).sort_values("score", ascending=False)
    print(f"\n{'─'*80}")
    print(f"  TOP {n} (score 내림차순)")
    print(f"{'─'*80}")
    print(f"  {col_label:40}  {'avg_2026':>10}  {'MDD':>6}  {'hist':>8}  {'score':>8}")
    for _, r in df.head(n).iterrows():
        print(f"  {r['label']:40}  {r['avg_2026']:>+10.1f}%  {r['max_mdd']*100:>5.1f}%"
              f"  {r['hist_pass']}/{r['hist_total']:>3}  {r['score']:>8.4f}")
    return df.iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2a: dt_rsi_lo × dt_rsi_hi
# ─────────────────────────────────────────────────────────────────────────────

def phase2a(data, base_rsi=None, trail=None):
    if base_rsi is None: base_rsi = BASE_PROD
    if trail    is None: trail    = BEST_TRAIL
    print(f"\n{'='*80}")
    print(f"  PHASE 2a: dt_rsi_lo × dt_rsi_hi  ({len(DT_LO_VALS)}×{len(DT_HI_VALS)}={len(DT_LO_VALS)*len(DT_HI_VALS)} 조합)")
    print(f"  trail={trail['trail_atr_init']}/{trail['trail_atr_tight']}  "
          f"rg={base_rsi['rg_rsi_lo']}/{base_rsi['rg_rsi_hi']}  "
          f"ut={base_rsi['ut_rsi_lo']}/{base_rsi['ut_rsi_hi']} (고정)")
    print(f"{'='*80}")
    rows = []
    for lo in DT_LO_VALS:
        for hi in DT_HI_VALS:
            if lo >= hi:
                continue
            cfg = {**base_rsi, **trail, "dt_rsi_lo": lo, "dt_rsi_hi": hi}
            label = f"dt_lo={lo}  dt_hi={hi}"
            sc, row = run_combo(data, cfg, label)
            row.update(dt_lo=lo, dt_hi=hi)
            rows.append(row)
    best = print_top(rows, col_label="dt_rsi 설정")
    print(f"\n  ★ Phase 2a 최적: dt_rsi_lo={best['dt_lo']}  dt_rsi_hi={best['dt_hi']}")
    return int(best["dt_lo"]), int(best["dt_hi"]), rows


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2b: ut_rsi_lo × ut_rsi_hi
# ─────────────────────────────────────────────────────────────────────────────

def phase2b(data, base_rsi=None, trail=None):
    if base_rsi is None: base_rsi = BASE_PROD
    if trail    is None: trail    = BEST_TRAIL
    print(f"\n{'='*80}")
    print(f"  PHASE 2b: ut_rsi_lo × ut_rsi_hi  ({len(UT_LO_VALS)}×{len(UT_HI_VALS)}={len(UT_LO_VALS)*len(UT_HI_VALS)} 조합)")
    print(f"  trail={trail['trail_atr_init']}/{trail['trail_atr_tight']}  "
          f"dt={base_rsi['dt_rsi_lo']}/{base_rsi['dt_rsi_hi']}  "
          f"rg={base_rsi['rg_rsi_lo']}/{base_rsi['rg_rsi_hi']} (고정)")
    print(f"{'='*80}")
    rows = []
    for lo in UT_LO_VALS:
        for hi in UT_HI_VALS:
            if lo >= hi:
                continue
            cfg = {**base_rsi, **trail, "ut_rsi_lo": lo, "ut_rsi_hi": hi}
            label = f"ut_lo={lo}  ut_hi={hi}"
            sc, row = run_combo(data, cfg, label)
            row.update(ut_lo=lo, ut_hi=hi)
            rows.append(row)
    best = print_top(rows, col_label="ut_rsi 설정")
    print(f"\n  ★ Phase 2b 최적: ut_rsi_lo={best['ut_lo']}  ut_rsi_hi={best['ut_hi']}")
    return int(best["ut_lo"]), int(best["ut_hi"]), rows


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2c: rg_rsi_lo × rg_rsi_hi
# ─────────────────────────────────────────────────────────────────────────────

def phase2c(data, base_rsi=None, trail=None):
    if base_rsi is None: base_rsi = BASE_PROD
    if trail    is None: trail    = BEST_TRAIL
    print(f"\n{'='*80}")
    print(f"  PHASE 2c: rg_rsi_lo × rg_rsi_hi  ({len(RG_LO_VALS)}×{len(RG_HI_VALS)}={len(RG_LO_VALS)*len(RG_HI_VALS)} 조합)")
    print(f"  trail={trail['trail_atr_init']}/{trail['trail_atr_tight']}  "
          f"dt={base_rsi['dt_rsi_lo']}/{base_rsi['dt_rsi_hi']}  "
          f"ut={base_rsi['ut_rsi_lo']}/{base_rsi['ut_rsi_hi']} (고정)")
    print(f"{'='*80}")
    rows = []
    for lo in RG_LO_VALS:
        for hi in RG_HI_VALS:
            if lo >= hi:
                continue
            cfg = {**base_rsi, **trail, "rg_rsi_lo": lo, "rg_rsi_hi": hi}
            label = f"rg_lo={lo}  rg_hi={hi}"
            sc, row = run_combo(data, cfg, label)
            row.update(rg_lo=lo, rg_hi=hi)
            rows.append(row)
    best = print_top(rows, col_label="rg_rsi 설정")
    print(f"\n  ★ Phase 2c 최적: rg_rsi_lo={best['rg_lo']}  rg_rsi_hi={best['rg_hi']}")
    return int(best["rg_lo"]), int(best["rg_hi"]), rows


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: 프리셋별 검증
# ─────────────────────────────────────────────────────────────────────────────

def phase3(data, best_rsi):
    """
    best_rsi: Phase 2a+2b+2c에서 찾은 최적 RSI 딕셔너리
    각 프리셋 특성에 맞는 RSI 변형 + trail 조합을 검증
    """
    print(f"\n{'='*80}")
    print(f"  PHASE 3: 프리셋별 파라미터 검증")
    print(f"  최적 RSI 기준: dt={best_rsi['dt_rsi_lo']}/{best_rsi['dt_rsi_hi']}  "
          f"ut={best_rsi['ut_rsi_lo']}/{best_rsi['ut_rsi_hi']}  "
          f"rg={best_rsi['rg_rsi_lo']}/{best_rsi['rg_rsi_hi']}")
    print(f"{'='*80}")

    # 프리셋별 RSI 후보: 최적값 + 성향별 변형
    preset_rsi_candidates = {
        "prod": [
            best_rsi,  # 최적값 그대로
            {**best_rsi, "ut_rsi_hi": best_rsi["ut_rsi_hi"] + 3},   # ut_hi +3
            {**best_rsi, "dt_rsi_hi": best_rsi["dt_rsi_hi"] - 3},   # dt_hi -3
        ],
        "stable": [
            # 안정: lo 상향 + hi 하향 (더 극단 조건, 덜 진입)
            {**best_rsi,
             "dt_rsi_lo": min(best_rsi["dt_rsi_lo"] + 3, 30),
             "dt_rsi_hi": max(best_rsi["dt_rsi_hi"] - 3, 55),
             "ut_rsi_lo": min(best_rsi["ut_rsi_lo"] + 2, 47),
             "ut_rsi_hi": max(best_rsi["ut_rsi_hi"] - 3, 70)},
            {**best_rsi,
             "dt_rsi_lo": min(best_rsi["dt_rsi_lo"] + 5, 30),
             "ut_rsi_hi": max(best_rsi["ut_rsi_hi"] - 5, 70)},
            {**best_rsi,
             "dt_rsi_lo": min(best_rsi["dt_rsi_lo"] + 3, 30),
             "ut_rsi_hi": max(best_rsi["ut_rsi_hi"] - 5, 70)},
        ],
        "aggressive": [
            # 공격: lo 하향 + hi 상향 (더 완화 조건, 더 진입)
            {**best_rsi,
             "dt_rsi_lo": max(best_rsi["dt_rsi_lo"] - 3, 15),
             "dt_rsi_hi": min(best_rsi["dt_rsi_hi"] + 3, 75),
             "ut_rsi_lo": max(best_rsi["ut_rsi_lo"] - 3, 30),
             "ut_rsi_hi": min(best_rsi["ut_rsi_hi"] + 3, 90)},
            {**best_rsi,
             "dt_rsi_lo": max(best_rsi["dt_rsi_lo"] - 5, 15),
             "ut_rsi_hi": min(best_rsi["ut_rsi_hi"] + 5, 90)},
            {**best_rsi,
             "dt_rsi_lo": max(best_rsi["dt_rsi_lo"] - 3, 15),
             "ut_rsi_hi": min(best_rsi["ut_rsi_hi"] + 3, 90)},
        ],
        "conservative": [
            # 보수: 더 극단 조건 + 넓은 trail
            {**best_rsi,
             "dt_rsi_lo": min(best_rsi["dt_rsi_lo"] + 5, 32),
             "dt_rsi_hi": max(best_rsi["dt_rsi_hi"] - 5, 55),
             "ut_rsi_lo": min(best_rsi["ut_rsi_lo"] + 5, 50),
             "ut_rsi_hi": max(best_rsi["ut_rsi_hi"] - 5, 68)},
            {**best_rsi,
             "dt_rsi_lo": min(best_rsi["dt_rsi_lo"] + 3, 32),
             "ut_rsi_hi": max(best_rsi["ut_rsi_hi"] - 8, 65)},
            {**best_rsi,
             "dt_rsi_lo": min(best_rsi["dt_rsi_lo"] + 5, 32),
             "ut_rsi_hi": max(best_rsi["ut_rsi_hi"] - 5, 68)},
        ],
    }

    final_presets = {}
    for preset_name, trail in PRESET_TRAILS.items():
        print(f"\n  {'─'*70}")
        print(f"  [{preset_name.upper()}] trail={trail['trail_atr_init']}/{trail['trail_atr_tight']}")
        print(f"  {'─'*70}")
        rows = []
        for i, rsi_cfg in enumerate(preset_rsi_candidates[preset_name]):
            cfg = {**rsi_cfg, **trail}
            label = (f"dt={rsi_cfg['dt_rsi_lo']}/{rsi_cfg['dt_rsi_hi']}  "
                     f"ut={rsi_cfg['ut_rsi_lo']}/{rsi_cfg['ut_rsi_hi']}  "
                     f"rg={rsi_cfg['rg_rsi_lo']}/{rsi_cfg['rg_rsi_hi']}")
            sc, row = run_combo(data, cfg, label)
            row.update(**rsi_cfg, **trail)
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("score", ascending=False)
        best = df.iloc[0]
        final_presets[preset_name] = best
        print(f"\n  ★ [{preset_name}] 최적: score={best['score']:.4f}  "
              f"2026={best['avg_2026']:+.1f}%  MDD={best['max_mdd']*100:.1f}%")

    # 최종 요약
    print(f"\n\n{'='*80}")
    print(f"  최종 프리셋 파라미터 요약")
    print(f"{'='*80}")
    for pname, best in final_presets.items():
        print(f"\n  [{pname.upper()}]")
        print(f"    trail_init={best['trail_atr_init']}  trail_tight={best['trail_atr_tight']}")
        print(f"    dt_lo={best['dt_rsi_lo']}  dt_hi={best['dt_rsi_hi']}")
        print(f"    ut_lo={best['ut_rsi_lo']}  ut_hi={best['ut_rsi_hi']}")
        print(f"    rg_lo={best['rg_rsi_lo']}  rg_hi={best['rg_rsi_hi']}")
        print(f"    → 2026={best['avg_2026']:+.1f}%  MDD={best['max_mdd']*100:.1f}%  "
              f"hist={best['hist_pass']}/{best['hist_total']}  score={best['score']:.4f}")

    return final_presets


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["2a","2b","2c","3","all"])
    args = parser.parse_args()

    print("데이터 로딩 중...")
    data = load_all()
    print("로딩 완료.\n")

    # Phase 2: 각 RSI 축 독립 스윕 (prod trail 고정)
    if args.phase in ("2a", "all"):
        dt_lo, dt_hi, _ = phase2a(data)
    else:
        dt_lo, dt_hi = BASE_PROD["dt_rsi_lo"], BASE_PROD["dt_rsi_hi"]

    if args.phase in ("2b", "all"):
        # dt 최적값 적용
        base_after_2a = {**BASE_PROD, "dt_rsi_lo": dt_lo, "dt_rsi_hi": dt_hi}
        ut_lo, ut_hi, _ = phase2b(data, base_rsi=base_after_2a)
    else:
        base_after_2a = BASE_PROD
        ut_lo, ut_hi = BASE_PROD["ut_rsi_lo"], BASE_PROD["ut_rsi_hi"]

    if args.phase in ("2c", "all"):
        base_after_2b = {**base_after_2a, "ut_rsi_lo": ut_lo, "ut_rsi_hi": ut_hi}
        rg_lo, rg_hi, _ = phase2c(data, base_rsi=base_after_2b)
    else:
        rg_lo, rg_hi = BASE_PROD["rg_rsi_lo"], BASE_PROD["rg_rsi_hi"]

    # Phase 3: 프리셋별 검증
    if args.phase in ("3", "all"):
        best_rsi = dict(
            dt_rsi_lo=dt_lo, dt_rsi_hi=dt_hi,
            ut_rsi_lo=ut_lo, ut_rsi_hi=ut_hi,
            rg_rsi_lo=rg_lo, rg_rsi_hi=rg_hi,
        )
        if args.phase == "3":
            # Phase 3만 실행 시 prod 기본값 사용
            best_rsi = dict(
                dt_rsi_lo=22, dt_rsi_hi=65,
                ut_rsi_lo=40, ut_rsi_hi=75,   # Phase 2에서 찾은 ut_hi=75 반영
                rg_rsi_lo=25, rg_rsi_hi=75,
            )
        phase3(data, best_rsi)


if __name__ == "__main__":
    main()
