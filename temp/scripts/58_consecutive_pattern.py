"""
2026 OOS 기간 패턴 분석:
- 청산 → 같은 방향 재진입 → 청산 연속 패턴 횟수
- 1시간(12봉) 이내 같은 방향 3회 이상 연속 청산 케이스
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from backtest_antifragile import run_antifragile, load_coin_full, _normalize_index
from config.af_params import get_preset

LEVERAGE  = 7
OOS_START = "2026-01-01"
OOS_END   = "2026-06-01"
COINS     = ["btc", "eth", "sol", "xrp"]
CFG       = get_preset("prod")
ONE_HOUR  = 12  # 5분봉 기준 1시간 = 12봉


def analyze_patterns(trade_log):
    """
    연속 같은 방향 클러스터 분석.
    클러스터: 인접 거래가 같은 방향으로 이어지는 그룹
    """
    if not trade_log:
        return []

    clusters = []
    cur = [trade_log[0]]

    for t in trade_log[1:]:
        prev = cur[-1]
        # 같은 방향 + 재진입 (이전 exit_bar >= 현재 entry_bar - 5 허용)
        if t["direction"] == prev["direction"] and t["entry_bar"] <= prev["exit_bar"] + 5:
            cur.append(t)
        else:
            if len(cur) >= 2:
                clusters.append(cur)
            cur = [t]
    if len(cur) >= 2:
        clusters.append(cur)

    return clusters


def analyze_coin(coin):
    df = _normalize_index(load_coin_full(coin))
    sub = df[(df.index >= OOS_START) & (df.index < OOS_END)].copy()

    r  = run_antifragile(sub, leverage=LEVERAGE, max_dd_cb=0.99, **CFG)
    tl = r["trade_log"]

    if not tl:
        return

    # entry_bar/exit_bar → 실제 타임스탬프 매핑
    sub_reset = sub.reset_index()

    def bar_to_ts(bar):
        if 0 <= bar < len(sub_reset):
            return sub_reset.iloc[bar]["timestamp"]
        return None

    # 클러스터 분석
    clusters = analyze_patterns(tl)

    # 1시간 이내 3회 이상 클러스터
    hot_clusters = []
    for cl in clusters:
        span_bars = cl[-1]["exit_bar"] - cl[0]["entry_bar"]
        if len(cl) >= 3 and span_bars <= ONE_HOUR:
            hot_clusters.append(cl)

    print(f"\n{'='*70}")
    print(f"  {coin.upper()}  |  2026 OOS  |  총 거래: {len(tl)}건")
    print(f"{'='*70}")

    # ── 연속 같은 방향 클러스터 통계 ──
    print(f"\n  [연속 같은 방향 클러스터 (2회 이상)]")
    cnt_by_size = defaultdict(int)
    for cl in clusters:
        cnt_by_size[len(cl)] += 1
    for size in sorted(cnt_by_size):
        print(f"    {size}회 연속: {cnt_by_size[size]}건")
    print(f"    합계: {sum(cnt_by_size.values())}개 클러스터")

    # ── 1시간 이내 3회 이상 ──
    print(f"\n  [1시간(12봉) 이내 같은 방향 3회 이상]")
    if not hot_clusters:
        print("    해당 케이스 없음")
    else:
        print(f"    총 {len(hot_clusters)}건")
        for i, cl in enumerate(hot_clusters, 1):
            span_bars = cl[-1]["exit_bar"] - cl[0]["entry_bar"]
            t0 = bar_to_ts(cl[0]["entry_bar"])
            t1 = bar_to_ts(cl[-1]["exit_bar"])
            dir_label = "LONG" if cl[0]["direction"] == 1 else "SHORT"
            pnls = [t["pnl"]*100 for t in cl]
            print(f"\n    [{i}] {dir_label}  {len(cl)}회  {span_bars}봉({span_bars*5}분)")
            if t0: print(f"        {t0} ~ {t1}")
            for j, (t, p) in enumerate(zip(cl, pnls), 1):
                hold = t["exit_bar"] - t["entry_bar"]
                print(f"        거래{j}: hold={hold}봉({hold*5}분)  pnl={p:+.2f}%")

    # ── 전체 "청산→같은방향재진입" 단순 횟수 ──
    same_dir_reentry = 0
    for i in range(1, len(tl)):
        prev, cur = tl[i-1], tl[i]
        if (cur["direction"] == prev["direction"]
                and cur["entry_bar"] <= prev["exit_bar"] + 5):
            same_dir_reentry += 1

    print(f"\n  [청산→같은방향 재진입 총 횟수]")
    print(f"    {same_dir_reentry}회  (전체 {len(tl)}건 중 {same_dir_reentry/(len(tl)-1)*100:.1f}%)")


print("데이터 로딩 및 분석 중...\n")
for coin in COINS:
    analyze_coin(coin)
