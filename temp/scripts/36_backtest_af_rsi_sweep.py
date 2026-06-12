"""
Antifragile AdaptRSI 임계값 조합 스윕
추세(dn/rg/up)별 RSI lo/hi 를 다양하게 조합, 최적 구성 탐색.

현재 라이브: dt(22,65) rg(30,70) ut(35,78)
유저 제안 예시: DN일때 lo=15 hi=70

Usage:
  python temp/scripts/36_backtest_af_rsi_sweep.py
  python temp/scripts/36_backtest_af_rsi_sweep.py --coin eth
  python temp/scripts/36_backtest_af_rsi_sweep.py --coin all --mode both
  python temp/scripts/36_backtest_af_rsi_sweep.py --mode both --windows 10
"""
import sys, argparse, random
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
from pathlib import Path
from hybrid_engine import compute_metrics
from backtest_antifragile import load_coin_full, remove_top_n, COIN_CONFIG, run_antifragile

ROOT = Path(__file__).parent.parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# 탐색 구성 (이름: (dt_lo, dt_hi, rg_lo, rg_hi, ut_lo, ut_hi))
#
# 설계 철학별 분류
#   Current     : 현재 라이브 기준선
#   A(유저)     : DN lo↓ hi↑ — DN 양방향 더 보수적
#   B(추세추종) : DN숏 공격(hi↓), UT롱 공격(lo↑), 역방향 극단값만 진입
#   C(DN숏공격) : B에서 UT만 현재 유지 — DN숏만 공격적
#   D(극단값)   : 전체 임계 폭 넓힘 — 거래수↓ 품질↑ 추구
#   E(UT보수)   : UT에서 숏 거의 안 함 (hi↑) — 상승장 롱 집중
#   F(추세무시) : 추세 무관 동일 임계값 — 순수 역추세 MR
#   G(촘촘)     : 전체 임계 좁힘 — 거래수↑ 빈번 진입
#   H(균형스트) : 전체 균등하게 약간 넓힘
#   I(DN15/60)  : 유저 아이디어 + DN숏 공격 결합
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    "Current    ": (22, 65, 30, 70, 35, 78),  # ◀ 현재 라이브
    "A.User-DN  ": (15, 70, 30, 70, 35, 78),  # DN lo↓ hi↑
    "B.TrendAgg ": (18, 60, 30, 70, 40, 82),  # DN숏공격+UT롱공격
    "C.DN-Short ": (18, 60, 30, 70, 35, 78),  # DN숏만 공격적
    "D.WideOnly ": (15, 75, 22, 78, 28, 82),  # 극단값만 진입
    "E.UT-NoShrt": (22, 65, 30, 70, 40, 85),  # UT 숏 거의 안 함
    "F.Flat-MR  ": (28, 72, 28, 72, 28, 72),  # 추세 무시 MR
    "G.Tight    ": (25, 62, 32, 68, 38, 75),  # 촘촘 (거래↑)
    "H.BalStrch ": (20, 67, 28, 72, 33, 80),  # 균형 스트레치
    "I.DN15/60  ": (15, 60, 30, 70, 35, 78),  # 유저+DN숏 공격 결합
}


def _rsi_params(cfg_tuple):
    dt_lo, dt_hi, rg_lo, rg_hi, ut_lo, ut_hi = cfg_tuple
    return dict(
        dt_rsi_lo=dt_lo, dt_rsi_hi=dt_hi,
        rg_rsi_lo=rg_lo, rg_rsi_hi=rg_hi,
        ut_rsi_lo=ut_lo, ut_rsi_hi=ut_hi,
    )


def run_all_configs(df, days_total):
    rows = []
    for name, cfg in CONFIGS.items():
        res = run_antifragile(df, **_rsi_params(cfg))
        m   = res["metrics"]
        tl  = res["trade_log"]
        tpd = m.get("tpd", round(len(tl) / max(days_total, 1), 2))
        r5  = remove_top_n(tl, 5)
        ok  = sum([m["total_return"] > 0, tpd >= 1.5, r5 > 0])
        rows.append({
            "name": name,
            "cfg":  cfg,
            "n":    m["n_trades"],
            "wr":   round(m["win_rate"], 1),
            "tpd":  tpd,
            "ret":  round(m["total_return"], 1),
            "mdd":  round(m["mdd"], 1),
            "pf":   round(m.get("profit_factor", 0), 3),
            "top5": round(r5, 1),
            "ok":   ok,
            "long": m.get("long_cnt", 0),
            "shrt": m.get("short_cnt", 0),
        })
    return rows


def print_sweep_table(rows, title=""):
    print(f"\n{'='*100}")
    if title:
        print(f"  {title}")
    # 파라미터 키 출력
    print(f"  {'구성':<14} {'dt lo/hi':<10} {'rg lo/hi':<10} {'ut lo/hi':<10} "
          f"{'n':>5} {'L/S':>7} {'WR':>6} {'TPD':>5} {'수익':>8} {'MDD':>5} {'PF':>6} {'Top5':>7} {'판정':>4}")
    print(f"  {'─'*96}")
    for r in rows:
        cfg = r["cfg"]
        dt  = f"{cfg[0]}/{cfg[1]}"
        rg  = f"{cfg[2]}/{cfg[3]}"
        ut  = f"{cfg[4]}/{cfg[5]}"
        ls  = f"{r['long']}/{r['shrt']}"
        mark = "✅" if r["ok"] == 3 else ("⚠️" if r["ok"] >= 2 else "❌")
        is_current = "Current" in r["name"]
        prefix = "▶ " if is_current else "  "
        print(f"{prefix}{r['name']:<14} {dt:<10} {rg:<10} {ut:<10} "
              f"{r['n']:>5} {ls:>7} {r['wr']:>5.1f}% "
              f"{r['tpd']:>4.2f} {r['ret']:>+7.1f}% {r['mdd']:>4.1f}% "
              f"{r['pf']:>5.3f} {r['top5']:>+6.1f}%  {mark}({r['ok']}/3)")
    print(f"{'='*100}")

    # 현재 대비 개선율 요약
    cur = next((r for r in rows if "Current" in r["name"]), None)
    if cur:
        print(f"\n  현재(▶) 대비 수익 차이:")
        print(f"  {'구성':<14} {'수익 차이':>10} {'MDD 차이':>10} {'TPD 차이':>8} {'비고':>8}")
        print(f"  {'─'*52}")
        for r in rows:
            if "Current" in r["name"]:
                continue
            d_ret = r["ret"] - cur["ret"]
            d_mdd = r["mdd"] - cur["mdd"]
            d_tpd = r["tpd"] - cur["tpd"]
            mark = "✅" if r["ok"] == 3 else ("⚠️" if r["ok"] >= 2 else "❌")
            sign_r = "+" if d_ret >= 0 else ""
            sign_m = "+" if d_mdd >= 0 else ""
            sign_t = "+" if d_tpd >= 0 else ""
            print(f"  {r['name']:<14} {sign_r}{d_ret:>7.1f}%   {sign_m}{d_mdd:>6.1f}%   "
                  f"{sign_t}{d_tpd:>5.2f}   {mark}({r['ok']}/3)")


def run_random_validation(all_df, coin_label, seed, windows, window_days, hist_start):
    all_df = all_df.dropna(subset=["_rsi", "_atr"])
    rng = random.Random(seed)
    possible = all_df[
        (all_df.index >= hist_start) &
        (all_df.index <= all_df.index[-1] - pd.Timedelta(days=window_days))
    ].index
    chosen = sorted(rng.choices(possible, k=windows))

    print(f"\n랜덤 {windows}창 검증 ({window_days}일 윈도우, seed={seed}) — {coin_label}")

    # 구성별 집계
    agg = {name: [] for name in CONFIGS}

    for sd in chosen:
        ed  = sd + pd.Timedelta(days=window_days)
        seg = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
        if len(seg) < 500:
            continue
        days = len(seg) / 288
        for name, cfg in CONFIGS.items():
            res = run_antifragile(seg, **_rsi_params(cfg))
            m   = res["metrics"]
            tl  = res["trade_log"]
            tpd = m.get("tpd", round(len(tl) / max(days, 1), 2))
            r5  = remove_top_n(tl, 5)
            ok  = sum([m["total_return"] > 0, tpd >= 1.5, r5 > 0])
            agg[name].append({"ret": m["total_return"], "ok": ok, "tpd": tpd})

    print(f"\n  {'구성':<14} {'avg수익':>9} {'3/3':>5} {'avg TPD':>8} {'비고'}")
    print(f"  {'─'*55}")
    cur_avg = None
    for name, lst in agg.items():
        if not lst:
            continue
        avg_ret = np.mean([x["ret"] for x in lst])
        avg_tpd = np.mean([x["tpd"] for x in lst])
        p3      = sum(x["ok"] == 3 for x in lst)
        n       = len(lst)
        mark = "✅" if p3 >= n * 0.8 else ("⚠️" if p3 >= n * 0.5 else "❌")
        flag = " ◀ 현재" if "Current" in name else ""
        if "Current" in name:
            cur_avg = avg_ret
        diff = f"  ({avg_ret - cur_avg:+.1f}%)" if cur_avg is not None and "Current" not in name else ""
        print(f"  {name:<14} {avg_ret:>+7.1f}%   {p3:>2}/{n}  {avg_tpd:>6.2f}   {mark}{diff}{flag}")


def main():
    parser = argparse.ArgumentParser(description="AdaptRSI 임계값 조합 스윕")
    parser.add_argument("--coin",        default="btc", choices=["btc","eth","sol","xrp","both","all"])
    parser.add_argument("--mode",        default="2026", choices=["2026","random","both"])
    parser.add_argument("--windows",     type=int, default=10)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--window-days", type=int, default=91)
    args = parser.parse_args()

    coin_map = {"both": ["btc","eth"], "all": ["btc","eth","sol","xrp"]}
    coins = coin_map.get(args.coin, [args.coin])

    for coin in coins:
        cfg_info = COIN_CONFIG[coin]
        print(f"\n{'█'*70}")
        print(f"  {cfg_info['label']}/USDT — AdaptRSI 임계값 스윕")
        print(f"{'█'*70}")

        all_df = load_coin_full(coin)

        if args.mode in ("2026", "both"):
            seg26 = all_df[all_df.index >= "2026-01-01"].copy()
            days  = len(seg26) / 288
            print(f"\n▶ 2026 OOS ({len(seg26):,}봉 / {days:.0f}일)")
            rows = run_all_configs(seg26, days)
            print_sweep_table(rows, f"{cfg_info['label']} 2026 OOS — RSI 임계값 스윕")

        if args.mode in ("random", "both"):
            run_random_validation(
                all_df, cfg_info["label"],
                seed=args.seed,
                windows=args.windows,
                window_days=args.window_days,
                hist_start=cfg_info["hist_start"],
            )


if __name__ == "__main__":
    main()
