"""
scripts/parity_check.py — 백테스트 ↔ 실거래(paper) 거래별 대조 하니스.

목적: 백테스트 trade_log 와 live_trader.py --paper 모드 CSV(logs/paper_trades*.csv,
PAPER_CSV_HEADER)를 거래별로 정렬·대조해 진입/청산가·PnL이 허용오차 내 일치하는지 검증한다.
"백테스트 == paper == 거래소 실현손익" 일치를 지속 측정하는 게이트 — 과거엔 이 검증이 없어
실거래 손실 후에야 괴리를 발견했다. go-live 전 필수 통과 항목.

모드:
  --paper PATH : 실제 paper CSV vs 백테스트 대조 (--coin/--start/--end 로 백테스트 구간 지정)
  --selftest   : paper 데이터 부재 시 — 백테스트 trade_log 를 paper 포맷으로 합성하고 1건에
                 의도적 perturbation 을 주입한 뒤, 대조 로직이 일치/불일치를 정확히 판별하는지 입증.

Usage:
  .venv/bin/python scripts/parity_check.py --selftest --coin btc
  .venv/bin/python scripts/parity_check.py --paper logs/paper_trades.csv --coin btc \
      --start "2026-06-23 07:20" --end "2026-06-24 02:00"
"""
import os, sys, csv, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "src")

import pandas as pd
from strategies.backtest_engine import AntifragileBacktestRunner
from config.af_params import DEFAULT_PARAMS

ENTRY_TOL_BPS = 5.0    # 진입/청산가 허용오차 (basis point = 0.01%)
PNL_TOL       = 0.01   # PnL 허용오차 (절대, 자본수익률 단위)
TIME_TOL_S    = 600    # 청산시각 매칭 허용 (초)


def _parse_paper_csv(path) -> list:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "ts":    pd.Timestamp(str(r["timestamp"]).replace(" UTC", "")),
                "dir":   1 if r["direction"] == "long" else -1,
                "entry": float(r["entry_price"]), "exit": float(r["exit_price"]),
                "pnl":   float(r["pnl"]),
            })
    return rows


def _bt_trades(res) -> list:
    out = []
    for t in res["trade_log"]:
        if t.get("exit_ts") is None:
            continue
        out.append({"ts": pd.Timestamp(t["exit_ts"]), "dir": t["direction"],
                    "entry": float(t["entry"]), "exit": float(t["exit"]), "pnl": float(t["pnl"])})
    return out


def match_trades(paper: list, bt: list) -> list:
    """청산시각+방향 최근접 매칭 후 진입/청산가·PnL 허용오차 비교."""
    used, results = set(), []
    for pr in paper:
        best, bestd = None, None
        for j, b in enumerate(bt):
            if j in used or b["dir"] != pr["dir"]:
                continue
            d = abs((b["ts"] - pr["ts"]).total_seconds())
            if bestd is None or d < bestd:
                best, bestd = j, d
        if best is None or bestd > TIME_TOL_S:
            results.append((pr, None, "unmatched"))
            continue
        used.add(best)
        b = bt[best]
        e_bps = abs(pr["entry"] - b["entry"]) / (b["entry"] + 1e-9) * 1e4
        x_bps = abs(pr["exit"]  - b["exit"])  / (b["exit"]  + 1e-9) * 1e4
        pnl_d = abs(pr["pnl"] - b["pnl"])
        ok = e_bps <= ENTRY_TOL_BPS and x_bps <= ENTRY_TOL_BPS and pnl_d <= PNL_TOL
        tag = "match" if ok else f"MISMATCH(entry {e_bps:.1f}bps, exit {x_bps:.1f}bps, pnl {pnl_d:.4f})"
        results.append((pr, b, tag))
    return results


def _report(results) -> dict:
    n = len(results)
    matched  = sum(1 for _, _, t in results if t == "match")
    mismatch = [(p, t) for p, _, t in results if t.startswith("MISMATCH")]
    unmatch  = [p for p, _, t in results if t == "unmatched"]
    print(f"\n  대조 거래 {n}건 | 일치 {matched} | 불일치 {len(mismatch)} | 미매칭 {len(unmatch)}")
    for p, t in mismatch:
        print(f"    ✗ {p['ts']} dir={p['dir']:+d} {t}")
    for p in unmatch:
        print(f"    ? unmatched: {p['ts']} dir={p['dir']:+d} entry={p['entry']}")
    verdict = (len(mismatch) == 0 and len(unmatch) == 0)
    print(f"  parity: {'✅ 일치' if verdict else '❌ 불일치 존재'}")
    return {"n": n, "matched": matched, "mismatch": len(mismatch), "unmatched": len(unmatch)}


def _synth_paper_from_bt(bt: list) -> list:
    """백테스트 거래를 paper 포맷으로 합성 + 1건에 perturbation 주입 (대조로직 입증용)."""
    paper = [dict(b) for b in bt]
    if len(paper) >= 2:
        paper[1] = dict(paper[1])
        paper[1]["exit"] *= 1.02          # 청산가 +2% 왜곡
        paper[1]["pnl"]  += 0.05          # PnL 왜곡
    return paper


def main():
    ap = argparse.ArgumentParser(description="백테스트 ↔ paper 거래별 parity 대조")
    ap.add_argument("--coin",  default="btc", choices=["btc", "eth", "sol", "xrp"])
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end",   default="2026-06-01")
    ap.add_argument("--leverage", type=int, default=int(os.getenv("LEVERAGE", "10")))
    ap.add_argument("--paper", default=None, help="paper CSV 경로 (미지정+--selftest 시 합성)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=str(ROOT / "models/af_ensemble/saved"))
    ap.add_argument("--preset", default=None, help="af_params 프리셋 (예: candidate) — paper와 동일 설정으로 대조")
    ap.add_argument("--theta", type=float, default=None, help="ML threshold override (예: 0.45)")
    args = ap.parse_args()

    if args.preset:
        from config.af_params import get_preset
        base = get_preset(args.preset)
    else:
        base = dict(DEFAULT_PARAMS)
    base["leverage"] = args.leverage
    runner = AntifragileBacktestRunner.from_saved(args.model, params=base)
    if args.theta is not None:
        runner.ensemble.threshold = args.theta
    df, df_ml = runner.load_coin(args.coin, start=args.start, end=args.end)
    res = runner.run(df, df_ml)
    bt  = _bt_trades(res)
    print(f"[parity] {args.coin.upper()} {args.start}~{args.end} | 백테스트 거래 {len(bt)}건")

    if args.selftest or not args.paper:
        print("  모드: selftest (백테스트 trade_log 합성 + 1건 perturbation 주입)")
        paper = _synth_paper_from_bt(bt)
        results = match_trades(paper, bt)
        rep = _report(results)
        # 입증: 정확히 perturbation 1건만 불일치, 나머지 일치
        expect_mismatch = 1 if len(bt) >= 2 else 0
        ok = (rep["mismatch"] == expect_mismatch and rep["unmatched"] == 0
              and rep["matched"] == len(bt) - expect_mismatch)
        print(f"\n  [selftest 판정] 기대 불일치 {expect_mismatch}건, 실제 {rep['mismatch']}건 "
              f"→ {'✅ 대조로직 정상 (일치+불일치 모두 정확 판별)' if ok else '❌ 대조로직 오류'}")
        sys.exit(0 if ok else 1)
    else:
        paper = _parse_paper_csv(args.paper)
        print(f"  paper CSV {args.paper} | {len(paper)}건")
        results = match_trades(paper, bt)
        rep = _report(results)
        sys.exit(0 if (rep["mismatch"] == 0 and rep["unmatched"] == 0) else 1)


if __name__ == "__main__":
    main()
