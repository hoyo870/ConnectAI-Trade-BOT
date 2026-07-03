"""
temp/scripts/lstm_ablation.py
빠른 실험: 앙상블에서 LSTM이 기여하는가? (LSTM val AUC 0.51 ≈ 랜덤 → 죽은 가중치 의심)

검증된 saved/ 모델로, 가중치를 바꿔가며 2026 OOS 비교:
  - ensemble 0.5/0.5 (현행)
  - LGBM-solo (lstm_weight=0) — θ 스윕(확률 스케일 다름)
  - LSTM-solo (lgbm_weight=0) — 참고

후보 설정(δ=10 RSI + trail_atr_init=2.0 + add_levels=0) 고정. 4코인 2026 OOS.
LGBM-solo가 앙상블 이상이면 → LSTM은 노이즈/죽은가중치, 제거가 답.

Usage: .venv/bin/python temp/scripts/lstm_ablation.py
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "src")

from strategies.backtest_engine import AntifragileBacktestRunner, robust_metrics
from config.af_params import DEFAULT_PARAMS

_LO = ["dt_rsi_lo", "rg_rsi_lo", "ut_rsi_lo"]
_HI = ["dt_rsi_hi", "rg_rsi_hi", "ut_rsi_hi"]
COINS = ["btc", "eth", "sol", "xrp"]
START, END, LEV = "2026-01-01", "2026-06-01", 10


def candidate_params(lev, rsi_strict=10, trail_init=2.0):
    p = {**DEFAULT_PARAMS, "leverage": lev, "add_levels": 0, "trail_atr_init": trail_init}
    for k in _LO: p[k] = max(5, DEFAULT_PARAMS[k] - rsi_strict)
    for k in _HI: p[k] = min(95, DEFAULT_PARAMS[k] + rsi_strict)
    return p


def run_config(runner, lgbm_w, lstm_w, theta, dfs):
    runner.ensemble.lgbm_weight = lgbm_w
    runner.ensemble.lstm_weight = lstm_w
    runner.ensemble.threshold = theta
    tot = 0.0; pos = 0; shs = []; ns = 0
    per = []
    for coin, (df, dfml) in dfs.items():
        m = runner.run(df, dfml)["metrics"]
        tot += m["total_return"]; pos += m["total_return"] > 0
        shs.append(m.get("sharpe", 0.0)); ns += m["n_trades"]
        per.append(m["total_return"])
    return tot, pos, sum(shs) / len(shs), ns, per


def main():
    runner = AntifragileBacktestRunner.from_saved("models/af_ensemble/saved",
                                                  params=candidate_params(LEV))
    # 코인 데이터 1회 로드 (재사용)
    dfs = {c: runner.load_coin(c, start=START, end=END) for c in COINS}

    configs = [
        ("ensemble 0.5/0.5 θ0.45", 0.5, 0.5, 0.45),
        ("LGBM-solo      θ0.40", 1.0, 0.0, 0.40),
        ("LGBM-solo      θ0.45", 1.0, 0.0, 0.45),
        ("LGBM-solo      θ0.50", 1.0, 0.0, 0.50),
        ("LGBM-solo      θ0.55", 1.0, 0.0, 0.55),
        ("LSTM-solo      θ0.50", 0.0, 1.0, 0.50),
    ]
    print(f"[ablation] saved 모델 | 2026 OOS {START}~{END} | 후보 δ10/trail2.0/lev{LEV} | 4코인 합산")
    print(f"  {'config':<24} {'합산수익':>9} {'양수':>4} {'avgSharpe':>9} {'총거래':>6}  {'BTC/ETH/SOL/XRP'}")
    print(f"  {'-'*88}")
    for name, lw, sw, th in configs:
        tot, pos, sh, ns, per = run_config(runner, lw, sw, th, dfs)
        pstr = "/".join(f"{x:+.0f}" for x in per)
        print(f"  {name:<24} {tot:>+8.1f}% {pos:>3}/4 {sh:>+8.2f} {ns:>6}  {pstr}")
    print("\n  → LGBM-solo가 ensemble 이상이면 LSTM은 죽은가중치(제거 권장).")


if __name__ == "__main__":
    main()
