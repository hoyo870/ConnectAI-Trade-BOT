"""
backtest_timeframe_compare.py — 타임프레임 리샘플 탐색 비교 (5m / 15m / 30m / 60m)

⚠️ 탐색용(rough) 비교임. ML 앙상블은 5m 피처로 학습돼 있어 15/30/60m 바를
   그대로 먹이면 학습 분포 밖(OOD) 입력이 된다. 지표(ATR span=14, RSI com=13)와
   trailing stop도 5m 변동성 기준으로 튜닝됨. 따라서 여기 수치는 "어느 타임프레임이
   유망한가"의 방향성 참고용이며, 공정한 벤치마크가 아니다. 절대 신뢰 기준 금지.

기존 5m 파이프라인(add_indicators_af + ML 컨텍스트 + AntifragileBacktestRunner)을
리샘플된 OHLCV 위에서 그대로 재실행한다. 모델/파라미터/레버리지/threshold는
scripts/backtest_af_exact.py --mode 2026 과 동일하게 .env에서 가져온다.

Usage:
  python scripts/backtest_timeframe_compare.py
  python scripts/backtest_timeframe_compare.py --coin btc
  python scripts/backtest_timeframe_compare.py --tfs 5m,15m,30m,60m --start 2026-01-01 --end 2026-06-01
"""

from __future__ import annotations
import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "src")

import pandas as pd

from config.loader import load_coin_raw
from strategies.indicators import add_indicators_af
from strategies.backtest_engine import AntifragileBacktestRunner
from config.af_params import DEFAULT_PARAMS, PRESETS as _PRESET_DEFS, get_preset

_DEFAULT_MODEL = str(ROOT / "models/af_ensemble/saved")
_BB_SIGMA = 0

# pandas resample 규칙 매핑 (5m은 원본 그대로 = baseline)
_TF_RULE = {"5m": None, "15m": "15min", "30m": "30min", "60m": "60min"}


def _load_env_file() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip(); value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _resample(raw: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    """5m OHLCV → 상위 타임프레임. rule=None이면 원본 반환."""
    if rule is None:
        return raw
    out = raw.resample(rule, label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    })
    return out.dropna(subset=["close"])


def _run_one(runner, raw: pd.DataFrame, tf: str, start: str, end: str) -> dict | None:
    """단일 코인×타임프레임 백테스트 → 요약 metrics."""
    rs = _resample(raw, _TF_RULE[tf])
    df    = add_indicators_af(rs, _BB_SIGMA)
    df_ml = runner._build_ml_context(rs)
    if start: df = df[df.index >= start]; df_ml = df_ml[df_ml.index >= start]
    if end:   df = df[df.index <  end];   df_ml = df_ml[df_ml.index <  end]
    if len(df) < 50:
        return None
    days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    res  = runner.run(df.copy(), df_ml.copy())
    m    = res["metrics"]
    return {
        "tf": tf, "bars": len(df), "days": days,
        "n_trades": m["n_trades"], "tpd": m["n_trades"] / days if days > 0 else 0,
        "win_rate": m["win_rate"], "total_return": m["total_return"],
        "mdd": m["mdd"], "pf": m["profit_factor"],
    }


def main():
    _load_env_file()
    parser = argparse.ArgumentParser(description="타임프레임 리샘플 탐색 비교 (탐색용, ML은 5m OOD)")
    parser.add_argument("--coin",  default="all", choices=["btc", "eth", "sol", "xrp", "all"])
    parser.add_argument("--tfs",   default="5m,15m,30m,60m")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end",   default="2026-06-01")
    parser.add_argument("--model", default=os.getenv("ML_MODEL_DIR", _DEFAULT_MODEL))
    parser.add_argument("--leverage", type=int, default=int(os.getenv("LEVERAGE", "7")))
    args = parser.parse_args()

    tfs = [t.strip() for t in args.tfs.split(",") if t.strip() in _TF_RULE]

    af_params = {**DEFAULT_PARAMS, "leverage": args.leverage}
    preset_name = os.getenv("AF_PARAM_PRESET", "").lower()
    if preset_name and preset_name != "prod" and preset_name in _PRESET_DEFS:
        af_params.update(get_preset(preset_name))
    af_params["leverage"] = args.leverage

    runner = AntifragileBacktestRunner.from_saved(args.model, params=af_params)
    _ml_th = os.getenv("ML_THRESHOLD")
    if _ml_th:
        runner.ensemble.threshold = float(_ml_th)

    print("=" * 78)
    print("  타임프레임 리샘플 탐색 비교  ⚠️ ML은 5m 학습(OOD) — 방향성 참고용, 신뢰기준 금지")
    print(f"  기간: {args.start} ~ {args.end}   LEVERAGE=x{args.leverage}   "
          f"theta={runner.ensemble.threshold:.3f}   preset={preset_name or 'prod'}")
    print("=" * 78)

    coins = ["btc", "eth", "sol", "xrp"] if args.coin == "all" else [args.coin]
    all_rows: dict[str, list] = {}

    for coin in coins:
        raw = load_coin_raw(coin)
        rows = []
        for tf in tfs:
            r = _run_one(runner, raw, tf, args.start, args.end)
            if r:
                rows.append(r)
        all_rows[coin] = rows

        print(f"\n■ {coin.upper()}/USDT")
        print(f"  {'TF':<5}{'봉수':>7}{'거래수':>7}{'WR':>8}{'TPD':>7}{'총수익':>10}{'MDD':>8}{'PF':>7}")
        print(f"  {'-'*58}")
        for r in rows:
            print(f"  {r['tf']:<5}{r['bars']:>7}{r['n_trades']:>7}{r['win_rate']:>7.1f}%"
                  f"{r['tpd']:>7.2f}{r['total_return']:>+9.1f}%{r['mdd']:>7.1f}%{r['pf']:>7.2f}")

    # ── 타임프레임별 4코인 집계 ──
    print(f"\n{'='*78}")
    print("  타임프레임별 집계 (4코인 평균)")
    print(f"  {'TF':<5}{'평균총수익':>12}{'평균WR':>9}{'평균TPD':>9}{'평균MDD':>9}{'양수코인':>9}")
    print(f"  {'-'*54}")
    for tf in tfs:
        vals = [r for rows in all_rows.values() for r in rows if r["tf"] == tf]
        if not vals:
            continue
        n = len(vals)
        avg_ret = sum(v["total_return"] for v in vals) / n
        avg_wr  = sum(v["win_rate"] for v in vals) / n
        avg_tpd = sum(v["tpd"] for v in vals) / n
        avg_mdd = sum(v["mdd"] for v in vals) / n
        pos     = sum(1 for v in vals if v["total_return"] > 0)
        print(f"  {tf:<5}{avg_ret:>+11.1f}%{avg_wr:>8.1f}%{avg_tpd:>9.2f}{avg_mdd:>8.1f}%{pos:>6}/{n}")
    print("=" * 78)
    print("  ⚠️ 재확인: ML 필터가 5m 분포 기준이라 상위 TF는 OOD. 유망 TF가 보이면")
    print("     '타임프레임별 정식 재구축'(ML 재학습+파라미터 재튜닝)으로만 실거래 판단할 것.")


if __name__ == "__main__":
    main()
