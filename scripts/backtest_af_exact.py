"""
backtest_af_exact.py — live_trader.py 완벽 모방 백테스트 CLI

유일한 공식 백테스트 엔진: strategies/backtest_engine.AntifragileBacktestRunner
ML 앙상블 필터는 필수이며 비활성화 불가.

Usage:
  python scripts/backtest_af_exact.py --mode 2026
  python scripts/backtest_af_exact.py --mode 2026 --coin all
  python scripts/backtest_af_exact.py --mode hist --windows 10 --seed 42
  python scripts/backtest_af_exact.py --mode jun1819
"""

import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config.loader import load_coin_raw
from strategies.indicators import add_indicators_af
from strategies.backtest_engine import AntifragileBacktestRunner
from config.af_params import DEFAULT_PARAMS, PRESETS as _PRESET_DEFS, get_preset

_DEFAULT_MODEL = str(ROOT / "models/af_ensemble/saved")
_BB_SIGMA = 0  # live_trader.py와 동일: bb_sigma=0 (EMA 크로스, BB 미사용)


def _load_env_file() -> None:
    """live_trader.py와 동일: .env 파일을 os.environ에 로드 (미설정 키만)."""
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


# ── main ──────────────────────────────────────────────────────────────────────

HIST_START = {
    "btc": "2020-01-01", "eth": "2021-04-01",
    "sol": "2021-06-01", "xrp": "2020-06-01",
}

CHART_DIR = ROOT / "temp" / "charts"


def _save_jun_charts(label: str, mode: str, df: pd.DataFrame,
                     res: dict, initial_capital: float = 10_000.0) -> None:
    """jun* 모드 전용: 종가 + 바이홀드 + 전략 수익 3개 서브플롯을 하나의 이미지로 저장."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    trade_log    = res["trade_log"]
    equity_curve = res["equity_curve"]
    closes       = df["close"].values
    timestamps   = df.index

    # ── 바이홀드 / 전략 equity ────────────────────────────────────────────────
    bh_equity = initial_capital * closes / closes[0]
    bh_ret    = (bh_equity / initial_capital - 1) * 100

    eq = np.array(equity_curve)
    if len(eq) < len(timestamps):
        eq = np.pad(eq, (0, len(timestamps) - len(eq)), mode="edge")
    else:
        eq = eq[:len(timestamps)]
    strat_ret = (eq / initial_capital - 1) * 100

    # ── 3행 1열 서브플롯 ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 13),
                             gridspec_kw={"hspace": 0.45})
    date_fmt = mdates.DateFormatter("%m/%d")
    date_loc = mdates.AutoDateLocator()

    # ── 서브플롯 1: 종가 + 진입/청산 마커 ────────────────────────────────────
    ax = axes[0]
    ax.plot(timestamps, closes, color="#4a90d9", linewidth=0.8, label="Close")
    for t in trade_log:
        entry_px = t.get("entry")
        exit_px  = t.get("exit")
        if entry_px is None or exit_px is None:
            continue
        direction   = t["direction"]
        color_entry = "#2ecc71" if direction == 1 else "#e74c3c"
        color_exit  = "#e74c3c" if direction == 1 else "#2ecc71"
        ei = int(np.argmin(np.abs(closes - entry_px)))
        xi = int(np.argmin(np.abs(closes[ei:] - exit_px))) + ei
        ax.scatter(timestamps[ei], entry_px, marker="^", color=color_entry, s=40, zorder=5)
        ax.scatter(timestamps[xi], exit_px,  marker="v", color=color_exit,  s=40, zorder=5)
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(date_loc)
    plt.setp(ax.get_xticklabels(), rotation=20)
    ax.set_title(f"{label} 종가  ({timestamps[0].strftime('%m/%d')} ~ {timestamps[-1].strftime('%m/%d')})",
                 fontsize=11)
    ax.set_ylabel("Price (USDT)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── 서브플롯 2: 바이홀드 수익률 ──────────────────────────────────────────
    ax = axes[1]
    ax.plot(timestamps, bh_ret, color="#9b59b6", linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.fill_between(timestamps, bh_ret, 0, where=(bh_ret >= 0), alpha=0.15, color="#2ecc71")
    ax.fill_between(timestamps, bh_ret, 0, where=(bh_ret <  0), alpha=0.15, color="#e74c3c")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(date_loc)
    plt.setp(ax.get_xticklabels(), rotation=20)
    ax.set_title(f"{label} 바이홀드  최종 {float(bh_ret[-1]):+.1f}%", fontsize=11)
    ax.set_ylabel("수익률 (%)")
    ax.grid(alpha=0.3)

    # ── 서브플롯 3: 전략 수익 곡선 ───────────────────────────────────────────
    ax = axes[2]
    ax.plot(timestamps, strat_ret, color="#e67e22", linewidth=1.2, label="Strategy")
    ax.plot(timestamps, bh_ret,    color="#9b59b6", linewidth=0.8,
            linestyle="--", alpha=0.6, label="Buy&Hold")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.fill_between(timestamps, strat_ret, 0, where=(strat_ret >= 0), alpha=0.12, color="#2ecc71")
    ax.fill_between(timestamps, strat_ret, 0, where=(strat_ret <  0), alpha=0.12, color="#e74c3c")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(date_loc)
    plt.setp(ax.get_xticklabels(), rotation=20)
    m = res["metrics"]
    ax.set_title(
        f"{label} 전략 수익  최종 {float(strat_ret[-1]):+.1f}%  "
        f"WR={m['win_rate']:.1f}%  거래={m['n_trades']}건  MDD={m['mdd']:.1f}%",
        fontsize=11,
    )
    ax.set_ylabel("수익률 (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    out = CHART_DIR / f"{label.lower()}_{mode}_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 차트 저장: {out}")


def main():
    _load_env_file()  # live_trader.py와 동일하게 .env 우선 로드

    # .env에서 실거래와 동일한 값 적용 (CLI 인수로 오버라이드 가능)
    _env_model   = os.getenv("ML_MODEL_DIR", _DEFAULT_MODEL)
    _env_lev     = int(os.getenv("LEVERAGE", "7"))

    parser = argparse.ArgumentParser(description="live_trader 완벽 모방 백테스트 (ML 필수)")
    parser.add_argument("--coin",        default="all", choices=["btc", "eth", "sol", "xrp", "all"])
    parser.add_argument("--mode",        default="2026", choices=["2026", "hist", "jun1819", "jun2022", "jun"])
    parser.add_argument("--windows",     type=int, default=10)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--window-days", type=int, default=91)
    parser.add_argument("--model",       default=_env_model,
                        help=f"AFEnsemble 저장 디렉터리 (기본: .env ML_MODEL_DIR 또는 {_DEFAULT_MODEL})")
    parser.add_argument("--leverage",    type=int, default=_env_lev,
                        help=f"레버리지 (기본: .env LEVERAGE 또는 7)")
    args = parser.parse_args()

    # live_trader.py와 동일: AF_PARAM_PRESET 프리셋 적용
    af_params = {**DEFAULT_PARAMS, "leverage": args.leverage}
    preset_name = os.getenv("AF_PARAM_PRESET", "").lower()
    if preset_name and preset_name != "prod" and preset_name in _PRESET_DEFS:
        af_params.update(get_preset(preset_name))
    af_params["leverage"] = args.leverage  # LEVERAGE는 프리셋보다 우선

    runner = AntifragileBacktestRunner.from_saved(args.model, params=af_params)
    print(f"[ML 필터] 로드 완료: {args.model}  theta={runner.ensemble.threshold:.3f}")
    print(f"[설정]    LEVERAGE={args.leverage}  AF_PARAM_PRESET={preset_name or 'prod(기본)'}  (.env LEVERAGE: {os.getenv('LEVERAGE', '미설정')})")

    coins = ["btc", "eth", "sol", "xrp"] if args.coin == "all" else [args.coin]

    for coin in coins:
        label = coin.upper()
        print(f"\n{'█'*66}")
        print(f"  {label}/USDT — live_trader 완벽 모방 백테스트 [ML 필수]")
        print(f"  BB σ={_BB_SIGMA}  ML theta={runner.ensemble.threshold:.3f}")
        print(f"{'█'*66}")

        if args.mode == "2026":
            df, df_ml = runner.load_coin(coin, start="2026-01-01", end="2026-06-01")
            if len(df) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df)}봉)"); continue
            days = (df.index[-1] - df.index[0]).days
            print(f"\n[{label}] 2026 OOS: {df.index[0].date()} ~ {df.index[-1].date()}  ({days}일)")
            res = runner.run(df, df_ml)
            runner.print_result(f"{label} 2026 OOS", res, days)

        elif args.mode == "hist":
            hs = HIST_START.get(coin, "2020-01-01")
            print(f"\n[{label}] 랜덤 히스토리 검증")
            runner.run_hist_validation(coin, seed=args.seed, windows=args.windows,
                                       window_days=args.window_days, hist_start=hs)

        elif args.mode == "jun1819":
            df, df_ml = runner.load_coin(coin, start="2026-06-18 03:37", end="2026-06-19 12:00")
            if len(df) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df)}봉)"); continue
            days = (df.index[-1] - df.index[0]).total_seconds() / 86400
            print(f"\n[{label}] Jun 18~19 실거래 기간")
            res = runner.run(df, df_ml)
            runner.print_result(f"{label} Jun18~19", res, days)
            print(f"\n  상세 거래:")
            for t in res["trade_log"]:
                dr   = "롱" if t["direction"] == 1 else "숏"
                flip = " ◀FLIP" if t.get("reason") == "reverse_flip" else ""
                print(f"    {dr}  pnl={t['pnl']*100:+.3f}%  {t.get('reason','')}{flip}")
            _save_jun_charts(label, "jun1819", df, res)

        elif args.mode == "jun2022":
            df, df_ml = runner.load_coin(coin, start="2026-06-20 18:30", end="2026-06-22 13:30")
            if len(df) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df)}봉)"); continue
            days = (df.index[-1] - df.index[0]).total_seconds() / 86400
            print(f"\n[{label}] Jun 20~22 실거래 기간")
            res = runner.run(df, df_ml)
            runner.print_result(f"{label} Jun20~22", res, days)
            print(f"\n  상세 거래:")
            for t in res["trade_log"]:
                dr   = "롱" if t["direction"] == 1 else "숏"
                flip = " ◀FLIP" if t.get("reason") == "reverse_flip" else ""
                print(f"    {dr}  pnl={t['pnl']*100:+.3f}%  {t.get('reason','')}{flip}")
            _save_jun_charts(label, "jun2022", df, res)

        elif args.mode == "jun":
            df, df_ml = runner.load_coin(coin, start="2026-06-01 00:00", end="2026-06-22 17:00")
            if len(df) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df)}봉)"); continue
            days = (df.index[-1] - df.index[0]).total_seconds() / 86400
            print(f"\n[{label}] Jun 01~22 실거래 기간")
            res = runner.run(df, df_ml)
            runner.print_result(f"{label} Jun01~22", res, days)
            print(f"\n  상세 거래:")
            for t in res["trade_log"]:
                dr   = "롱" if t["direction"] == 1 else "숏"
                flip = " ◀FLIP" if t.get("reason") == "reverse_flip" else ""
                print(f"    {dr}  pnl={t['pnl']*100:+.3f}%  {t.get('reason','')}{flip}")
            _save_jun_charts(label, "jun", df, res)


if __name__ == "__main__":
    main()
