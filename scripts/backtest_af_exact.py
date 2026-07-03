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

# 한글 폰트 설정 (macOS: AppleGothic, 없으면 경고만 억제)
import warnings
_KO_FONTS = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"]
_found_font = next(
    (f for f in _KO_FONTS
     if any(f.lower() in p.name.lower()
            for p in matplotlib.font_manager.fontManager.ttflist)),
    None,
)
if _found_font:
    plt.rcParams["font.family"] = _found_font
    plt.rcParams["axes.unicode_minus"] = False
else:
    warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*missing from font")

from config.loader import load_coin_raw
from strategies.indicators import add_indicators_af
from strategies.backtest_engine import AntifragileBacktestRunner
from config.af_params import DEFAULT_PARAMS, PRESETS as _PRESET_DEFS, get_preset
from src.data_fetcher import fetch_ohlcv

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

_SYMBOLS = {"btc": "BTC/USDT", "eth": "ETH/USDT", "sol": "SOL/USDT", "xrp": "XRP/USDT"}


def _existing_max_ts(coin: str):
    """data/raw의 해당 코인 CSV들에서 가장 최근 봉 timestamp(UTC) 조회. 없으면 None."""
    sym = _SYMBOLS[coin].replace("/", "")
    max_ts = None
    for f in sorted((ROOT / "data/raw").glob(f"{sym}_5m_*.csv")):
        try:
            last = pd.read_csv(f, usecols=["timestamp"]).iloc[-1, 0]
            ts = pd.Timestamp(last)
            ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        except Exception:
            continue
        if max_ts is None or ts > max_ts:
            max_ts = ts
    return max_ts


def ensure_data_coverage(coin: str, start: str, end) -> None:
    """요청 기간(end)이 보유 데이터보다 최신이면 Binance에서 부족분을 자동 다운로드."""
    end_ts   = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
    have_max = _existing_max_ts(coin)
    # 완성봉 기준 한 봉(5분) 여유: 보유 최신봉이 요청 end-1봉 이상이면 갱신 불필요
    if have_max is not None and have_max >= end_ts - pd.Timedelta(minutes=5):
        return
    fetch_start = have_max if have_max is not None else pd.Timestamp(start, tz="UTC")
    fetch_start_str = fetch_start.strftime("%Y-%m-%d %H:%M:%S")
    # data_fetcher._parse_dt는 'YYYY-MM-DD' 또는 'YYYY-MM-DD HH:MM:SS'만 허용 → 초 포함으로 정규화
    end_str = end_ts.strftime("%Y-%m-%d %H:%M:%S") if end else None
    print(f"  ⏬ [{coin.upper()}] 데이터 부족 (보유 최신={have_max} < 요청 {end_ts}) "
          f"→ 자동 갱신: {fetch_start_str} ~ {end_str or 'now'}")
    fetch_ohlcv(symbol=_SYMBOLS[coin], timeframe="5m", start_date=fetch_start_str, end_date=end_str)


def _load(runner, coin: str, start: str, end):
    """ensure_data_coverage(자동 갱신) 후 runner.load_coin 호출하는 래퍼."""
    ensure_data_coverage(coin, start, end)
    return runner.load_coin(coin, start=start, end=end)


def _save_charts(label: str, mode: str, df: pd.DataFrame,
                 res: dict, initial_capital: float = 10_000.0) -> None:
    """서브플롯 구성:
      1) 종가 + 롱 진입/청산 마커
      2) 종가 + 숏 진입/청산 마커
      3) 전략 수익 vs 바이홀드
    """
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    trade_log    = res["trade_log"]
    equity_curve = res["equity_curve"]
    closes       = df["close"].values
    timestamps   = df.index

    bh_ret    = (closes / closes[0] - 1) * 100

    eq = np.array(equity_curve)
    if len(eq) < len(timestamps):
        eq = np.pad(eq, (0, len(timestamps) - len(eq)), mode="edge")
    else:
        eq = eq[:len(timestamps)]
    strat_ret = (eq / initial_capital - 1) * 100

    longs  = [t for t in trade_log if t.get("direction") ==  1
              and t.get("entry") is not None and t.get("exit") is not None]
    shorts = [t for t in trade_log if t.get("direction") == -1
              and t.get("entry") is not None and t.get("exit") is not None]

    fig, axes = plt.subplots(3, 1, figsize=(14, 13),
                             gridspec_kw={"hspace": 0.50})
    date_fmt = mdates.DateFormatter("%m/%d %H:%M")
    date_loc = mdates.AutoDateLocator()

    def _plot_trades(ax, trades, title_prefix, entry_color, exit_color, direction):
        ax.plot(timestamps, closes, color="#4a90d9", linewidth=0.8, label="Close")
        # 마커 크기(s=70) 기준 오프셋: 반지름 sqrt(70)/2 pt × 70% → 데이터 좌표 변환
        ax.relim(); ax.autoscale_view()
        ylo, yhi = ax.get_ylim()
        ax_h_pts = ax.get_position().height * fig.get_figheight() * 72
        offset   = (np.sqrt(70) / 2 * 1.50) * (yhi - ylo) / ax_h_pts
        for t in trades:
            entry_px = t["entry"]; exit_px = t["exit"]
            # exit_ts / entry_ts 기반으로 정확한 봉 인덱스 결정
            if t.get("exit_ts") is not None:
                xi = int(timestamps.searchsorted(t["exit_ts"]))
                xi = min(xi, len(closes) - 1)
            else:
                xi = int(np.argmin(np.abs(closes - exit_px)))
            if t.get("entry_ts") is not None:
                ei = int(timestamps.searchsorted(t["entry_ts"]))
                ei = min(ei, xi)
            else:
                ei = int(np.argmin(np.abs(closes[:xi] - entry_px))) if xi > 0 else 0
            pnl_val  = t.get("pnl", 0)
            pnl_sign = "+" if pnl_val >= 0 else ""
            pnl_pct  = f"{pnl_sign}{pnl_val*100:.2f}%"
            pnl_color = "#27ae60" if pnl_val >= 0 else "#c0392b"
            # 롱: 진입 ^ 아래, 청산 v 위  |  숏: 진입 v 위, 청산 ^ 아래
            if direction == 1:
                e_y = entry_px - offset; e_marker = "^"
                x_y = exit_px  + offset; x_marker = "v"
                ann_xytext = (4, 6)
            else:
                e_y = entry_px + offset; e_marker = "v"
                x_y = exit_px  - offset; x_marker = "^"
                ann_xytext = (4, -10)
            ax.annotate("", xy=(timestamps[xi], exit_px),
                        xytext=(timestamps[ei], entry_px),
                        arrowprops=dict(arrowstyle="-|>", color=entry_color,
                                        lw=1.2, alpha=0.5))
            ax.scatter(timestamps[ei], e_y, marker=e_marker,
                       color=entry_color, s=70, zorder=6, alpha=0.85)
            ax.scatter(timestamps[xi], x_y, marker=x_marker,
                       color=exit_color,  s=70, zorder=6, alpha=0.85)
            ax.annotate(pnl_pct,
                        xy=(timestamps[xi], x_y),
                        xytext=ann_xytext, textcoords="offset points",
                        fontsize=7, color=pnl_color, alpha=0.9,
                        fontweight="bold")
        n = len(trades)
        wins = sum(1 for t in trades if t.get("pnl", 0) >= 0)
        wr_str = f"  WR={wins}/{n}" if n else "  (없음)"
        ax.set_title(
            f"{label} {title_prefix}  "
            f"({timestamps[0].strftime('%m/%d')} ~ {timestamps[-1].strftime('%m/%d')})"
            f"{wr_str}",
            fontsize=11,
        )
        ax.set_ylabel("Price (USDT)")
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(date_loc)
        plt.setp(ax.get_xticklabels(), rotation=20)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # ── 서브플롯 1: 롱 포지션 ─────────────────────────────────────────────────
    _plot_trades(axes[0], longs,  "롱 포지션", "#2ecc71", "#e74c3c", direction=1)

    # ── 서브플롯 2: 숏 포지션 ─────────────────────────────────────────────────
    _plot_trades(axes[1], shorts, "숏 포지션", "#e74c3c", "#2ecc71", direction=-1)

    # ── 서브플롯 3: 전략 수익 vs 바이홀드 ────────────────────────────────────
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
    parser.add_argument("--mode",        default="2026", choices=["2026", "hist", "jun0814", "jun1522", "jun", "daily"])
    parser.add_argument("--windows",     type=int, default=10)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--window-days", type=int, default=91)
    parser.add_argument("--model",       default=_env_model,
                        help=f"AFEnsemble 저장 디렉터리 (기본: .env ML_MODEL_DIR 또는 {_DEFAULT_MODEL})")
    parser.add_argument("--leverage",    type=int, default=_env_lev,
                        help=f"레버리지 (기본: .env LEVERAGE 또는 7)")
    parser.add_argument("--hide-details", action="store_true", default=False,
                        help="거래 상세 로그 출력 생략 (Jun 18~19, Jun 20~22, Jun 모드에서만 적용)")
    parser.add_argument("--skip-charts", action="store_true", default=False,
                        help="차트 저장 생략 (Jun 18~19, Jun 20~22, Jun 모드에서만 적용)")
    args = parser.parse_args()

    # live_trader.py와 동일: AF_PARAM_PRESET 프리셋 적용
    af_params = {**DEFAULT_PARAMS, "leverage": args.leverage}
    preset_name = os.getenv("AF_PARAM_PRESET", "").lower()
    # preset_name = 'aggressive'
    if preset_name and preset_name != "prod" and preset_name in _PRESET_DEFS:
        af_params.update(get_preset(preset_name))
    af_params["leverage"] = args.leverage  # LEVERAGE는 프리셋보다 우선

    runner = AntifragileBacktestRunner.from_saved(args.model, params=af_params)
    # live_trader.py와 동일: .env ML_THRESHOLD 로 ensemble threshold 오버라이드 (후보 θ=0.45 반영)
    _ml_th = os.getenv("ML_THRESHOLD")
    if _ml_th:
        runner.ensemble.threshold = float(_ml_th)
    print(f"[ML 필터] 로드 완료: {args.model}  theta={runner.ensemble.threshold:.3f}")
    print(f"[설정]    LEVERAGE={args.leverage}  AF_PARAM_PRESET={preset_name or 'prod(기본)'}  (.env LEVERAGE: {os.getenv('LEVERAGE', '미설정')})")

    coins = ["btc", "eth", "sol", "xrp"] if args.coin == "all" else [args.coin]

    for coin in coins:
        label = coin.upper()
        print(f"\n{'█'*66}")
        print(f"  {label}/USDT — live_trader 완벽 모방 백테스트 [ML 필수]")
        print(f"  BB σ={_BB_SIGMA}  ML theta={runner.ensemble.threshold:.3f}")
        print(f"  AF_PARAM_PRESET={preset_name or 'prod(기본)'}  (.env LEVERAGE: x{args.leverage})")
        print(f"{'█'*66}")

        if args.mode == "2026":
            df, df_ml = _load(runner, coin, start="2026-01-01", end="2026-06-01")
            if len(df) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df)}봉)"); continue
            days = (df.index[-1] - df.index[0]).days
            print(f"\n[{label}] 2026 OOS: {df.index[0].date()} ~ {df.index[-1].date()}  ({days}일)")
            res = runner.run(df, df_ml)
            runner.print_result(f"{label} 2026 OOS", res, days)
            if not args.skip_charts:
                _save_charts(label, "2026", df, res)

        elif args.mode == "hist":
            hs = HIST_START.get(coin, "2020-01-01")
            print(f"\n[{label}] 랜덤 히스토리 검증")
            runner.run_hist_validation(coin, seed=args.seed, windows=args.windows,
                                       window_days=args.window_days, hist_start=hs)

        elif args.mode == "jun0814":
            df, df_ml = _load(runner, coin, start="2026-06-08 00:00", end="2026-06-14 23:59")
            if len(df) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df)}봉)"); continue
            days = (df.index[-1] - df.index[0]).total_seconds() / 86400
            print(f"\n[{label}] Jun 08~14 실거래 기간")
            res = runner.run(df, df_ml)
            runner.print_result(f"{label} Jun08~14", res, days)
            if not args.hide_details:
                print(f"\n  상세 거래:")
                for t in res["trade_log"]:
                    dr   = "롱" if t["direction"] == 1 else "숏"
                    flip = " ◀FLIP" if t.get("reason") == "reverse_flip" else ""
                    print(f"    {dr}  pnl={t['pnl']*100:+.3f}%  {t.get('reason','')}{flip}")
            if not args.skip_charts:
                _save_charts(label, "jun0814", df, res)

        elif args.mode == "jun1522":
            df, df_ml = _load(runner, coin, start="2026-06-15 00:00", end="2026-06-22 23:59")
            if len(df) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df)}봉)"); continue
            days = (df.index[-1] - df.index[0]).total_seconds() / 86400
            print(f"\n[{label}] Jun 15~22 실거래 기간")
            res = runner.run(df, df_ml)
            runner.print_result(f"{label} Jun15~22", res, days)
            if not args.hide_details:
                print(f"\n  상세 거래:")
                for t in res["trade_log"]:
                    dr   = "롱" if t["direction"] == 1 else "숏"
                    flip = " ◀FLIP" if t.get("reason") == "reverse_flip" else ""
                    print(f"    {dr}  pnl={t['pnl']*100:+.3f}%  {t.get('reason','')}{flip}")
            if not args.skip_charts:
                _save_charts(label, "jun1522", df, res)

        elif args.mode == "jun":
            df, df_ml = _load(runner, coin, start="2026-06-01 00:00", end="2026-06-22 17:00")
            if len(df) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df)}봉)"); continue
            days = (df.index[-1] - df.index[0]).total_seconds() / 86400
            print(f"\n[{label}] Jun 01~22 실거래 기간")
            res = runner.run(df, df_ml)
            runner.print_result(f"{label} Jun01~22", res, days)
            if not args.hide_details:
                print(f"\n  상세 거래:")
                for t in res["trade_log"]:
                    dr   = "롱" if t["direction"] == 1 else "숏"
                    flip = " ◀FLIP" if t.get("reason") == "reverse_flip" else ""
                    print(f"    {dr}  pnl={t['pnl']*100:+.3f}%  {t.get('reason','')}{flip}")
            if not args.skip_charts:
                _save_charts(label, "jun", df, res)

        elif args.mode == "daily":
            df, df_ml = _load(runner, coin, start="2026-06-23 07:20", end="2026-06-24 02:00")
            if len(df) < 50:
                print(f"  ⚠️ 데이터 부족 ({len(df)}봉)"); continue
            days = (df.index[-1] - df.index[0]).total_seconds() / 86400
            print(f"\n[{label}] daily 실거래 기간")
            res = runner.run(df, df_ml)
            runner.print_result(f"{label} daily", res, days)
            if args.hide_details:
                print(f"\n  상세 거래:")
                for t in res["trade_log"]:
                    dr   = "롱" if t["direction"] == 1 else "숏"
                    flip = " ◀FLIP" if t.get("reason") == "reverse_flip" else ""
                    print(f"    {dr}  pnl={t['pnl']*100:+.3f}%  {t.get('reason','')}{flip}")
            if args.skip_charts:
                _save_charts(label, "daily", df, res)


if __name__ == "__main__":
    main()
