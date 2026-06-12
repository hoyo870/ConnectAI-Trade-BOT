#!/usr/bin/env python3
"""
Visualize BTC 2026 top/bottom pnl antifragile trades.

Outputs:
  temp/charts/chart_btc_top5pct.png
  temp/charts/chart_btc_bottom5pct.png
"""
import importlib.util
import ast
import sys
import types
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_antifragile import FEE_TOTAL, load_coin_full


START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-05-20 23:59:59")
CHART_DIR = ROOT / "temp" / "charts"
MODEL_SCRIPT = ROOT / "temp" / "scripts" / "33_af_newmodel.py"


def load_collect_entries():
    """Load collect_entries from temp/scripts/33_af_newmodel.py."""
    if "lightgbm" not in sys.modules:
        try:
            import lightgbm  # noqa: F401
        except ModuleNotFoundError:
            lgb_stub = types.ModuleType("lightgbm")

            class _MissingLightGBM:
                def __init__(self, *args, **kwargs):
                    raise ModuleNotFoundError(
                        "lightgbm is required only for model training, not collect_entries()"
                    )

            lgb_stub.LGBMClassifier = _MissingLightGBM
            sys.modules["lightgbm"] = lgb_stub

    spec = importlib.util.spec_from_file_location("af_newmodel_33", MODEL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {MODEL_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.collect_entries, module.AF
    except (ModuleNotFoundError, TypeError):
        return load_collect_entries_minimal()


def load_collect_entries_minimal():
    """
    Extract only AF and collect_entries when optional training imports fail.

    33_af_newmodel.py imports LightGBM and data_pipeline at module scope, but
    collect_entries depends only on pandas/numpy/FEE_TOTAL/AF.
    """
    tree = ast.parse(MODEL_SCRIPT.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "AF" in names:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "collect_entries":
            selected.append(node)

    if len(selected) != 2:
        raise ImportError(f"Could not extract AF and collect_entries from {MODEL_SCRIPT}")

    minimal = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(minimal)
    namespace = {"pd": pd, "np": np, "FEE_TOTAL": FEE_TOTAL}
    exec(compile(minimal, str(MODEL_SCRIPT), "exec"), namespace)
    return namespace["collect_entries"], namespace["AF"]


def collect_trade_events_with_exits(df_raw: pd.DataFrame, af: dict) -> pd.DataFrame:
    """
    Replay the same collect_entries simulation and add exit_ts/marker prices.

    temp/scripts/33_af_newmodel.py intentionally returns only entry_ts,
    direction, pnl. This helper preserves that behavior and recovers the exit
    timestamp needed for chart markers without modifying the source module.
    """
    df = df_raw.copy()
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    ts = df.index.copy()
    df = df.reset_index(drop=True)

    lev = af["leverage"]
    rb = af["rr_base"]
    ra = af["rr_add"]
    al = af["add_levels"]
    ast = af["atr_add_step"]
    ti = af["trail_atr_init"]
    tt = af["trail_atr_tight"]
    mh = af["max_hold_bars"]
    cb = af["cooling_bars"]
    md = af["max_dd_cb"]

    cap = 10_000.0
    pk = 10_000.0
    pos = 0
    ep = 0.0
    rr = 0.0
    ac = 0
    tsl = 0.0
    ppx = 0.0
    eb = 0
    cl = 0
    trades = []

    def close_trade(i: int, forced: bool = False) -> None:
        nonlocal cap, pos, rr, ac
        price = float(df.iloc[i]["close"])
        cp = price - FEE_TOTAL * price * pos
        pnl = max(pos * (cp - ep) / (ep + 1e-9) * lev * rr, -rr)
        cap *= 1 + pnl
        if trades:
            trades[-1].update(
                {
                    "exit_ts": ts[i],
                    "exit_price": price,
                    "pnl": pnl,
                    "forced": forced,
                }
            )
        pos = 0
        rr = 0.0
        ac = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        rsi = float(row["_rsi"])
        atr = float(row["_atr"])
        tup_i = int(row.get("_trend_up", 0))
        tdn_i = int(row.get("_trend_down", 0))

        rsi_lo = af["dt_rsi_lo"] if tdn_i else (af["ut_rsi_lo"] if tup_i else af["rg_rsi_lo"])
        rsi_hi = af["dt_rsi_hi"] if tdn_i else (af["ut_rsi_hi"] if tup_i else af["rg_rsi_hi"])

        equity = cap * (1 + pos * (price - ep) / (ep + 1e-9) * lev * rr) if pos else cap
        pk = max(pk, equity)
        dd = (pk - equity) / (pk + 1e-9)

        if dd > md and cl == 0:
            cl = cb
            if pos:
                close_trade(i, forced=True)

        if cl > 0:
            cl -= 1
            continue

        if pos:
            hold = i - eb
            if pos == 1:
                ppx = max(ppx, price)
                mult = tt if ac > 0 else ti
                tsl = max(tsl, ppx - mult * atr)
                hit = price <= tsl
            else:
                ppx = min(ppx, price)
                mult = tt if ac > 0 else ti
                tsl = min(tsl, ppx + mult * atr)
                hit = price >= tsl

            if hit or hold >= mh:
                close_trade(i)
            else:
                fav = pos * (price - ep) / (atr + 1e-9)
                if ac < al and fav >= (ac + 1) * ast:
                    rr += ra
                    ac += 1
                    tsl = max(tsl, price - tt * atr) if pos == 1 else min(tsl, price + tt * atr)

        if pos == 0:
            if rsi <= rsi_lo:
                ep = price * (1 + FEE_TOTAL)
                rr = rb
                ac = 0
                tsl = ep - ti * atr
                ppx = ep
                pos = 1
                eb = i
                trades.append(
                    {
                        "entry_ts": ts[i],
                        "entry_price": price,
                        "direction": 1,
                        "pnl": None,
                        "forced": False,
                    }
                )
            elif rsi >= rsi_hi:
                ep = price * (1 - FEE_TOTAL)
                rr = rb
                ac = 0
                tsl = ep + ti * atr
                ppx = ep
                pos = -1
                eb = i
                trades.append(
                    {
                        "entry_ts": ts[i],
                        "entry_price": price,
                        "direction": -1,
                        "pnl": None,
                        "forced": False,
                    }
                )

    if pos and trades and trades[-1]["pnl"] is None:
        close_trade(len(df) - 1, forced=True)

    return pd.DataFrame([t for t in trades if t["pnl"] is not None])


def select_tail_trades(trades: pd.DataFrame, side: str, pct: float = 0.05) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    n = max(1, int(np.ceil(len(trades) * pct)))
    ascending = side == "bottom"
    return trades.sort_values("pnl", ascending=ascending).head(n).sort_values("entry_ts")


def add_trade_markers(ax, trades: pd.DataFrame) -> None:
    long_entries = trades[trades["direction"] == 1]
    short_entries = trades[trades["direction"] == -1]

    ax.scatter(
        long_entries["entry_ts"],
        long_entries["entry_price"],
        marker="^",
        s=80,
        color="#16a34a",
        edgecolor="black",
        linewidth=0.5,
        zorder=4,
        label="Long entry",
    )
    ax.scatter(
        short_entries["entry_ts"],
        short_entries["entry_price"],
        marker="v",
        s=80,
        color="#dc2626",
        edgecolor="black",
        linewidth=0.5,
        zorder=4,
        label="Short entry",
    )
    ax.scatter(
        trades["exit_ts"],
        trades["exit_price"],
        marker="x",
        s=80,
        color="#111827",
        linewidth=1.5,
        zorder=5,
        label="Exit",
    )

    for _, trade in trades.iterrows():
        ax.plot(
            [trade["entry_ts"], trade["exit_ts"]],
            [trade["entry_price"], trade["exit_price"]],
            color="#6b7280",
            linewidth=0.8,
            alpha=0.65,
            zorder=2,
        )
        ax.annotate(
            f"{trade['pnl'] * 100:+.2f}%",
            xy=(trade["exit_ts"], trade["exit_price"]),
            xytext=(6, 8 if trade["pnl"] >= 0 else -14),
            textcoords="offset points",
            fontsize=8,
            color="#166534" if trade["pnl"] >= 0 else "#991b1b",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        )


def plot_chart(df: pd.DataFrame, trades: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.plot(df.index, df["close"], color="#2563eb", linewidth=1.0, label="BTC close")

    if not trades.empty:
        add_trade_markers(ax, trades)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("BTCUSDT")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    collect_entries, af = load_collect_entries()

    btc = load_coin_full("btc")
    btc_2026 = btc.loc[(btc.index >= START) & (btc.index <= END)].copy()
    if btc_2026.empty:
        raise ValueError(f"No BTC data found from {START.date()} to {END.date()}")

    entries = collect_entries(btc_2026)
    if entries.empty:
        raise ValueError("collect_entries() returned no closed trades for the selected period")

    trades = collect_trade_events_with_exits(btc_2026, af)
    if len(trades) != len(entries):
        raise ValueError(
            f"Replay trade count mismatch: collect_entries={len(entries)}, replay={len(trades)}"
        )

    trades["pnl"] = entries["pnl"].astype(float).values
    trades["direction"] = entries["direction"].astype(int).values
    trades["entry_ts"] = pd.to_datetime(entries["entry_ts"]).values

    top5 = select_tail_trades(trades, "top")
    bottom5 = select_tail_trades(trades, "bottom")

    plot_chart(
        btc_2026,
        top5,
        f"BTC 2026 Top 5% Trades by PnL ({len(top5)} of {len(trades)})",
        CHART_DIR / "chart_btc_top5pct.png",
    )
    plot_chart(
        btc_2026,
        bottom5,
        f"BTC 2026 Bottom 5% Trades by PnL ({len(bottom5)} of {len(trades)})",
        CHART_DIR / "chart_btc_bottom5pct.png",
    )

    print(f"BTC rows: {len(btc_2026):,}")
    print(f"Closed trades: {len(trades):,}")
    print(f"Top 5% chart: {CHART_DIR / 'chart_btc_top5pct.png'}")
    print(f"Bottom 5% chart: {CHART_DIR / 'chart_btc_bottom5pct.png'}")


if __name__ == "__main__":
    main()
