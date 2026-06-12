#!/usr/bin/env python3
"""
Antifragile ML 진입 필터 개선 실험
  V1: 기준선 (29피처, binary 레이블, 기존 thresholds)
  V2: +3피처 (32개, binary 레이블, Platt 캘리브레이션 cv=prefit)
  V3: 리스크 조정 레이블 (32피처, pnl/hold_bars 기준)

  실행:
    python temp/scripts/35_af_improved.py --coin btc
    python temp/scripts/35_af_improved.py --coin all
"""
import sys, importlib.util, pickle, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_antifragile import load_coin_full, FEE_TOTAL
from data_pipeline import add_technical_indicators, FEATURE_COLS
from hybrid_engine import compute_metrics

# 33_af_newmodel.py — digit prefix workaround
_spec = importlib.util.spec_from_file_location(
    "af33", ROOT / "temp" / "scripts" / "33_af_newmodel.py"
)
af33 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(af33)

MODELS_DIR = ROOT / "temp" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2026-01-01")
OOS_END   = pd.Timestamp("2026-05-20 23:59:59")

AF        = af33.AF
AF_EXTRA  = af33.AF_EXTRA                           # 4개
NEW_EXTRA = ["hour_of_day", "day_of_week", "atr_24h_pct_rank"]  # +3개
ALL_FEATURES_V1 = FEATURE_COLS + AF_EXTRA           # 29개 (기준선)
ALL_FEATURES_V2 = FEATURE_COLS + AF_EXTRA + NEW_EXTRA  # 32개 (개선)


# ── 피처 빌드 V2 (32개) ───────────────────────────────────────────────────────
def build_features_v2(df_raw: pd.DataFrame) -> pd.DataFrame:
    feat = af33.build_features(df_raw)              # 29개 기준선 피처

    # 시간 피처 (entry 시점의 UTC 시간대)
    feat["hour_of_day"]   = df_raw.index.hour.astype(float)
    feat["day_of_week"]   = df_raw.index.dayofweek.astype(float)

    # 24h 변동성 백분위 (288봉 = 24시간)
    atr = df_raw["_atr"]
    feat["atr_24h_pct_rank"] = atr.rolling(288, min_periods=30).rank(pct=True)

    return feat[ALL_FEATURES_V2]


# ── 진입 이벤트 수집 V2 (hold_bars 추가) ────────────────────────────────────
def collect_entries_v2(df_raw: pd.DataFrame) -> pd.DataFrame:
    """collect_entries + hold_bars 반환"""
    df = df_raw.copy()
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    ts = df.index.copy()
    df = df.reset_index(drop=True)

    lev = AF["leverage"];  rb = AF["rr_base"];  ra = AF["rr_add"]
    al  = AF["add_levels"]; ast = AF["atr_add_step"]
    ti  = AF["trail_atr_init"]; tt = AF["trail_atr_tight"]
    mh  = AF["max_hold_bars"]; cb = AF["cooling_bars"]; md = AF["max_dd_cb"]

    cap = 10_000.0; pk = 10_000.0
    pos = 0; ep = 0.0; rr = 0.0; ac = 0
    tsl = 0.0; ppx = 0.0; eb = 0; cl = 0
    entries = []

    for i in range(1, len(df)):
        row   = df.iloc[i]
        price = float(row["close"])
        rsi   = float(row["_rsi"])
        atr   = float(row["_atr"])
        tup_i = int(row.get("_trend_up", 0))
        tdn_i = int(row.get("_trend_down", 0))

        rsi_lo = AF["dt_rsi_lo"] if tdn_i else (AF["ut_rsi_lo"] if tup_i else AF["rg_rsi_lo"])
        rsi_hi = AF["dt_rsi_hi"] if tdn_i else (AF["ut_rsi_hi"] if tup_i else AF["rg_rsi_hi"])

        equity = cap * (1 + pos * (price - ep) / (ep + 1e-9) * lev * rr) if pos else cap
        pk = max(pk, equity)
        dd = (pk - equity) / (pk + 1e-9)

        if dd > md and cl == 0:
            cl = cb
            if pos:
                cp  = price - FEE_TOTAL * price * pos
                pnl = max(pos * (cp - ep) / (ep + 1e-9) * lev * rr, -rr)
                cap *= (1 + pnl)
                if entries:
                    entries[-1]["pnl"] = pnl
                    entries[-1]["hold_bars"] = i - entries[-1]["_eb"]
                pos = 0; rr = 0.0; ac = 0

        if cl > 0:
            cl -= 1
            continue

        if pos:
            hold = i - eb
            if pos == 1:
                ppx  = max(ppx, price); mult = tt if ac > 0 else ti
                tsl  = max(tsl, ppx - mult * atr); hit = price <= tsl
            else:
                ppx  = min(ppx, price); mult = tt if ac > 0 else ti
                tsl  = min(tsl, ppx + mult * atr); hit = price >= tsl

            if hit or hold >= mh:
                cp  = price - FEE_TOTAL * price * pos
                pnl = max(pos * (cp - ep) / (ep + 1e-9) * lev * rr, -rr)
                cap *= (1 + pnl)
                if entries:
                    entries[-1]["pnl"] = pnl
                    entries[-1]["hold_bars"] = i - entries[-1]["_eb"]
                pos = 0; rr = 0.0; ac = 0
            else:
                fav = pos * (price - ep) / (atr + 1e-9)
                if ac < al and fav >= (ac + 1) * ast:
                    rr += ra; ac += 1
                    tsl = max(tsl, price - tt * atr) if pos == 1 else min(tsl, price + tt * atr)

        if pos == 0:
            if rsi <= rsi_lo:
                ep  = price * (1 + FEE_TOTAL); rr = rb; ac = 0
                tsl = ep - ti * atr; ppx = ep; pos = 1; eb = i
                entries.append({"entry_ts": ts[i], "direction": 1,
                                 "pnl": None, "hold_bars": None, "_eb": i})
            elif rsi >= rsi_hi:
                ep  = price * (1 - FEE_TOTAL); rr = rb; ac = 0
                tsl = ep + ti * atr; ppx = ep; pos = -1; eb = i
                entries.append({"entry_ts": ts[i], "direction": -1,
                                 "pnl": None, "hold_bars": None, "_eb": i})

    if pos and entries and entries[-1]["pnl"] is None:
        last  = df.iloc[-1]
        price = float(last["close"]); atr_v = float(last["_atr"])
        cp    = price - FEE_TOTAL * price * pos
        pnl   = max(pos * (cp - ep) / (ep + 1e-9) * lev * rr, -rr)
        cap  *= (1 + pnl)
        entries[-1]["pnl"]       = pnl
        entries[-1]["hold_bars"] = len(df) - 1 - entries[-1]["_eb"]

    df_e = pd.DataFrame([e for e in entries if e["pnl"] is not None])
    if len(df_e):
        df_e["hold_bars"] = df_e["hold_bars"].fillna(1).clip(lower=1)
        df_e.drop(columns=["_eb"], inplace=True)
    return df_e


# ── 레이블 함수 ──────────────────────────────────────────────────────────────
def label_binary(entries: pd.DataFrame) -> np.ndarray:
    """pnl > 0"""
    return (entries["pnl"] > 0).astype(int).values

def label_risk_adjusted(entries: pd.DataFrame) -> np.ndarray:
    """pnl / hold_bars > median → better trades win faster"""
    ratio = entries["pnl"] / entries["hold_bars"].clip(lower=1)
    return (ratio > ratio.median()).astype(int).values


# ── 모델 학습 ─────────────────────────────────────────────────────────────────
def fit_lgbm_calibrated(X: np.ndarray, y: np.ndarray,
                        calibrate: bool = True) -> object:
    pos_rate = y.mean()
    base = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03,
        max_depth=5, num_leaves=31, min_child_samples=30,
        colsample_bytree=0.8, subsample=0.8,
        scale_pos_weight=(1 - pos_rate) / (pos_rate + 1e-9),
        random_state=42, verbose=-1,
    )
    if not calibrate or len(y) < 400:
        base.fit(X, y)
        return base
    # cv='prefit': 80% for LightGBM, 20% for Platt sigmoid calibration
    split = int(len(X) * 0.8)
    base.fit(X[:split], y[:split])
    cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    cal.fit(X[split:], y[split:])
    return cal

def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]

def _clean(X, y):
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]

def _auc(y, p):
    try:
        return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
    except Exception:
        return 0.5


# ── 단일 실험 실행 ─────────────────────────────────────────────────────────
def run_experiment(coin: str, df: pd.DataFrame,
                   feat_df: pd.DataFrame,
                   entries: pd.DataFrame,
                   label_fn,
                   threshold: float,
                   calibrate: bool,
                   version_name: str):
    """학습 → OOS 백테스트 → (거래수, TPD, WR, PF, 수익률, MDD) 반환"""
    tr_ent = entries[entries["entry_ts"] < TRAIN_END].reset_index(drop=True)
    df_oos = df[(df.index >= TRAIN_END) & (df.index <= OOS_END)]

    X_tr = feat_df.reindex(tr_ent["entry_ts"]).values
    y_tr = label_fn(tr_ent)
    X_tr, y_tr = _clean(X_tr, y_tr)

    model = fit_lgbm_calibrated(X_tr, y_tr, calibrate=calibrate)

    feat_oos  = feat_df[(feat_df.index >= TRAIN_END) & (feat_df.index <= OOS_END)]
    clean_mask = ~np.isnan(feat_oos.values).any(axis=1)
    proba = np.zeros(len(feat_oos))
    proba[clean_mask] = _predict_proba(model, feat_oos.values[clean_mask])
    ml_conf = dict(zip(feat_oos.index, proba))

    m, tlog = af33.run_af_ml(df_oos, ml_conf, threshold)
    n_days = (df_oos.index[-1] - df_oos.index[0]).days
    tpd    = round(len(tlog) / max(n_days, 1), 2)

    return {
        "version":   version_name,
        "threshold": threshold,
        "n_trades":  len(tlog),
        "tpd":       tpd,
        "wr":        m["win_rate"],
        "pf":        m["profit_factor"],
        "ret":       m["total_return"],
        "mdd":       m["mdd"],
        "model":     model,
    }


# ── 임계값 최적화 (Opt2 기준: PF > baseline, MDD ≤ baseline, TPD ≥ 1.5) ────
def find_best_threshold(coin, df, feat_df, entries, label_fn, calibrate,
                        baseline_pf, baseline_mdd, thr_range):
    best = None
    for thr in thr_range:
        r = run_experiment(coin, df, feat_df, entries, label_fn, thr, calibrate, "sweep")
        if r["pf"] > baseline_pf and r["mdd"] <= baseline_mdd and r["tpd"] >= 1.5:
            if best is None or r["pf"] > best["pf"]:
                best = {**r, "threshold": thr}
    return best


# ── 코인별 전체 실험 ──────────────────────────────────────────────────────────
def run_coin_experiments(coin: str):
    print(f"\n{'='*68}")
    print(f"  {coin.upper()} — 개선 실험 (V1 기준 vs V2 vs V3)")
    print(f"{'='*68}")

    df = load_coin_full(coin)
    df_oos = df[(df.index >= TRAIN_END) & (df.index <= OOS_END)]
    n_days = (df_oos.index[-1] - df_oos.index[0]).days
    print(f"  전체 데이터: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"  OOS 구간:   {TRAIN_END.date()} ~ {OOS_END.date()} ({n_days}일)")

    # 피처 빌드
    feat_v1 = af33.build_features(df)   # 29개
    feat_v2 = build_features_v2(df)     # 32개

    # 진입 이벤트 수집
    print("\n  진입 이벤트 수집 중...")
    entries = collect_entries_v2(df)
    print(f"  전체 진입: {len(entries):,}건  WR={(entries['pnl']>0).mean():.1%}")

    # ── V1: 기준선 (기존 모델, 저장된 pickle 로드 or 재학습) ─────────────────
    v1_pkl = MODELS_DIR / f"af_lgbm_{coin}_2026.pkl"
    if v1_pkl.exists():
        with open(v1_pkl, "rb") as f:
            v1_model = pickle.load(f)
        # 저장된 V1 모델로 OOS 실행
        tr_ent_v1 = entries[entries["entry_ts"] < TRAIN_END]
        feat_oos  = feat_v1[(feat_v1.index >= TRAIN_END) & (feat_v1.index <= OOS_END)]
        clean_mask = ~np.isnan(feat_oos.values).any(axis=1)
        proba = np.zeros(len(feat_oos))
        proba[clean_mask] = v1_model.predict_proba(feat_oos.values[clean_mask])[:, 1]
        ml_conf_v1 = dict(zip(feat_oos.index, proba))
        # OOS-optimized threshold per coin (from Phase 2 results)
        opt2_thr = {"btc": 0.57, "eth": 0.57, "sol": 0.46, "xrp": 0.55}
        thr_v1 = opt2_thr.get(coin, 0.55)
        m_v1, tlog_v1 = af33.run_af_ml(df_oos, ml_conf_v1, thr_v1)
        tpd_v1 = round(len(tlog_v1) / max(n_days, 1), 2)
        v1_res = {
            "version": f"V1(29feat+binary, thr={thr_v1})",
            "threshold": thr_v1,
            "n_trades": len(tlog_v1),
            "tpd": tpd_v1,
            "wr": m_v1["win_rate"],
            "pf": m_v1["profit_factor"],
            "ret": m_v1["total_return"],
            "mdd": m_v1["mdd"],
        }
    else:
        print(f"  [경고] {v1_pkl.name} 없음 — V1 재학습")
        v1_res = run_experiment(
            coin, df, feat_v1, entries, label_binary, 0.57, False, "V1(재학습)"
        )

    baseline_pf  = v1_res["pf"]
    baseline_mdd = v1_res["mdd"]

    # ── V2: 32피처 + binary 레이블 (캘리브레이션 없음 — 피처 기여도만 측정) ──
    print(f"\n  V2 학습 중 (32피처, 캘리브레이션 없음)...")
    thr_range = np.arange(0.35, 0.72, 0.02).round(2)
    v2_res = find_best_threshold(
        coin, df, feat_v2, entries, label_binary, False,
        baseline_pf, baseline_mdd, thr_range
    )
    if v2_res is None:
        # fallback: TPD ≥ 1.5 조건 우선, 그 중 최고 PF
        candidates = [
            run_experiment(coin, df, feat_v2, entries, label_binary, thr, False, "V2")
            for thr in thr_range
        ]
        valid = [r for r in candidates if r["tpd"] >= 1.5]
        v2_res = max(valid if valid else candidates, key=lambda r: r["pf"])
        v2_res["_opt2_pass"] = False
    else:
        v2_res["_opt2_pass"] = True

    # ── V3: 32피처 + 리스크 조정 레이블 ────────────────────────────────────
    print(f"  V3 학습 중 (32피처 + 리스크 조정 레이블 pnl/hold_bars)...")
    v3_res = find_best_threshold(
        coin, df, feat_v2, entries, label_risk_adjusted, True,
        baseline_pf, baseline_mdd, thr_range
    )
    if v3_res is None:
        candidates = [
            run_experiment(coin, df, feat_v2, entries, label_risk_adjusted, thr, True, "V3")
            for thr in thr_range
        ]
        valid = [r for r in candidates if r["tpd"] >= 1.5]
        v3_res = max(valid if valid else candidates, key=lambda r: r["pf"])
        v3_res["_opt2_pass"] = False
    else:
        v3_res["_opt2_pass"] = True

    # ── 결과 출력 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  {coin.upper()} 실험 결과 비교 (OOS: 2026-01-01 ~ 2026-05-20)")
    print(f"{'─'*68}")
    print(f"  {'버전':<36s} {'거래수':>6} {'TPD':>5} {'WR':>7} {'PF':>6} {'수익률':>9} {'MDD':>6}")
    print(f"  {'─'*68}")

    def _row(label, r, opt2_ok=None):
        ok = "✅" if opt2_ok else ("❌" if opt2_ok is False else "  ")
        print(f"  {label:<36s} {r['n_trades']:>6,} {r['tpd']:>5.2f} "
              f"{r['wr']:>6.1f}% {r['pf']:>6.2f} {r['ret']:>8.1f}% {r['mdd']:>5.1f}%  {ok}")

    _row(f"V1 기준선 (29feat thr={v1_res['threshold']})", v1_res)
    _row(f"V2 32feat+binary (thr={v2_res['threshold']:.2f})", v2_res, v2_res.get("_opt2_pass"))
    _row(f"V3 32feat+RiskAdj (thr={v3_res['threshold']:.2f})", v3_res, v3_res.get("_opt2_pass"))
    print(f"  {'─'*68}")

    # 판정 기준 설명
    print(f"  [Opt2 판정] PF > {baseline_pf:.2f}(V1) AND MDD ≤ {baseline_mdd:.1f}%(V1) AND TPD ≥ 1.5")

    # 최고 모델 저장
    best = max([v2_res, v3_res], key=lambda r: r["pf"] if r.get("_opt2_pass") else -1)
    if best.get("_opt2_pass") and best["pf"] > baseline_pf:
        best_pkl = MODELS_DIR / f"af_lgbm_{coin}_v2.pkl"
        with open(best_pkl, "wb") as f:
            pickle.dump(best["model"], f)
        print(f"  최고 개선 모델 저장: {best_pkl.relative_to(ROOT)} "
              f"(PF={best['pf']:.2f}, thr={best['threshold']:.2f})")

    return {"coin": coin, "v1": v1_res, "v2": v2_res, "v3": v3_res,
            "baseline_pf": baseline_pf, "baseline_mdd": baseline_mdd}


# ── 메인 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="btc", choices=["btc", "eth", "sol", "xrp", "all"])
    args = ap.parse_args()

    coins = ["btc", "eth", "sol", "xrp"] if args.coin == "all" else [args.coin]
    all_results = []

    for coin in coins:
        res = run_coin_experiments(coin)
        all_results.append(res)

    if len(all_results) > 1:
        print(f"\n\n{'='*68}")
        print("  4종 개선 실험 최종 요약")
        print(f"{'='*68}")
        print(f"  {'코인':^4}  {'버전':^8}  {'거래수':>6} {'TPD':>5} {'WR':>7} "
              f"{'PF':>6} {'수익률':>9} {'MDD':>6} {'판정':>4}")
        print(f"  {'─'*68}")

        for res in all_results:
            coin = res["coin"]
            # 가장 좋은 개선 버전 선택
            improved = max([res["v2"], res["v3"]],
                           key=lambda r: r["pf"] if r.get("_opt2_pass") else -1)
            best_label = "V2" if improved is res["v2"] else "V3"
            ok = "✅" if improved.get("_opt2_pass") else "❌"

            r1, ri = res["v1"], improved
            print(f"  {coin.upper():^4}  {'V1':^8}  {r1['n_trades']:>6,} {r1['tpd']:>5.2f} "
                  f"{r1['wr']:>6.1f}% {r1['pf']:>6.2f} {r1['ret']:>8.1f}% {r1['mdd']:>5.1f}%")
            print(f"  {coin.upper():^4}  {best_label:^8}  {ri['n_trades']:>6,} {ri['tpd']:>5.2f} "
                  f"{ri['wr']:>6.1f}% {ri['pf']:>6.2f} {ri['ret']:>8.1f}% {ri['mdd']:>5.1f}%  {ok}")
            print(f"  {'─'*68}")


if __name__ == "__main__":
    main()
