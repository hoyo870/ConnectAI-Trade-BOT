#!/usr/bin/env python3
"""
Antifragile ML 진입 신호 모델 검증
  Phase 1 (BTC):    python temp/scripts/33_af_newmodel.py --coin btc
  Phase 2 (4종):    python temp/scripts/33_af_newmodel.py --coin all

구조: exit(ATR trailing) + sizing(pyramiding) 동일, 진입 신호만 LightGBM으로 교체
레이블: Antifragile trailing stop 결과 → pnl > 0 = 1 (승리)
"""
import sys, argparse, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_antifragile import load_coin_full, FEE_TOTAL
from data_pipeline import add_technical_indicators, FEATURE_COLS
from hybrid_engine import compute_metrics

MODELS_DIR = ROOT / "temp" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── AF 파라미터 (backtest_antifragile.py 기본값과 동일) ───────────────────────
AF = dict(
    leverage=3, rr_base=0.10, rr_add=0.15, add_levels=3,
    atr_add_step=0.5, trail_atr_init=0.5, trail_atr_tight=0.8,
    max_hold_bars=288, cooling_bars=100, max_dd_cb=0.30,
    dt_rsi_lo=22, dt_rsi_hi=65,
    rg_rsi_lo=30, rg_rsi_hi=70,
    ut_rsi_lo=35, ut_rsi_hi=78,
)

AF_EXTRA      = ["rsi_dist_lo", "rsi_dist_hi", "atr_pct_rank", "trend_dir"]
ALL_FEATURES  = FEATURE_COLS + AF_EXTRA  # 25 + 4 = 29개


# ── 피처 빌드 ─────────────────────────────────────────────────────────────────
def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + AF 지표 df → ALL_FEATURES 컬럼 DataFrame (원본 index 유지)"""
    feat = add_technical_indicators(df_raw)  # FEATURE_COLS 25개 추가

    tup = df_raw["_trend_up"].fillna(0).astype(float)
    tdn = df_raw["_trend_down"].fillna(0).astype(float)
    rsi = df_raw["_rsi"].fillna(50.0)
    atr = df_raw["_atr"]

    rsi_lo = np.where(tdn, AF["dt_rsi_lo"], np.where(tup, AF["ut_rsi_lo"], AF["rg_rsi_lo"]))
    rsi_hi = np.where(tdn, AF["dt_rsi_hi"], np.where(tup, AF["ut_rsi_hi"], AF["rg_rsi_hi"]))

    feat["rsi_dist_lo"]  = rsi.values - rsi_lo          # 음수 = 과매도 깊이
    feat["rsi_dist_hi"]  = rsi_hi - rsi.values           # 음수 = 과매수 깊이
    feat["atr_pct_rank"] = atr.rolling(576, min_periods=50).rank(pct=True)
    feat["trend_dir"]    = tup.values - tdn.values        # 1=상승, -1=하락, 0=횡보

    return feat[ALL_FEATURES]


# ── 진입 이벤트 수집 (run_antifragile 재구현 + entry_ts 기록) ─────────────────
def collect_entries(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Antifragile 시뮬레이션 실행 → 진입 이벤트 DataFrame (entry_ts, direction, pnl)"""
    df = df_raw.copy()
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    ts = df.index.copy()          # timestamp 보존 (reset_index 전)
    df = df.reset_index(drop=True)

    lev = AF["leverage"];   rb = AF["rr_base"];       ra = AF["rr_add"]
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
                cp   = price - FEE_TOTAL * price * pos
                pnl  = max(pos * (cp - ep) / (ep + 1e-9) * lev * rr, -rr)
                cap *= (1 + pnl)
                if entries: entries[-1]["pnl"] = pnl; entries[-1]["forced"] = True
                pos = 0; rr = 0.0; ac = 0

        if cl > 0:
            cl -= 1
            continue

        # trailing stop + pyramiding
        if pos:
            hold = i - eb
            if pos == 1:
                ppx  = max(ppx, price)
                mult = tt if ac > 0 else ti
                tsl  = max(tsl, ppx - mult * atr)
                hit  = price <= tsl
            else:
                ppx  = min(ppx, price)
                mult = tt if ac > 0 else ti
                tsl  = min(tsl, ppx + mult * atr)
                hit  = price >= tsl

            if hit or hold >= mh:
                cp   = price - FEE_TOTAL * price * pos
                pnl  = max(pos * (cp - ep) / (ep + 1e-9) * lev * rr, -rr)
                cap *= (1 + pnl)
                if entries: entries[-1]["pnl"] = pnl
                pos = 0; rr = 0.0; ac = 0
            else:
                fav = pos * (price - ep) / (atr + 1e-9)
                if ac < al and fav >= (ac + 1) * ast:
                    rr += ra; ac += 1
                    tsl = max(tsl, price - tt * atr) if pos == 1 else min(tsl, price + tt * atr)

        # 신규 진입
        if pos == 0:
            if rsi <= rsi_lo:
                ep  = price * (1 + FEE_TOTAL); rr = rb; ac = 0
                tsl = ep - ti * atr; ppx = ep; pos = 1; eb = i
                entries.append({"entry_ts": ts[i], "direction": 1, "pnl": None, "forced": False})
            elif rsi >= rsi_hi:
                ep  = price * (1 - FEE_TOTAL); rr = rb; ac = 0
                tsl = ep + ti * atr; ppx = ep; pos = -1; eb = i
                entries.append({"entry_ts": ts[i], "direction": -1, "pnl": None, "forced": False})

    # 미결 포지션 강제 청산
    if pos and entries and entries[-1]["pnl"] is None:
        last = df.iloc[-1]
        price = float(last["close"]); atr = float(last["_atr"])
        cp   = price - FEE_TOTAL * price * pos
        pnl  = max(pos * (cp - ep) / (ep + 1e-9) * lev * rr, -rr)
        cap *= (1 + pnl)
        entries[-1]["pnl"] = pnl

    return pd.DataFrame([e for e in entries if e["pnl"] is not None])


# ── AF 백테스트 with ML 신뢰도 필터 ──────────────────────────────────────────
def run_af_ml(df_raw: pd.DataFrame, ml_conf: dict | None, threshold: float):
    """
    ml_conf: {timestamp: win_probability} dict — None이면 필터 없음 (원본 AF)
    threshold: 이 값 이상일 때만 진입 허용
    """
    df = df_raw.copy()
    df.dropna(subset=["_rsi", "_atr"], inplace=True)
    ts = df.index.copy()
    df = df.reset_index(drop=True)

    lev = AF["leverage"];   rb = AF["rr_base"];       ra = AF["rr_add"]
    al  = AF["add_levels"]; ast = AF["atr_add_step"]
    ti  = AF["trail_atr_init"]; tt = AF["trail_atr_tight"]
    mh  = AF["max_hold_bars"]; cb = AF["cooling_bars"]; md = AF["max_dd_cb"]

    cap = 10_000.0; pk = 10_000.0
    pos = 0; ep = 0.0; rr = 0.0; ac = 0
    tsl = 0.0; ppx = 0.0; eb = 0; cl = 0
    eq = [cap]; tlog = []

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
                cp   = price - FEE_TOTAL * price * pos
                pnl  = max(pos * (cp - ep) / (ep + 1e-9) * lev * rr, -rr)
                cap *= (1 + pnl)
                tlog.append({"pnl": pnl, "direction": pos})
                pos = 0; rr = 0.0; ac = 0

        if cl > 0:
            cl -= 1; eq.append(cap); continue

        if pos:
            hold = i - eb
            if pos == 1:
                ppx  = max(ppx, price); mult = tt if ac > 0 else ti
                tsl  = max(tsl, ppx - mult * atr); hit = price <= tsl
            else:
                ppx  = min(ppx, price); mult = tt if ac > 0 else ti
                tsl  = min(tsl, ppx + mult * atr); hit = price >= tsl

            if hit or hold >= mh:
                cp   = price - FEE_TOTAL * price * pos
                pnl  = max(pos * (cp - ep) / (ep + 1e-9) * lev * rr, -rr)
                cap *= (1 + pnl)
                tlog.append({"pnl": pnl, "direction": pos})
                pos = 0; rr = 0.0; ac = 0
            else:
                fav = pos * (price - ep) / (atr + 1e-9)
                if ac < al and fav >= (ac + 1) * ast:
                    rr += ra; ac += 1
                    tsl = max(tsl, price - tt * atr) if pos == 1 else min(tsl, price + tt * atr)

        if pos == 0:
            conf = (ml_conf.get(ts[i], 0.0) if ml_conf is not None else 1.0)
            if rsi <= rsi_lo and conf >= threshold:
                ep  = price * (1 + FEE_TOTAL); rr = rb; ac = 0
                tsl = ep - ti * atr; ppx = ep; pos = 1; eb = i
            elif rsi >= rsi_hi and conf >= threshold:
                ep  = price * (1 - FEE_TOTAL); rr = rb; ac = 0
                tsl = ep + ti * atr; ppx = ep; pos = -1; eb = i

        eq.append(cap)

    m = compute_metrics(eq, tlog)
    if tlog:
        m["tpd"] = round(len(tlog) / (len(df) / 288), 2)
    return m, tlog


# ── LightGBM 학습 ─────────────────────────────────────────────────────────────
def fit_lgbm(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    pos_rate = y.mean()
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03,
        max_depth=5, num_leaves=31, min_child_samples=30,
        colsample_bytree=0.8, subsample=0.8,
        scale_pos_weight=(1 - pos_rate) / (pos_rate + 1e-9),
        random_state=42, verbose=-1,
    )
    model.fit(X, y)
    return model


def _auc(y, p):
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
    except Exception:
        return 0.5


def _clean(X, y):
    """NaN 행 제거"""
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


# ── 코인별 Phase 1 실행 ────────────────────────────────────────────────────────
def run_coin(coin: str, threshold: float = 0.55) -> bool:
    print(f"\n{'='*64}")
    print(f"  {coin.upper()} — Antifragile ML 진입 신호 검증  (threshold={threshold})")
    print(f"{'='*64}")

    # 1. 데이터 로드 (add_indicators 포함 — _rsi, _atr, _trend_up/down 이미 있음)
    df = load_coin_full(coin)

    # 2. ML 피처 빌드
    feat_df = build_features(df)

    # 3. 전체 진입 이벤트 수집
    print(f"\n[Step 1] Antifragile 백테스트 — 진입 이벤트 수집...")
    entries = collect_entries(df)
    wr_all  = (entries["pnl"] > 0).mean()
    print(f"  전체: {len(entries):,}건  WR={wr_all:.1%}  "
          f"롱={( entries['direction']==1).sum():,}  숏={( entries['direction']==-1).sum():,}")

    # 4. 75/25 시간 분할 (train / test)
    test_cutoff = df.index[int(len(df) * 0.75)]
    tr_ent = entries[entries["entry_ts"] <  test_cutoff].reset_index(drop=True)
    te_ent = entries[entries["entry_ts"] >= test_cutoff].reset_index(drop=True)
    df_test = df[df.index >= test_cutoff]
    print(f"  Train: {len(tr_ent):,}건  Test: {len(te_ent):,}건")
    print(f"  테스트 구간: {df_test.index[0].date()} ~ {df_test.index[-1].date()}")

    # 5. walk-forward 3-fold (train 구간만, AUC 안정성 확인)
    print(f"\n[Step 2] Walk-forward 3-fold (train 구간)...")
    n = len(tr_ent)
    fold_sz = n // 4
    fold_aucs = []
    for fold in range(3):
        t_end   = (fold + 1) * fold_sz
        v_start = t_end
        v_end   = min(v_start + fold_sz, n)

        X_tr = feat_df.reindex(tr_ent.loc[:t_end-1,  "entry_ts"]).values
        y_tr = (tr_ent.loc[:t_end-1,  "pnl"] > 0).astype(int).values
        X_va = feat_df.reindex(tr_ent.loc[v_start:v_end-1, "entry_ts"]).values
        y_va = (tr_ent.loc[v_start:v_end-1, "pnl"] > 0).astype(int).values

        X_tr, y_tr = _clean(X_tr, y_tr)
        X_va, y_va = _clean(X_va, y_va)

        m = fit_lgbm(X_tr, y_tr)
        preds = m.predict_proba(X_va)[:, 1]
        auc = _auc(y_va, preds)
        fold_aucs.append(auc)
        print(f"  Fold {fold+1}: train={len(y_tr):,}  val={len(y_va):,}  WR={y_va.mean():.1%}  AUC={auc:.3f}")

    # 6. 비교 모델 학습 (train 75% 전체)
    X_tr_all = feat_df.reindex(tr_ent["entry_ts"]).values
    y_tr_all = (tr_ent["pnl"] > 0).astype(int).values
    X_tr_all, y_tr_all = _clean(X_tr_all, y_tr_all)
    comp_model = fit_lgbm(X_tr_all, y_tr_all)

    # 7. 테스트 구간 ML 신뢰도 계산 (바 단위)
    feat_test  = feat_df[feat_df.index >= test_cutoff]
    clean_mask = ~np.isnan(feat_test.values).any(axis=1)
    proba      = np.zeros(len(feat_test))
    proba[clean_mask] = comp_model.predict_proba(feat_test.values[clean_mask])[:, 1]
    ml_conf = dict(zip(feat_test.index, proba))

    # 8. 원본 AF (테스트 구간)
    print(f"\n[Step 3] 백테스트 비교...")
    m_orig, tl_orig = run_af_ml(df_test, None,    threshold=0.0)

    # 9. ML 필터 AF (테스트 구간)
    m_ml,   tl_ml   = run_af_ml(df_test, ml_conf, threshold=threshold)

    pf_orig = m_orig["profit_factor"]
    pf_ml   = m_ml["profit_factor"]

    # 10. 결과 출력
    print(f"\n{'─'*64}")
    print(f"  {coin.upper()} 결과  (테스트: {df_test.index[0].year} ~ {df_test.index[-1].year})")
    print(f"{'─'*64}")
    print(f"  {'':20s} {'거래수':>7} {'WR':>7} {'PF':>7} {'수익률':>9} {'MDD':>7}")
    print(f"  {'AdaptRSI (기존)':20s} {len(tl_orig):>7,} {m_orig['win_rate']:>6.1f}% "
          f"{pf_orig:>7.2f} {m_orig['total_return']:>8.1f}% {m_orig['mdd']:>6.1f}%")
    print(f"  {'LightGBM (신규)':20s} {len(tl_ml):>7,} {m_ml['win_rate']:>6.1f}% "
          f"{pf_ml:>7.2f} {m_ml['total_return']:>8.1f}% {m_ml['mdd']:>6.1f}%")
    print(f"  Walk-forward AUC: {[f'{a:.3f}' for a in fold_aucs]}  "
          f"평균={np.mean(fold_aucs):.3f}")
    print(f"{'─'*64}")

    # 11. Phase 1 판정
    pf_thr     = pf_orig * 1.10
    trade_thr  = max(int(len(tl_orig) * 0.30), 1)
    ok_pf      = pf_ml >= pf_thr
    ok_trades  = len(tl_ml) >= trade_thr
    passed     = ok_pf and ok_trades

    print(f"\n[Phase 1 판정]")
    print(f"  PF:    {pf_ml:.2f} {'✅' if ok_pf else '❌'}  (기준: ≥{pf_thr:.2f} = {pf_orig:.2f}×1.10)")
    print(f"  거래수: {len(tl_ml):,} {'✅' if ok_trades else '❌'}  "
          f"(기준: ≥{trade_thr:,} = {len(tl_orig):,}×30%)")
    print(f"  최종: {'✅ Phase 2 진행' if passed else '❌ 중단 — Antifragile rule-based 유지'}")

    # 12. Feature Importance
    imp = pd.Series(
        comp_model.feature_importances_, index=ALL_FEATURES
    ).sort_values(ascending=False)
    mx = imp.max()
    print(f"\n[Feature Importance Top-10]")
    for name, v in imp.head(10).items():
        bar = "█" * max(1, int(v / mx * 25))
        print(f"  {name:<25s} {bar}  ({v:.0f})")

    # 13. 모델 저장
    model_path = MODELS_DIR / f"af_lgbm_{coin}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(comp_model, f)
    print(f"\n모델 저장: {model_path.relative_to(ROOT)}")

    return passed


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="btc", choices=["btc", "eth", "sol", "xrp", "all"])
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="ML 신뢰도 임계값 (기본: 0.55)")
    args = ap.parse_args()

    coins = ["btc", "eth", "sol", "xrp"] if args.coin == "all" else [args.coin]
    results = {}

    for coin in coins:
        passed = run_coin(coin, args.threshold)
        results[coin] = passed
        if coin == "btc" and not passed and args.coin != "all":
            print("\nBTC Phase 1 실패 → 중단. Antifragile rule-based 유지.")
            break

    if len(results) > 1:
        print(f"\n{'='*64}")
        print("  4종 결과 요약")
        print(f"{'='*64}")
        for c, p in results.items():
            print(f"  {c.upper():6s}: {'✅ 통과' if p else '❌ 실패'}")


if __name__ == "__main__":
    main()
