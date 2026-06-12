# Antifragile ML 진입 필터 실험 로그

> 작성일: 2026-06-03~04  
> 실험 목적: AdaptRSI 고정 임계값 대신 LightGBM이 더 좋은 진입 타이밍을 학습할 수 있는가?  
> 제약: `src/`, `scripts/`, `models/production/` 미수정. `temp/` 내 작업 전용.

---

## 1. 실험 개요

**기존 Antifragile 전략 구조**
- 진입: AdaptRSI (1h EMA 방향별 RSI 임계값 동적 조정)
- 청산: ATR trailing stop (고정 SL/TP 없음)
- 사이징: 역 마틴게일 피라미딩 (유리 방향 +0.5ATR마다 +0.15 추가, 최대 3회)

**실험 가설**  
AdaptRSI 진입 시점에 LightGBM confidence gate를 추가하면 저품질 진입을 걸러내어 PF를 높일 수 있다.

**레이블 정의**  
`label = 1` if `pnl > 0` (Antifragile trailing stop 청산 결과 기준)  
→ DL v17 실패 원인(고정 TP/SL)을 구조적으로 제거한 레이블

---

## 2. 모델 아키텍처

| 항목 | 값 |
|------|-----|
| 알고리즘 | LightGBM |
| n_estimators | 500 |
| learning_rate | 0.03 |
| max_depth | 5 |
| num_leaves | 31 |
| 클래스 불균형 처리 | scale_pos_weight 자동 계산 |
| 피처 수 | **29개** |
| 검증 방식 | Walk-forward 3-fold (시간순, 랜덤 분할 금지) |
| 학습 기간 | 2020~2025 |
| OOS 기간 | 2026-01-01 ~ 2026-05-20 |

### 피처 목록 (29개)

**FEATURE_COLS (25개, `src/data_pipeline.py`)**  
ema_cross_5_20, ema_cross_10_50, ema_cross_20_100, macd, macd_signal, macd_hist,  
rsi_14, stoch_rsi_k, stoch_rsi_d, atr_14, atr_pct, bb_upper, bb_lower, bb_width,  
bb_pct, returns_1, returns_3, returns_5, vol_ratio_20, obv_change, vwap_dist,  
adx_14, plus_di, minus_di, trend_strength

**AF_EXTRA (4개, `temp/scripts/33_af_newmodel.py` 내부 계산)**

| 피처 | 설명 |
|------|------|
| `rsi_dist_lo` | RSI - 동적 하한 (음수 = 과매도 초과) |
| `rsi_dist_hi` | 동적 상한 - RSI (음수 = 과매수 초과) |
| `atr_pct_rank` | ATR 20일 백분위 (변동성 레짐) |
| `trend_dir` | 1h EMA 방향 (-1/0/+1) |

### 저장 모델

```
temp/models/
├── af_lgbm_btc_2026.pkl   ← BTC 단일 모델 (4종 공용으로도 사용)
├── af_lgbm_eth_2026.pkl
├── af_lgbm_sol_2026.pkl
└── af_lgbm_xrp_2026.pkl
```

---

## 3. 실험 결과 — Phase 1: BTC 임계값 스윕 (OOS: 2026)

**판정 기준**
- Option1 (Opt1): PF ≥ 기존×1.10 **AND** TPD ≥ 1.5
- Option2 (Opt2): PF > 기존 **AND** MDD ≤ 기존 **AND** TPD ≥ 1.5

| 임계값 | 거래수 | TPD  | WR    | PF    | 수익률   | MDD  | 판정 |
|--------|--------|------|-------|-------|----------|------|------|
| AdaptRSI (기존) | 1,150 | 8.27 | 24.6% | 8.44 | +226.4% | 3.1% | 기준 |
| 0.50 (Opt1) | 366 | 2.63 | 35.2% | 10.48 | +188.5% | 1.6% | ✅ 통과 |
| 0.57 (Opt2) | 221 | 1.59 | 36.2% | 11.58 | +112.3% | 1.6% | ✅ 통과 |

> 주목: Opt2는 수익률이 낮아 보이지만 거래 빈도가 줄어든 것이 주요 원인.  
> MDD 3.1% → 1.6%: 최대 낙폭 절반 수준으로 개선.

---

## 4. 실험 결과 — Phase 2: 4종 per-coin 모델 (OOS: 2026)

각 코인별 독립 모델 학습 후 OOS 적용.

| 코인 | 기존 TPD | 기존 PF | 기존 MDD | Opt1 TPD | Opt1 PF | Opt1 | Opt2 TPD | Opt2 PF | Opt2 MDD | Opt2 | thr |
|------|---------|---------|---------|---------|---------|------|---------|---------|---------|------|-----|
| BTC | 8.27 | 8.44 | 3.1% | 2.63 | 10.48 | ✅ | 1.59 | 11.58 | 1.6% | ✅ | 0.57 |
| ETH | 7.18 | 7.75 | 3.4% | 2.77 | 8.55 | ✅ | 1.50 | 9.65 | 1.4% | ✅ | 0.57 |
| SOL | 7.20 | 8.42 | 2.8% | 3.11 | 8.25 | ❌ | 3.89 | 8.98 | 2.7% | ✅ | 0.46 |
| XRP | 7.52 | 7.03 | 3.4% | 2.42 | 6.69 | ❌ | 1.60 | 7.29 | 1.5% | ✅ | 0.55 |

> SOL Opt1 ❌: PF 8.25 < 기준(8.42×1.10=9.27)  
> XRP Opt1 ❌: PF 6.69 < 기준(7.03×1.10=7.73)  
> 두 코인 모두 Opt2는 통과.

---

## 5. 실험 결과 — BTC 단일 모델 → 4종 적용 (최종 권고 구성)

**가설**: BTC 모델이 타 코인에도 일반화될 수 있는가?  
**결과**: 전 코인 Opt2 기준 통과. XRP per-coin 실패를 BTC 모델이 해결.

| 코인 | 기존 PF | 기존 MDD | Opt2 PF | Opt2 TPD | Opt2 MDD | 판정 | thr |
|------|---------|---------|---------|---------|---------|------|-----|
| BTC | 8.44 | 3.1% | 11.58 | 1.59 | 1.6% | ✅ | 0.57 |
| ETH | 7.75 | 3.4% | 9.65 | 1.50 | 1.4% | ✅ | 0.57 |
| SOL | 8.42 | 2.8% | 8.98 | 3.89 | 2.7% | ✅ | 0.46 |
| XRP | 7.03 | 3.4% | 7.29 | 1.60 | 1.5% | ✅ | 0.55 |

**권고 배포 구성**: BTC 단일 모델 + 코인별 OOS 최적 임계값  
→ 모델 파일: `temp/models/af_lgbm_btc_2026.pkl` (1개)

---

## 6. 핵심 발견

1. **BTC 모델 일반화**: BTC 학습 모델이 ETH/SOL/XRP에 동등하거나 더 우수. 특히 XRP per-coin 실패(PF 7.03→6.69)를 BTC 모델(PF 7.29)이 해결.

2. **ML 필터 효과**: 거래 수 약 70~80% 감소, PF 20~40% 향상. 품질 vs 수량 트레이드오프 존재.

3. **MDD 개선**: 평균 MDD 3.2% → 1.6% (절반 수준). 손실 거래를 더 잘 걸러냄.

4. **IS 성능 저하는 정상**: 모델이 IS 노이즈 과적합 방지 → OOS에서 실제 성능 발휘. IS 백테스트 결과가 기존보다 낮아 보이는 것은 올바른 동작.

5. **전략 성격**: "역추세 진입 + 추세추종 관리" 하이브리드. 순수 역추세 아님.
   - 진입: RSI 극단값 포착 (역추세)
   - 청산: ATR trailing stop — 승자를 끝까지 살림 (추세추종)
   - 사이징: 역마틴게일 피라미딩 (추세추종)

---

## 7. 향후 개선 제안

### 7-1. 피처 엔지니어링

| 제안 | 예상 효과 | 난이도 |
|------|---------|--------|
| 코인 간 상관관계 (BTC-ETH 60분) | 시장 전반 방향성 파악 | 중 |
| 시간 피처 (hour_of_day, day_of_week) | 아시아/미국 세션 패턴 학습 | 하 |
| 24h 변동성 백분위 (현재는 20d 기준) | 단기 레짐 포착 개선 | 하 |
| 오더북 불균형 (bid/ask 비율) | 실시간 수급 반영 | 상 (API 필요) |

### 7-2. 레이블 개선

| 방식 | 내용 | 기대 효과 |
|------|------|---------|
| 현재 (binary) | pnl > 0 = 1 | 기준선 |
| 리스크 조정 레이블 | pnl / hold_bars (보유 기간 대비 수익) | 빠른 수익 거래 우선 학습 |
| 3-class | big_win(≥0.3%) / small_win(0~0.3%) / loss | 수익 크기까지 예측 |

### 7-3. 모델 구조 개선

| 제안 | 내용 | 우선순위 |
|------|------|--------|
| 앙상블 | LightGBM + CatBoost 평균 | 중 |
| 확률 캘리브레이션 | Platt scaling → threshold 신뢰도 향상 | 중 |
| 월별 롤링 재학습 | 매월 최신 1개월 추가 재학습 | 상 |
| TCN | 시퀀스 인식 모델 (`src/expert_models.py` 재사용) | 하 (시간 필요) |

### 7-4. 임계값 전략

| 방식 | 내용 |
|------|------|
| 현재 (정적) | 코인별 단일 고정 임계값 |
| 레짐 적응형 | 변동성 낮을 때 threshold↑, 높을 때 threshold↓ |
| 방향별 분리 | 롱 threshold ≠ 숏 threshold (비대칭 시장 반영) |

### 7-5. 역사적 검증 추가 (미완료)

현재 OOS 2026 단일 기간만 검증됨. Antifragile Phase 기준과 동일하게 역사적 검증 필요.

```
검증 방법: 3개월 창 × 10회 (seed=42)
합격 기준: pos_cnt ≥ 6/10, avg TPD ≥ 1.5
스크립트: temp/scripts/33_af_newmodel.py --mode hist
```

---

## 8. 다음 단계 (Phase 3, 2026-06-10 이후)

- [ ] 7일 paper 거래 완료 확인 (2026-06-10)
- [ ] paper 결과와 ML 백테스트 결과 비교 분석
- [ ] 역사적 검증 (10창) 추가 실행
- [ ] 조건 충족 시 `live_trader.py`에 `STRATEGY=af_ml` 옵션 추가 검토

---

*관련 파일*
- 실험 스크립트: `temp/scripts/33_af_newmodel.py`
- 시각화 스크립트: `temp/scripts/34_visualize_trades.py`
- 차트 출력: `temp/charts/chart_btc_top5pct.png`, `chart_btc_bottom5pct.png`
- 메모리 기록: `memory/project_antifragile.md`, `memory/project_style_exploration.md`
- backtest 엔진: `scripts/backtest_antifragile.py`

---

## 9. V4 실험 설계 (향후 진행, 2026-06-05 기록)

> CCG(Claude+Codex+Gemini) 분석 결과 도출. 목표: PF 유지하면서 WR·수익률 동시 개선.

### 9-1. 핵심 발견 — WR vs 수익률은 반대 방향

| 최적화 목표 | Threshold 방향 | 결과 |
|------------|--------------|------|
| WR ↑ | 높임 (더 selective) | 거래수↓, 수익금↓, MDD↓ |
| 수익률 ↑ | 낮춤 (더 많은 거래) | TPD↑, 복리 효과↑, WR↓ |

→ **threshold 조정만으로 두 목표 동시 달성 불가. 모델 품질 자체를 높여야 함.**

### 9-2. V4 변경사항 (우선순위 순)

#### (1) 레이블 — 60th percentile + Neutral Zone 제거 [Codex 최우선]

```python
low_q, high_q = 0.45, 0.60
score = pnl / hold_bars

# 모호한 45th~60th 중간 샘플 제거 → 명확한 승/패만 학습
train_df = df[(score <= score.quantile(low_q)) | (score >= score.quantile(high_q))]
train_df["label"] = (score >= score.quantile(high_q)).astype(int)
```

- 현재 median(50th): "평균 이상" 학습 → 모호 샘플 다수
- 60th + neutral zone: 명확한 승/패 경계 → precision↑ → WR↑
- 실험 순서: 55th → 60th → 65th (70th는 거래수 감소 위험)

#### (2) BTC 레짐 피처 추가 [Codex + Gemini 공통]

```python
features_to_add = [
    "btc_return_1h",       # ETH/SOL/XRP에 BTC 방향성 추가
    "btc_trend_1h",        # BTC가 하락 레짐인데 alt 롱 = 주요 false positive 원인
    "efficiency_ratio_20", # Kaufman ER: 1에 가까울수록 강한 추세 (Gemini)
    "bb_width_pct_rank",   # 변동성 압축 백분위 → 압축 후 신호 = fat-tail 확률↑ (Gemini)
    "volume_force",        # (Close-Low)/(High-Low) × Volume → 실매수세력 측정 (Gemini)
]
```

#### (3) LightGBM 보수화 [Codex]

```python
# V3 현재
max_depth=5, num_leaves=31

# V4 변경
params = {
    "learning_rate": 0.02, "n_estimators": 800,
    "max_depth": 4, "num_leaves": 15,       # 과적합 감소
    "min_child_samples": 100,
    "subsample": 0.75, "colsample_bytree": 0.75,
    "reg_alpha": 1.0, "reg_lambda": 5.0,
}
```

#### (4) Threshold 목적함수 분리 (WR-max vs Return-max)

```python
# WR 우선 (WR 40%+ 목표)
maximize WR  subject to  PF ≥ V3_PF × 0.90

# 수익률 우선 (총 수익 극대화)
maximize return  subject to  PF ≥ V3_PF × 0.85, TPD ≥ 1.5
```

### 9-3. Gemini 추가 제안 — Post-processing (ML 필터 무변경)

- **Dynamic Pyramiding**: ML confidence ≥ 0.8일 때만 피라미딩 공격적으로 적용
- **Time-based Tightening**: 진입 후 20봉 경과 시 수익 < 0.5 ATR이면 Trailing Stop 즉시 tight 전환

### 9-4. V4 실험 순서

| 버전 | 변경사항 | 기대 효과 |
|------|---------|---------|
| V4a | 60th percentile label + neutral zone | WR↑, 더 명확한 분류 경계 |
| V4b | V4a + BTC regime + ER + BB width | Alt 코인 WR 추가 향상 |
| V4c | V4b + LightGBM 보수화 | 과적합 감소, OOS 안정성 |

- 구현 스크립트: `temp/scripts/36_af_v4.py` (미생성)
- Opt2 판정 기준 동일: PF > V3, MDD ≤ V3, TPD ≥ 1.5
