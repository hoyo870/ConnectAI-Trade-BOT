# Antifragile 제로베이스 ML 진입 신호 모델 — 계획서

> 작성일: 2026-06-03 | 상태: 대기 중 (진행 준비 시 TODO_af_newmodel.md 참고)

---

## 목표

기존 모델 완전 유지. 진입 신호만 ML로 교체하여 가능성 검증.

```
[기존 Antifragile]  entry = AdaptRSI (규칙 기반)  → exit = ATR trail  → sizing = pyramiding
[신규 ML 모델]      entry = LightGBM/TCN (학습)   → exit = ATR trail  → sizing = pyramiding
```

**핵심**: exit(trailing stop) + sizing(pyramiding)은 그대로. 진입 신호만 교체.  
**제약**: `src/` 미수정, `models/production/` 절대 미접촉, `temp/` 폴더만 사용.

---

## DL v17 실패와 이번 접근의 차이

| 항목 | DL v17 (실패) | 이번 ML 모델 |
|------|-------------|-------------|
| exit 구조 | 고정 SL/TP → avg_win < avg_loss | ATR trailing stop → avg_win >> avg_loss |
| 레이블 | 미래 가격 방향 예측 | trailing stop 결과 (이기는 진입 예측) |
| PF | < 1 (구조적 문제) | 기존 7-10 유지 or 개선 기대 |
| top-5 제거 | -18.9% | 기대: 여전히 양수 |

---

## 구현 파일

**파일 1개**: `temp/scripts/33_af_newmodel.py`

### 내부 구조

#### Step 1 — 레이블 생성
```python
import sys
sys.path.insert(0, 'scripts')
from backtest_antifragile import load_coin_full, COIN_CONFIG, AF_PARAMS

# Antifragile 백테스트 실행 → 각 진입 이벤트 추출
# label = 1 if pnl > 0 else 0
```
예상 레이블 수: BTC ~16,400건 / 4코인 합산 ~50,000건

#### Step 2 — 피처 추출
```python
sys.path.insert(0, 'src')
from data_pipeline import add_technical_indicators, FEATURE_COLS
```

기존 23개 피처 + AF-specific 추가:
- `rsi_dist_lo` = RSI - 동적하한 (얼마나 깊이 눌렸나)
- `rsi_dist_hi` = 동적상한 - RSI (얼마나 과열됐나)
- `atr_pct_20d` = ATR 20일 백분위
- `1h_ema_slope` = 1h 추세 강도
- `symbol_id` = 코인 구분 (4종 통합 시)

#### Step 3 — 모델 학습

**Phase 1 (BTC만, ~5-10분): LightGBM**
- walk-forward 3-fold (시간순, 랜덤 분할 금지)
- 저장: `temp/models/af_lgbm_btc.pkl`

**Phase 2 (BTC 검증 통과 시 4종 + TCN)**
- `temp/models/af_lgbm_{coin}.pkl` × 4개
- `temp/models/af_tcn_all.pt` (TCN 통합 모델)

#### Step 4 — 결과 출력

```
=== BTC 진입 신호 비교 (test: 2025~2026) ===
                    거래수   WR      PF      수익률   MDD
AdaptRSI (기존):    1,850  28.6%   8.45   +226%    3.1%
LightGBM (신규):    ????   ????    ????   ????     ????

=== Feature Importance Top-10 ===
  rsi_dist_lo:  0.21
  atr_pct_20d:  0.15
  ...
```

---

## 판단 기준

| 조건 | 기준 |
|------|------|
| PF 향상 | ≥ 원본 × 1.10 (최소 10%) |
| 거래수 유지 | ≥ 원본의 30% (fat-tail 제거 방지) |
| 일관성 | walk-forward 3-fold 모두 향상 |

**판정 통과 시**: 7일 paper 완료 후 `STRATEGY=af_ml` 옵션 추가 검토  
**판정 실패 시**: ML 진입 신호 방향 폐기, Antifragile 유지

---

## 실행 명령

```bash
# Phase 1: BTC 1종 검증 (~5-10분)
/opt/homebrew/Caskroom/miniforge/base/envs/cryptobot/bin/python \
  temp/scripts/33_af_newmodel.py --coin btc

# Phase 2: 4종 전체 (~15-25분, Phase 1 통과 후)
/opt/homebrew/Caskroom/miniforge/base/envs/cryptobot/bin/python \
  temp/scripts/33_af_newmodel.py --coin all
```

---

## 파일 구조 (완료 후)

```
temp/
├── PLAN_af_newmodel.md       ← 이 파일
├── TODO_af_newmodel.md       ← TODO 체크리스트
├── scripts/
│   └── 33_af_newmodel.py     ← 신규 생성 (기존 파일 삭제 후)
└── models/                    ← 자동 생성
    ├── af_lgbm_btc.pkl
    ├── af_lgbm_eth.pkl
    ├── af_lgbm_sol.pkl
    ├── af_lgbm_xrp.pkl
    └── af_tcn_all.pt
```
