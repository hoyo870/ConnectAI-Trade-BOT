# Antifragile Trailing Stop 전략

> 승격일: 2026-06-03 | 검증: BTC hist 9/10 · ETH hist 10/10  
> 백테스트 스크립트: `scripts/backtest_antifragile.py`  
> 활성화: `.env`에 `STRATEGY=antifragile` 추가

---

## 핵심 원리

**기존 전략의 실패 원인**: 모든 rule-based 전략이 `avg_win < avg_loss` (PF < 1) 구조적 문제 보유.  
BB_mid 청산이 수익을 일찍 잘라냄 → 손실이 이익보다 항상 큼.

**Antifragile 해법**: 청산 방식을 바꿔 수학적으로 역전.

```
avg_win  = +0.671% (BTC) / +0.849% (ETH)
avg_loss = −0.080% (BTC) / −0.109% (ETH)
비율     =   8.4x  (BTC) /   7.8x  (ETH)
```

---

## 매매 규칙

### 1. 진입 — AdaptRSI (적응형 RSI 임계값)

1h EMA(20)의 방향에 따라 RSI 임계값을 동적으로 조정한다.

| 1h 추세 | 롱 진입 조건 | 숏 진입 조건 |
|---------|------------|------------|
| **하락** (close < EMA20) | RSI ≤ **22** | RSI ≥ **65** |
| **횡보** (중립) | RSI ≤ **30** | RSI ≥ **70** |
| **상승** (close > EMA20) | RSI ≤ **35** | RSI ≥ **78** |

**직관**: 하락장에서 롱 진입은 더 깊은 과매도를 요구하고, 숏은 추세에 올라타기 위해 기준을 완화함.

**BB 이탈 조건**: `require_bb=False` — BB 밴드 이탈 없이 RSI 조건만으로 진입.  
(BB 이탈 추가 시 거래수 감소, 수익률 하락)

### 2. 진입 가격 / 레버리지 / 초기 포지션

```
레버리지:     3x (고정)
초기 rr:      0.10  (자본의 10%를 리스크로)
초기 SL 거리: entry_price ± 0.5 × ATR(14)
```

### 3. 청산 — ATR Trailing Stop

고정 SL/TP 없음. trailing stop이 포지션을 추적한다.

**롱 포지션:**
```
peak_price = max(peak_price, current_price)    # 최고가 갱신
trail_stop = peak_price − trail_mult × ATR(14) # 최고가에서 N×ATR 아래
```

**숏 포지션:**
```
peak_price = min(peak_price, current_price)    # 최저가 갱신
trail_stop = peak_price + trail_mult × ATR(14) # 최저가에서 N×ATR 위
```

**trail_mult 값:**
- 피라미딩 전: `trail_atr_init = 0.5` (더 넓게)
- 피라미딩 후: `trail_atr_tight = 0.8` (더 타이트)

price가 trail_stop에 닿으면 즉시 시장가 청산.

### 4. 피라미딩 — Antifragile 포지션 증가

진입 후 가격이 유리하게 이동할 때마다 포지션을 추가한다.

```
진입 후 +0.5×ATR 이동 시: rr += 0.15 (1회 추가)
진입 후 +1.0×ATR 이동 시: rr += 0.15 (2회 추가)
진입 후 +1.5×ATR 이동 시: rr += 0.15 (3회 추가)
최대 rr = 0.10 + 3×0.15 = 0.55
```

피라미딩 후 trailing stop을 더 타이트(0.8×ATR)하게 조정 → 이익 보호.

**결과**: 이기는 거래는 포지션이 커지고, 지는 거래는 처음부터 작게 진입 후 빠르게 cut.

### 5. 최대 보유 시간

288봉 = 1일 초과 시 강제 청산.

---

## 파라미터 (AF_PARAMS)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `dt_rsi_lo` / `dt_rsi_hi` | 22 / 65 | 하락추세 진입 RSI |
| `rg_rsi_lo` / `rg_rsi_hi` | 30 / 70 | 횡보 진입 RSI |
| `ut_rsi_lo` / `ut_rsi_hi` | 35 / 78 | 상승추세 진입 RSI |
| `trail_atr_init` | 0.5 | 초기 trailing stop 거리 (ATR 배수) |
| `trail_atr_tight` | 0.8 | 피라미딩 후 trailing stop 거리 |
| `rr_base` | 0.10 | 초기 자본 위험 비율 |
| `rr_add` | 0.15 | 피라미딩 1회당 추가 rr |
| `add_levels` | 3 | 최대 피라미딩 횟수 |
| `atr_add_step` | 0.5 | 피라미딩 트리거 (ATR 배수) |
| `leverage` | 3 | 레버리지 |
| `max_hold_bars` | 288 | 최대 보유 (1일) |

---

## 검증 결과 (2026-06-03)

### 2026 백테스트

| 코인 | 수익률 | WR | TPD | MDD | PF | Top5 제거 |
|------|-------|-----|-----|-----|-----|---------|
| BTC | +226% | 24.6% | 8.30 | 3.1% | 8.447 | **+127%** |
| ETH | +558% | 29.7% | 7.27 | 3.4% | 7.820 | **+371%** |

> WR 25~30%는 trailing stop 전략에서 정상 (소규모 손실 多, 대규모 이익 少)

### 역사적 랜덤 검증 (seed=42, 3개월 × 10회)

| 코인 | 통과(3/3) | 수익 양수 | 평균 수익 | 최저 수익 |
|------|----------|--------|---------|---------|
| BTC | **9/10** | 10/10 | **+212%** | +10.5% |
| ETH | **10/10** | 10/10 | **+326%** | +56.0% |

### 판정 기준 (일반 전략과 다름)

trailing stop 전략은 WR이 구조적으로 낮으므로 아래 기준 적용:

| 기준 | 값 |
|------|-----|
| 수익률 | > 0% |
| TPD | ≥ 1.5 |
| Top-5 제거 후 수익 | > 0% |

---

## 활성화 방법

### paper 모드 (테스트)

```bash
# .env 수정
STRATEGY=antifragile
TRADE_MODE=paper

# 재시작 (기존 포지션 없는 상태에서)
python src/live_trader.py
```

### 전략 전환 시 주의사항

1. 기존 봇 중지: `kill <PID>`
2. 열린 포지션 없는지 확인
3. `.env`에 `STRATEGY=antifragile` 추가
4. paper_state.json 초기화 (선택): 삭제 또는 유지
5. 봇 재시작

### 백테스트 재실행

```bash
# 2026 검증
python scripts/backtest_antifragile.py --coin btc --mode 2026
python scripts/backtest_antifragile.py --coin eth --mode 2026
python scripts/backtest_antifragile.py --coin both --mode 2026

# 역사적 검증
python scripts/backtest_antifragile.py --coin both --mode random --seed 42

# BB 이탈 포함 버전 (보수적)
python scripts/backtest_antifragile.py --coin both --mode 2026 --require-bb
```

---

## DL v17 전략과 비교

| 항목 | DL v17 (기존) | Antifragile (신규) |
|------|-------------|------------------|
| 진입 신호 | TCN+Attention DL 모델 | AdaptRSI (rule-based) |
| SL/TP | 고정 (SL=2%, TP=6%) | ATR trailing stop |
| 포지션 크기 | 티어 기반 (2.5x~10x) | 피라미딩 (0.1~0.55) |
| 레버리지 | 2.5x~10x | 3x 고정 |
| 2026 BTC 수익 | +181% | **+226%** |
| BTC Top-5 제거 | **-18.9%** (outlier 의존) | **+127%** (강건) |
| ETH 2026 수익 | +164.8% | **+558%** |
| hist 통과 | BTC 6/10 | **BTC 9/10 · ETH 10/10** |
| MDD | 57% (BTC) | **3.1%** (BTC) |
