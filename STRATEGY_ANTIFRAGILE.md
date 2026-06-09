# Antifragile Trailing Stop 전략

> 승격일: 2026-06-03 | 검증: BTC 10/10 · ETH 10/10 · SOL 10/10 · XRP 10/10  
> 백테스트 스크립트: `scripts/backtest_antifragile.py`  
> 활성화: `.env`에 `STRATEGY=antifragile` → 4종목 자동 25%씩 분할 매매  
> Trail 파라미터 업데이트: 0.5/0.8 → **1.0/1.5** (2026-06-09 적용)

---

## 핵심 원리

**기존 전략의 실패 원인**: 모든 rule-based 전략이 `avg_win < avg_loss` (PF < 1) 구조적 문제 보유.  
BB_mid 청산이 수익을 일찍 잘라냄 → 손실이 이익보다 항상 큼.

**Antifragile 해법**: 청산 방식을 바꿔 수학적으로 역전.

```
avg_win  = +0.875% (BTC) / +1.040% (ETH)  [trail 1.0/1.5 기준]
avg_loss = −0.124% (BTC) / −0.188% (ETH)
비율     =   7.1x  (BTC) /   5.5x  (ETH)
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
초기 SL 거리: entry_price ± 1.0 × ATR(14)
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

**trail_mult 값 (2026-06-09 업데이트):**
- 피라미딩 전: `trail_atr_init = 1.0` (초기 stop 거리 확대 → 2봉 즉시 손절 감소)
- 피라미딩 후: `trail_atr_tight = 1.5` (피라미딩 후 이익 보호)

price가 trail_stop에 닿으면 즉시 시장가 청산.

### 4. 피라미딩 — Antifragile 포지션 증가

진입 후 가격이 유리하게 이동할 때마다 포지션을 추가한다.

```
진입 후 +0.5×ATR 이동 시: rr += 0.15 (1회 추가)
진입 후 +1.0×ATR 이동 시: rr += 0.15 (2회 추가)
진입 후 +1.5×ATR 이동 시: rr += 0.15 (3회 추가)
최대 rr = 0.10 + 3×0.15 = 0.55
```

피라미딩 후 trailing stop을 타이트(1.5×ATR)하게 조정 → 이익 보호.

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
| `trail_atr_init` | **1.0** | 초기 trailing stop 거리 (ATR 배수) ← 0.5에서 변경 |
| `trail_atr_tight` | **1.5** | 피라미딩 후 trailing stop 거리 ← 0.8에서 변경 |
| `rr_base` | 0.10 | 초기 자본 위험 비율 |
| `rr_add` | 0.15 | 피라미딩 1회당 추가 rr |
| `add_levels` | 3 | 최대 피라미딩 횟수 |
| `atr_add_step` | 0.5 | 피라미딩 트리거 (ATR 배수) |
| `leverage` | 3 | 레버리지 |
| `max_hold_bars` | 288 | 최대 보유 (1일) |

---

## 검증 결과

### 섹션 1: 2026 OOS 백테스트 (기준 vs 신규 파라미터)

> 기간: BTC/SOL/XRP 2026-01-01~05-20, ETH 2026-01-01~05-30  
> 스크립트: `temp/scripts/38_trail_param_test.py` (2026-06-09 실행)

| 코인 | 기준(0.5/0.8) 수익률 | 신규(1.0/1.5) 수익률 | 개선율 | TPD(기준→신규) | MDD(기준→신규) | PF(기준→신규) |
|------|---------------------|---------------------|--------|---------------|---------------|--------------|
| BTC | +226.4% | **+269.1%** | +18.9% ✅ | 8.27→6.05 | 3.07%→2.82% | 8.44→7.07 |
| ETH | +574.9% | +526.0% | -8.5% ❌ | 7.15→5.34 | 3.36%→3.83% | 7.86→5.54 |
| SOL | +710.9% | **+932.0%** | +31.1% ✅ | 7.15→5.36 | 2.82%→5.60% | 8.39→7.24 |
| XRP | +348.2% | **+494.7%** | +42.1% ✅ | 7.34→5.37 | 3.39%→7.63% | 6.95→6.21 |

**트레이드오프 요약**:
- ✅ 수익률 BTC/SOL/XRP 개선 (+19~42%)
- ✅ avgWin 증가 (더 큰 추세를 잡음)
- ⚠ ETH 소폭 열세 (-8.5%): 2026 ETH 구조적 하락장 특성
- ⚠ TPD 27% 감소 (8.3→6.1): 거래 수 줄고 건당 수익 증가
- ⚠ SOL/XRP MDD 증가: 넓은 stop이 큰 반전 구간에서 더 많이 노출됨

### 섹션 2: 역사적 검증 (3개월 창 × 10회, seed=42, 2021~2025)

| 코인 | 기준(0.5/0.8) | 신규(1.0/1.5) | 신규 우세 창 |
|------|-------------|-------------|------------|
| BTC | 10/10, avg +293.7% | 10/10, avg **+341.4%** | 9/10 |
| ETH | 10/10, avg +368.9% | 10/10, avg **+451.0%** | 9/10 |
| SOL | 10/10, avg +1,401% | 10/10, avg **+1,885%** | 8/10 |
| XRP | 10/10, avg +5,843% | 10/10, avg **+14,797%** | 7/10 |

> XRP 신규 평균이 큰 이유: 2020-12~2021-03 역사적 랠리 포함 (+123,128%). 해당 창 제외 시 XRP 신규 평균 약 +1,500%/3개월.  
> 두 파라미터 모두 전 코인 10/10 수익 달성. 신규가 수익률 측면에서 대부분 우위.

### 섹션 3: Paper Trading 실측 + 백테스트 비교 (2026-06-03~06-09, 6일)

| 코인 | 실측(기준 0.5/0.8) | 백테스트(기준 0.5/0.8) | 백테스트(신규 1.0/1.5) |
|------|-------------------|----------------------|----------------------|
| BTC | +20.2% (PF 17.10, TPD 3.83) | +12.8% (PF 9.23) | **+24.8%** (PF 11.12) |
| ETH | +11.6% (PF 5.89, TPD 5.67) | +8.7% (PF 6.53) | **+10.7%** (PF 6.21) |
| SOL | +31.6% (PF 14.80, TPD 5.17) | +24.2% (PF 11.48) | **+35.0%** (PF 9.35) |
| XRP | +13.3% (PF 8.96, TPD 4.67) | **+23.5%** (PF 14.99) | +22.4% (PF 9.76) |

> 실측 vs 백테스트 괴리 원인: paper trading은 5분봉 마감가 기준 신호, 백테스트는 동일 방식이나 봉 경계 타이밍 미세 차이 발생 가능.  
> XRP는 신규 파라미터가 유일하게 기준보다 소폭 낮음 (백테스트 기준, 0.5% 차이).

### 판정 기준 (trailing stop 전략 전용)

trailing stop 전략은 WR이 구조적으로 낮으므로 아래 기준 적용:

| 기준 | 값 |
|------|-----|
| 수익률 | > 0% |
| TPD | ≥ 1.5 |
| Top-5 제거 후 수익 | > 0% |

---

## 활성화 방법

### .env 필수 설정

```env
STRATEGY=antifragile    # 필수 — 없으면 dl_v17로 폴백
TRADE_MODE=paper        # paper | sandbox | real
EXCHANGE=bybit          # bybit | bingx
PAPER_SEED=10000        # paper 모드 전용 (종목당 2,500 USDT 자동 분할)
```

### paper 모드 — 4종목 동시 (단일 프로세스)

```bash
python src/live_trader.py
# 또는 백그라운드
nohup python src/live_trader.py > logs/paper_multi.log 2>&1 &
```

**생성 파일:**
```
logs/paper_multi.log           ← 공유 로그 (4종목 통합)
logs/paper_state.json          ← BTC 상태 · paper_trades.csv
logs/paper_state_eth.json      ← ETH 상태 · paper_trades_eth.csv
logs/paper_state_sol.json      ← SOL 상태 · paper_trades_sol.csv
logs/paper_state_xrp.json      ← XRP 상태 · paper_trades_xrp.csv
```

### 거래소 선택 (EXCHANGE)

| 거래소 | sandbox 지원 | 심볼 형식 | 비고 |
|--------|------------|---------|------|
| bybit | ✅ | `BTC/USDT:USDT` | 기본값 |
| bingx | ❌ (자동 paper 전환) | `BTC/USDT:USDT` | USDT 선물 잔고 이체 필요 |

### 전략 전환 시 주의사항

1. 기존 봇 중지: `kill <PID>`
2. 열린 포지션 없는지 확인 (거래소 대시보드)
3. `.env`에 `STRATEGY=antifragile` 설정 확인
4. 상태파일 초기화 (선택): `rm logs/paper_state*.json`
5. 봇 재시작 (COIN 환경변수 불필요)

### 백테스트 재실행

```bash
# 4종목 전체 2026 검증
python scripts/backtest_antifragile.py --coin all --mode 2026

# 역사적 랜덤 검증 (4종목)
python scripts/backtest_antifragile.py --coin all --mode random --seed 42

# Trail 파라미터 비교 검증
python temp/scripts/38_trail_param_test.py
```

---

## DL v17 전략과 비교

| 항목 | DL v17 (기존) | Antifragile (신규) |
|------|-------------|------------------|
| 진입 신호 | TCN+Attention DL 모델 | AdaptRSI (rule-based) |
| SL/TP | 고정 (SL=2%, TP=6%) | ATR trailing stop |
| 포지션 크기 | 티어 기반 (2.5x~10x) | 피라미딩 (0.1~0.55) |
| 레버리지 | 2.5x~10x | 3x 고정 |
| 2026 BTC 수익 | +181% | **+269%** |
| BTC Top-5 제거 | **-18.9%** (outlier 의존) | **+127%** (강건) |
| ETH 2026 수익 | +164.8% | **+526%** |
| hist 통과 | BTC 6/10 | **전 코인 10/10** |
| MDD | 57% (BTC) | **2.8%** (BTC) |

---

## 파라미터 변경 이력

| 날짜 | 변경 항목 | 이전 값 | 변경 값 | 근거 |
|------|---------|--------|--------|------|
| 2026-06-03 | 전략 승격 | — | trail 0.5/0.8 | hist 9~10/10 통과 |
| 2026-06-09 | trail_atr_init | 0.5 | **1.0** | 2봉 즉시손절 감소, 수익률 +19~42% 개선 |
| 2026-06-09 | trail_atr_tight | 0.8 | **1.5** | 피라미딩 후 더 긴 추세 포착 |
| 2026-06-09 | 거래소 | bybit only | **bybit + bingx** | 거래소 다변화 |
