# ConnectAI Trade Bot — 실행 스케줄

> 시작: 2026-06-03 | Paper 모드 종료/분석: 2026-06-10  
> 소액 실계좌: paper 분석 후 판단

---

## ✅ 완료 (2026-06-03)

- [x] CCG 3회 분석: 완성도 갭, 모델 성능, paper 플랜
- [x] `live_trader.py` 버그 3개 수정
  - seed 재초기화 버그 (`not state_existed` 조건)
  - 원자적 상태 저장 (`os.replace`)
  - 중복 틱 방지 (`last_candle_ts`)
- [x] Paper 모드 재시작 (PID 4455, 자본 9,578 USDT 복원 확인)
- [x] `scripts/analyze_funding.py` — BTC 펀딩비 포함 수익률
  - 결과: BTC +180.7% → **+162.8%** (차이 -18%p, 무시 불가)
- [x] `scripts/analyze_top_trades.py` — outlier 의존도 검증
  - 결과: Top-3 제거까지 양수, **Top-5 제거 시 -18.9%** → outlier 의존 확인

---

## Day 1 오후 (2026-06-03)

### 1-A. Top-5 문제 대응: 임계값 상향 실험 ✅
- [x] `tier4 threshold 0.50 → 0.55` 변경 후 BTC 2026 재백테스트
  - 결과: ≥0.55 시 **-90.95%** (WR 43.4%, PF 0.971) — 오히려 악화
  - ≥0.52 테스트도 +18.5%로 대폭 하락
  - **결론: tier4(0.50~0.55) 거래가 실제 수익에 기여 중, outlier는 T1(10x) 거래에서 발생**
  - **결정: 파라미터 현행 유지, paper 데이터 축적 후 T1 레버리지 캡 재검토**

### 1-B. ETH 펀딩비 정확 분석 ✅
- [x] `scripts/analyze_funding_eth.py` 신규 작성 — raw OHLCV에서 gate 재계산 후 펀딩비 반영
  - 게이트 통과율: long 4.9% / short 10.8% (매우 엄격한 필터)
  - 게이트 없음: **-76.6%** (WR=41.0%, n=232) — 게이트가 전략의 생사를 결정
  - 게이트 포함: **+164.8%** (WR=48.3%, n=174) → 펀딩비 후 **+155.7%** (차이 -9.19%p)
  - 게이트 효과: +241%p — live_trader.py에서 ETH 게이트 로직 정상 작동이 필수

---

## Day 2 (2026-06-04) — 실계좌 인프라 준비

### 2-A. 코드: 일일 손실 한도 + Kill Switch ✅
- [x] `live_trader.py`에 일일 손실 2% 한도 추가
  - `daily_start_capital` / `daily_date` / `daily_halt` 상태 추가
  - UTC 날짜 바뀌면 자동 리셋, 손실 ≥2% 시 포지션 청산 + `daily_halt=True`
- [x] `telegram_notifier.py`에 `poll_commands()` 추가
  - getUpdates polling으로 `/stop` 감지
  - `live_trader.py` 메인 루프에서 매 틱 전 폴링 → 수신 시 포지션 청산 + 봇 종료
- [x] PID 10981로 재시작 확인 (자본 9,578 USDT 복원)

### 2-B. 배포 인프라 ✅
- [x] `deploy/live_trader.service` — Linux systemd (Restart=always, RestartSec=30)
- [x] `deploy/com.connectai.tradebot.plist` — macOS launchd (Crashed 재시작)
- [x] `deploy/run_paper.sh` — nohup 시작 스크립트 (중복 실행 방지 포함)
- [x] `RotatingFileHandler` 적용 (10MB × 5개 보관)

### 2-C. 보안 체크리스트
- [ ] Bybit API: `Withdrawal` 권한 OFF 확인 ← **수동: Bybit 콘솔에서 직접 확인**
- [ ] Bybit API: `Futures Trading` Only 확인 ← **수동: Bybit 콘솔에서 직접 확인**
- [x] `.env` 파일 권한: `chmod 600 .env` ✅ (644→600 수정 완료)
- [ ] IP 화이트리스팅: 봇 실행 서버 IP 등록 (실계좌 전환 시)

---

## Day 3~7 (2026-06-05~11) — 다른 매매 스타일 탐색

> 목표: 기존 BTC DL 모델과 다른 접근법을 5일 동안 탐색하여 비교 가능한 대안 발굴  
> 방법: 기존 데이터(`data/signals_2026/`) 활용, 각 스타일 간단 백테스트 후 결과 기록

### 탐색 대상 스타일

| # | 스타일 | 핵심 아이디어 | 검증 기준 |
|---|--------|-------------|---------|
| A | **변동성 돌파** | 이전 봉 고가/저가 돌파 + ATR 배수 | WR≥55%, PF≥1.3 |
| B | **RSI+BB 역추세** | RSI 과매수/과매도 + BB 밴드 이탈 반전 | MDD<25%, WR≥52% |
| C | **멀티타임프레임** | 1h 추세 방향 + 5m 진입 타점 | TPD≥1.5, hist≥6/10 |
| D | **펀딩비 역이용** | 극단적 펀딩비(±0.1%↑) 시 역방향 진입 | 거래수≥50/월 |
| E | **볼륨 스파이크** | 거래량 이상 급증 + 가격 방향 확인 | WR≥50%, PF≥1.2 |

### 일별 계획

| 일차 | 날짜 | 작업 |
|------|------|------|
| Day 3 | 2026-06-03 | 스타일 A(변동성 돌파) ❌ 탈락 — WR 22~37%, 수익 -28~-31% (모든 파라미터 조합) |
| Day 4 | 2026-06-03 | 스타일 B(RSI+BB) ⚠️ 2/3 — WR=51%, TPD=1.95지만 PF=0.70, 수익 -23% (payoff 역비대칭) |
| Day 5 | 2026-06-03 | 스타일 C(멀티TF 1h EMA gate) ❌ 탈락 — 1h gate가 DL 신호 차단, 기준선보다 악화 |
| Day 6 | 2026-06-03 | 스타일 D(펀딩비 역이용) ❌ — WR 25~41%, 수익 -2~-28%, 8h return proxy는 신호 품질 부족 |
| Day 7 | 2026-06-03 | 스타일 E(볼륨 스파이크) ❌ — WR 30~40%, spike≥5x면 TPD 0.36으로 거래수 부족 |

### 탐색 규칙
- 기존 DL 모델 재학습 금지 (Phase 3 결정 유지)
- 각 스타일은 `temp/scripts/` 에 임시 작성, 통과 시 `scripts/` 승격
- 통과 기준: **WR≥48%, TPD≥1.5, hist pos≥6/10, Top-5 제거 후 양수**

### 스타일 탐색 최종 비교 (2026-06-03 완료)

| 스타일 | WR | TPD | 수익 | PF | Top-5 | 판정 |
|--------|-----|-----|------|----|-------|------|
| A — 변동성 돌파 | 28% | 1.08 | -30% | 1.47 | -38% | ❌ |
| B — RSI+BB 역추세 | 51% | 1.95 | -23% | 0.70 | -33% | ⚠️ 추후 연구 |
| C — 멀티TF 1h gate | 47% | 1.80 | -20% | 1.18 | -68% | ❌ |
| D — 펀딩비 역이용 | 35% | 1.24 | -15% | 1.35 | -22% | ❌ |
| E — 볼륨 스파이크 | 40% | 0.36 | -6% | 1.14 | -14% | ❌ |
| **기준 DL v17** | **50%** | **1.85** | **+181%** | **1.41** | -19% | ⚠️ Top-5 의존 |

**결론: 5가지 rule-based 스타일 모두 기존 DL(v17) 미만. 신규 스타일 추가보다 DL outlier 의존도 개선이 우선.**

---

## Day 8 (2026-06-10) — Paper 1주일 분석

### 분석 항목
- [ ] `logs/paper_trades.csv` 분석
  - 백테스트 동기간 결과와 비교
  - 신호 일치도 (동일 타임스탬프 진입 여부)
  - 진입가 오차 (백테스트 close vs paper close)
  - WR, TPD, 수익률 일치도
- [ ] `paper_state.json` 기준 누적 자본 확인
- [ ] 실계좌 전환 판단: 아래 기준 모두 충족 시 진행

### 실계좌 전환 판단 기준
| 항목 | 기준 | 확인 |
|------|------|------|
| 신호 일치도 | ≥90% | |
| Paper WR | 백테스트 ±5%p 이내 | |
| 1주일 수익 | 양수 또는 -5% 이내 | |
| 코드 안정성 | 7일 무중단 실행 | |
| Kill switch | 텔레그램 `/stop` 작동 확인 | |

---

## CCG 분석 결과 (2026-06-03) — Gemini + Codex 종합

> `/oh-my-claudecode:ccg` 실행: Style B + Pullback 개선 방향 탐색

### CCG 제안 → 구현 결과

| # | 제안 | 출처 | 구현파일 | 결과 |
|---|------|------|---------|------|
| 1 | Staged Exit + ATR Trail (BB_mid 50%/러너) | 양측 합의 | `24_style_b_improved.py` V2 | ❌ WR -13%p 하락, 수익 변화 없음 |
| 2 | DL Confluence (signal≥0.50 진입 필터) | Codex #3 | V3 | ⚠️ WR +6%p 상승, TPD 0.71로 부족 |
| 3 | Regime Filter (ADX<25 ranging only) | Codex #2 | V4 | ❌ 효과 없음 (거래수만 감소) |
| **4** | **Adaptive RSI 임계값** (하락 22/65, 중립 30/70, 상승 35/78) | **Codex #4** | **V5 ← 핵심!** | **✅ +12.9%, WR=60.5%, MDD=10.6%** |
| 5 | BTC Gate (ema9/21 + rsi + vol + 1h trend) | Codex #7 | `23_pullback_improved.py` V2 | ❌ -43%로 악화 (게이트 방향 불일치) |
| 6 | Confirmed Local Peak (rolling_max 4봉) | Codex #5 | V3 | ⚠️ WR 50% MDD 14% 좋지만 n=20 너무 적음 |
| 7 | Signal Decay Exit (감쇠 신호 청산) | Codex #6 | V4 | ❌ pullback은 진입 자체가 감쇠 후 → 즉시 발동 |
| 8 | Meta-Label Classifier | Codex #8 | `25_metalabel_train.py` | 🔄 실행 중 |

### Style B AdaptRSI V5 — 핵심 수치 (BTC 2026)

| 항목 | 기준선(30/70) | AdaptRSI(22/65-30/70-35/78) |
|------|-------------|---------------------------|
| WR | 53.2% | **60.5%** (+7.3%p) |
| TPD | 2.85 | **2.65** |
| 수익률 | -29.6% | **+12.9%** |
| MDD | 30.0% | **10.6%** |
| PF | 0.651 | **0.762** |
| Top-3 제거 | -35.1% | **+2.3%** ← outlier 의존 대폭 감소 |
| Top-5 제거 | -37.6% | **-1.3%** ← 경계선 |

**원리**: 하락 추세에서 롱 진입 기준 강화(RSI 30→22), 숏 진입 기준 완화(RSI 70→65) → 추세 팔딱임 필터링

### Pullback 결론 (2026-06-03)

BTC pullback 전략은 구조적 문제:
- BTC gate (ETH식): 게이트가 틀린 방향 필터링 → 악화
- Signal Decay Exit: pullback 진입 자체가 "감쇠 후"라 즉시 발동 → WR 29.7%
- Confirmed Peak (n=20): 품질은 좋지만 거래 기회 너무 적음
- **결론: BTC pullback ❌ 포기. ETH pullback도 hist 0/10 → 추가 연구 가치 낮음**

---

## 추후 연구 대상

| 항목 | 파일 | 메모 |
|------|------|------|
| **Style B AdaptRSI** | `temp/scripts/26_style_b_adptrsi_hist.py` | ⚠️ 2026 +12.9% but hist 0/10 통과 / 평균 -4.7% (기준 -28.5%) — PF<1 문제 잔존 |
| Style B AdaptRSI + SL튜닝 | 미작성 | PF<1 원인: BB_mid 청산이 avg_win 제한 — SL 축소(1.5%) + TP 확대로 해결 가능성 |
| Meta-Label Classifier | `temp/scripts/25_metalabel_train.py` | ❌ label=1 비율 0.69% 극심 불균형 → 실제 전략 청산 결과로 레이블 재정의 필요 |

---

## 🔥 핵심 발견: Antifragile Trailing Stop (2026-06-03)

### 왜 다른가
기존 모든 전략 실패 원인 = **avg_win < avg_loss** (PF<1).
Antifragile Trailing은 구조적으로 역전:
- 진입: AdaptRSI (RSI 극단값, 소규모 초기 포지션 rr=0.10)
- 청산: **고정 SL/TP 없음** — ATR trailing stop이 따라감
- 증가: 유리하게 +0.5×ATR 이동 시마다 포지션 추가 (최대 3회)
- 결과: avg_win=+0.701% vs avg_loss=-0.084% = **8.4배 비대칭**

### 2026 BTC 결과 (최고 설정: trail_init=0.5, trail_tight=0.8)
| 항목 | 값 |
|------|-----|
| 수익률 | +132.4% |
| WR | 26.7% (trailing 전략 정상: 손실 작고 이익 큼) |
| PF | 8.364 |
| MDD | 2.7% |
| Top-5 제거 | +70.3% ← outlier 독립성 확인 |
| avg_win | +0.701% |
| avg_loss | -0.084% |

### 역사적 검증 최종 (require_bb=False, hist 9/10 ✅)
- **10/10 구간 모두 수익 양수**
- 평균 수익률: **+212.1% / 3개월**
- 평균 PF: 8.02
- [03] 2020-11: +909.7% / [04] 2021-09: +376.6% / [05] 2021-12: +221.3%

### 생산 스크립트
`scripts/backtest_antifragile.py` ← **승격 완료**

```bash
python scripts/backtest_antifragile.py --mode 2026
python scripts/backtest_antifragile.py --mode random --seed 42
```

### ETH 검증 결과 (2026-06-03)

| 코인 | 2026 수익 | TPD | MDD | Top5 제거 | hist 통과 | hist 평균 |
|------|----------|-----|-----|---------|---------|---------|
| BTC | +226% | 8.30 | 3.1% | +127% | 9/10 | +212% |
| **ETH** | **+558%** | **7.27** | **3.4%** | **+371%** | **10/10** | **+326%** |

ETH hist 전 구간 PF ≥ 6.1 / MDD ≤ 11.9% / TPD ≥ 6.4 / 10/10 수익 양수

### live_trader.py 통합 완료 (2026-06-03) ✅

`src/live_trader.py`에 Antifragile 전략 추가 (DL v17와 공존):

**활성화 방법** — `.env` 파일에 추가:
```
STRATEGY=antifragile   # antifragile 모드
# STRATEGY=dl_v17     # 기존 DL v17 모드 (기본값)
```

**변경 내용:**
- `DEFAULT_STATE`에 AF 전용 필드 5개 추가 (`af_trail_sl`, `af_peak_price`, `af_pyramid_count`, `af_current_rr`, `af_entry_atr`)
- `AF_PARAMS` 상수 추가 (검증된 파라미터: trail_init=0.5, trail_tight=0.8)
- `_compute_af_indicators()` — raw OHLCV에서 ATR, RSI, 1h EMA 추세 계산
- `process_tick_af()` — AdaptRSI 진입 + ATR trailing stop + 피라미딩 전체 로직
- `_close_and_log()`에 AF 상태 리셋 추가

**전략 전환 시 주의사항:**
1. 봇 중지 후 포지션 없는 상태 확인
2. `.env`에 `STRATEGY=antifragile` 추가
3. 봇 재시작

**다음 단계:**
- [ ] antifragile paper 모드 실사 테스트 시작
- [ ] 기존 DL v17 paper (Day 8, 2026-06-10) 분석 후 비교 판단

### AdaptRSI 역사적 검증 상세 (2026-06-03)

| 항목 | 기준선(30/70) | AdaptRSI(22/65-30/70-35/78) |
|------|-------------|---------------------------|
| hist 통과(3/3) | 0/10 | 0/10 |
| hist 2/3↑ | 7/10 | **9/10** |
| 수익 양수 구간 | 0/10 | **3/10** |
| 평균 수익 | -28.5% | **-4.7%** (+23.8%p) |

양수 구간: [03]2020-07 +13.7%, [05]2021-09 +23.2%, [09]2024-07 +8.0%
실패 이유: Top-5 제거 시 항상 음수 → PF<1 구조적 문제

---

## 미결 이슈 (언제든 처리 가능)

- [ ] `exchange_client.py` SYMBOL 하드코딩 제거 (ETH 지원 준비)
- [ ] `get_position()` markPrice 반환 버그 (hourly report PnL 왜곡)
- [ ] `max_dd_cb=1.0` 검토 (사실상 CB 비활성)
- [ ] `models/signal_model/` 경로 불일치 해결 (production 모델 직접 사용)

---

## 핵심 수치 기록 (분석 기준)

| 항목 | 값 | 비고 |
|------|-----|------|
| BTC 2026 수익률 | +180.7% | 펀딩비 포함 +162.8% |
| BTC Top-5 제거 | -18.9% | outlier 의존 ❌ |
| BTC Top-3 제거 | +13.9% | ✅ |
| ETH 2026 수익률 | +164.8% | 게이트 포함 기준 (게이트 없으면 -76.6%) |
| ETH 펀딩비 포함 | +155.7% | 게이트 포함 + 펀딩비 (차이 -9.19%p) |
| BTC hist pos | 6/10 | 통계적으로 약함 |
| ETH hist pos | 7/10 | |
| Paper 시작 자본 | 9,578 USDT | 2026-06-03 기준 |
