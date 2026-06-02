# ConnectAI Trade Bot

BTC/USDT · ETH/USDT 5분봉 암호화폐 선물 자동매매 봇.  
TCN + Multi-Head Self-Attention 기반 Expert 모델로 롱/숏 진입 시그널을 생성하고, Bybit 선물에 자동 주문합니다.

---

## 성과 (2026년 백테스트 기준)

| 코인 | 승률 | TPD | 일 평균 수익률 | MDD |
|------|------|-----|-------------|-----|
| BTC/USDT | 49.6% ✅ | 1.84 ✅ | +1.30% ✅ | 57.4% |
| ETH/USDT | 48.3% ✅ | 1.16 | +1.11% ✅ | 28.8% |

> 판정 기준: 승률 ≥ 45% · TPD ≥ 1.5 · 일 평균 수익률 ≥ 1%

---

## 아키텍처

```
OHLCV 5분봉
    │
    ▼
data_pipeline.py         ← 기술 지표 30+ 개 계산, 스케일링
    │
    ▼
expert_models.py         ← TCN (6 layers) + Multi-Head Attention
    ├── Long Expert      → signal_long  (0~1)
    ├── Short Expert     → signal_short (0~1)
    └── Context Expert   → signal_context (0~1)
    │
    ▼
hybrid_engine.py         ← v17 백테스트 / 진입-청산 엔진
    ├── 4단계 Tier 레버리지 (0.48 ~ 0.72)
    ├── Phase2_gate (ETH: EMA9/21 + RSI + vol_ratio + 1h trend)
    └── Drawdown Circuit Breaker (max_dd_cb)
    │
    ▼
live_trader.py           ← 실거래 루프 (5분봉마다 실행)
```

### 모델 파라미터

| 항목 | BTC | ETH |
|------|-----|-----|
| HIDDEN_DIM | 128 | 256 |
| NUM_TCN_LAYERS | 4 | 6 |
| SEQ_LEN | 60봉 (5h) | 60봉 (5h) |
| min_hold_bars | 120봉 (10h) | 180봉 (15h) |

---

## 설치

### 1. 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 API 키와 Telegram 토큰 입력
```

### 3. 모델 파일 준비

`models/production/` 디렉토리에 아래 파일이 있어야 합니다:

```
models/production/
├── btc_expert_long.pt
├── btc_expert_short.pt
├── btc_expert_context.pt
├── btc_scaler.pkl
├── btc_params.json          ← 커밋 포함
├── eth_expert_long.pt
├── eth_expert_short.pt
├── eth_expert_context.pt
├── eth_scaler.pkl
└── eth_params.json          ← 커밋 포함
```

> `.pt` / `.pkl` 파일은 용량 문제로 git에서 제외됩니다. 별도 스토리지에서 수령하거나 직접 학습해 주세요.

---

## 실행

### 라이브 트레이더

```bash
# 포그라운드
python src/live_trader.py

# 백그라운드 (권장)
nohup python src/live_trader.py > logs/live.log 2>&1 &
```

`TRADE_MODE` 환경 변수로 모드를 제어합니다:
- `paper` — 실시세 조회, 가상 주문만 (기본값, 테스트용)
- `sandbox` — Bybit 테스트넷 실제 주문
- `real` — 실계좌 주문 (주의!)

### 백테스트 CLI

```bash
# 2026년 기본 백테스트
python scripts/backtest.py --coin both --mode 2026

# 히스토리 랜덤 10회
python scripts/backtest.py --coin both --mode random --windows 10 --seed 42

# ETH Pullback 전략 테스트
python scripts/backtest.py --coin eth --mode 2026 --strategy pullback

# 특정 구간
python scripts/backtest.py --coin btc --mode custom --start 2024-01-01 --end 2024-04-01
```

**파라미터:**

| 옵션 | 값 | 설명 |
|------|----|------|
| `--coin` | `btc` \| `eth` \| `both` | 대상 코인 |
| `--mode` | `2026` \| `random` \| `custom` | 백테스트 구간 |
| `--strategy` | `instant` \| `pullback` | 진입 전략 |
| `--windows` | int | 랜덤 구간 수 (기본 10) |
| `--seed` | int | 랜덤 시드 (기본 42) |
| `--start` | `YYYY-MM-DD` | custom 시작일 |
| `--end` | `YYYY-MM-DD` | custom 종료일 |

---

## 디렉토리 구조

```
connectai-trade-bot/
├── src/                        # 핵심 라이브러리
│   ├── data_pipeline.py        # 지표 계산, 스케일러
│   ├── expert_models.py        # TCN+Attention 모델 아키텍처
│   ├── hybrid_engine.py        # 백테스트 엔진 + compute_metrics
│   ├── signal_extractor.py     # 배치 시그널 추출
│   ├── live_trader.py          # 실거래 루프 (5분봉)
│   ├── exchange_client.py      # Bybit API 클라이언트
│   ├── telegram_notifier.py    # 텔레그램 알림
│   ├── data_fetcher.py         # OHLCV 원시 데이터 수집
│   └── rl/                     # RL 보류 (미사용)
│
├── scripts/
│   └── backtest.py             # CLI 백테스트
│
├── models/
│   └── production/
│       ├── btc_params.json     # BTC 프로덕션 파라미터
│       └── eth_params.json     # ETH 프로덕션 파라미터
│
├── data/                       # OHLCV 데이터 (git 제외)
├── logs/                       # 런타임 로그 (git 제외)
│
├── config.yaml                 # 모델/RL 학습 설정
├── requirements.txt
├── .env.example                # 환경 변수 템플릿
├── SCRIPTS.md                  # 스크립트 사용법
├── MODEL_HISTORY.md            # 모델 이력
└── TODO.md                     # 연구 과제
```

---

## 진입 전략

### Instant (기본)
시그널이 tier 기준 초과 시 즉시 진입. BTC에 적합.

### Pullback
이전 봉이 tier 기준 충족 후, 현재 봉 시그널이 감소할 때 진입. 피크-후-풀백 패턴.  
ETH 히스토리 기준 WR +2.3%p, daily +0.20%p, MDD -5.5%p 개선 확인 (연구 중).

### Tier 구조 (4단계 레버리지)

| 시그널 임계값 | 레버리지 | RR |
|---|---|---|
| ≥ 0.72 | 10x | 0.50 |
| ≥ 0.62 | 4x | 0.40 |
| ≥ 0.55 | 3.5x | 0.30 |
| ≥ 0.48 | 2.5x | 0.25 |

---

## 알림

텔레그램으로 아래 이벤트를 실시간 알림합니다:
- 포지션 진입 / 청산
- 서킷 브레이커 발동
- 1시간 주기 상태 보고
- 오류 발생

---

## 주의사항

- `TRADE_MODE=real` 설정 전 반드시 `sandbox` 또는 `paper`로 충분히 검증하세요.
- 모든 백테스트 결과는 과거 성과이며 미래 수익을 보장하지 않습니다.
- 레버리지 선물 거래는 원금 손실 위험이 있습니다.
