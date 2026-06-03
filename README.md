# ConnectAI Trade Bot

BTC/USDT · ETH/USDT 5분봉 암호화폐 선물 자동매매 봇.  
두 가지 독립 전략을 지원합니다: **DL v17** (TCN+Attention 신경망) / **Antifragile** (AdaptRSI + ATR trailing).

---

## 성과 (2026 백테스트)

### Antifragile Trailing Stop (권장 · 2026-06-03 검증)

| 코인 | 수익률 | TPD | MDD | PF | Top-5 제거 | hist 통과 |
|------|-------|-----|-----|-----|----------|---------|
| BTC | +226% | 8.30 ✅ | 3.1% | 8.45 | +127% ✅ | 9/10 |
| ETH | +558% | 7.27 ✅ | 3.4% | 7.82 | +371% ✅ | 10/10 |

> Top-5 제거 후에도 강력한 양수 수익 → outlier 독립적

### DL v17 Instant (기존)

| 코인 | 승률 | TPD | 수익률 | MDD |
|------|------|-----|-------|-----|
| BTC | 49.6% | 1.84 | +181% | 57.4% |
| ETH | 48.3% | 1.16 | +165% | 28.8% |

> BTC Top-5 제거 시 -18.9% (outlier 의존 주의)

---

## 아키텍처

```
OHLCV 5분봉
    │
    ├─── [DL v17 경로]
    │        │
    │        ▼
    │    data_pipeline.py    ← 기술 지표 30+개, 스케일링
    │        │
    │        ▼
    │    expert_models.py    ← TCN(6 layers) + Multi-Head Attention
    │        ├── Long Expert  → signal_long  (0~1)
    │        ├── Short Expert → signal_short (0~1)
    │        └── Context      → signal_context (0~1)
    │        │
    │        ▼
    │    hybrid_engine.py    ← 4단계 Tier 레버리지(2.5x~10x) + CB
    │
    └─── [Antifragile 경로]
             │
             ▼
         AdaptRSI            ← 1h EMA 방향별 RSI 임계값 동적 조정
             │
             ▼
         ATR trailing stop   ← 고정 SL/TP 없음 · peak 추적
             │
             ▼
         Pyramiding          ← 유리방향 +0.5ATR마다 포지션 추가
    │
    ▼
live_trader.py               ← STRATEGY 환경변수로 경로 선택
```

---

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env: API 키, Telegram 토큰 입력
```

`models/production/` 디렉토리에 모델 파일 필요 (`.pt` / `.pkl` — git 제외, 별도 수령):

```
models/production/
├── btc_expert_long.pt    btc_expert_short.pt    btc_expert_context.pt
├── btc_scaler.pkl        btc_params.json
├── eth_expert_long.pt    eth_expert_short.pt    eth_expert_context.pt
├── eth_scaler.pkl        eth_params.json
```

---

## 실행

### 전략 선택

`.env` 파일에 전략을 설정합니다:

```bash
STRATEGY=antifragile   # Antifragile Trailing Stop (권장)
# STRATEGY=dl_v17      # DL v17 Instant (기존 기본값)
```

### 라이브 트레이더

```bash
# 포그라운드
python src/live_trader.py

# 백그라운드 (권장)
nohup python src/live_trader.py > logs/live.log 2>&1 &

# 또는 deploy 스크립트 사용
bash deploy/run_paper.sh
```

`TRADE_MODE`:
- `paper` — 실시세 조회, 가상 주문 (기본값 · 테스트용)
- `sandbox` — Bybit 테스트넷
- `real` — 실계좌 (**주의!**)

### 백테스트

```bash
# Antifragile 전략
python scripts/backtest_antifragile.py --coin btc --mode 2026
python scripts/backtest_antifragile.py --coin eth --mode 2026
python scripts/backtest_antifragile.py --coin both --mode random --seed 42

# DL v17 전략
python scripts/backtest.py --coin both --mode 2026
python scripts/backtest.py --coin both --mode random --windows 10 --seed 42
```

---

## 디렉토리 구조

```
connectai-trade-bot/
├── src/
│   ├── live_trader.py          # 실거래 루프 (STRATEGY 분기)
│   ├── data_pipeline.py        # 지표 계산, 스케일링
│   ├── expert_models.py        # TCN+Attention 모델
│   ├── hybrid_engine.py        # DL v17 백테스트 엔진
│   ├── signal_extractor.py     # 배치 시그널 추출
│   ├── exchange_client.py      # Bybit API
│   ├── telegram_notifier.py    # 텔레그램 알림
│   ├── data_fetcher.py         # OHLCV 수집
│   └── rl/                     # RL (보류)
│
├── scripts/
│   ├── backtest_antifragile.py # Antifragile 전략 백테스트 (승격됨)
│   ├── backtest.py             # DL v17 백테스트 CLI
│   ├── analyze_funding.py      # BTC 펀딩비 분석
│   ├── analyze_funding_eth.py  # ETH 펀딩비 분석
│   └── analyze_top_trades.py   # outlier 의존도 분석
│
├── temp/scripts/               # 연구용 실험 스크립트 (참고용)
│
├── models/production/          # 프로덕션 모델 파일
├── deploy/                     # 배포 설정 (systemd, launchd, nohup)
├── data/                       # OHLCV 데이터 (git 제외)
├── logs/                       # 런타임 로그 (git 제외)
│
├── STRATEGY_ANTIFRAGILE.md     # Antifragile 전략 규칙
├── MODEL_HISTORY.md            # DL 모델 이력
├── SCRIPTS.md                  # 스크립트 사용 가이드
├── TODO_SCHEDULE.md            # 연구 스케줄
└── TODO.md                     # 미결 이슈
```

---

## 알림

텔레그램으로 실시간 알림:
- 포지션 진입 / 청산 (trailing SL 도달 포함)
- 피라미딩 추가 신호 (Antifragile)
- 서킷 브레이커 · 일일 손실 한도
- 1시간 주기 상태 보고

```bash
/stop   # 텔레그램에서 봇 긴급 정지 (포지션 청산 후 종료)
```

---

## 주의사항

- `TRADE_MODE=real` 전 반드시 paper 모드로 충분히 검증
- 전략 전환(`STRATEGY` 변경) 시 포지션 없는 상태에서 재시작
- 모든 백테스트 결과는 과거 성과이며 미래 수익을 보장하지 않음
- 레버리지 선물 거래는 원금 손실 위험이 있음
