# hy-trade-bot (NAS 경량화 버전)

Antifragile 전략 전용 경량화 트레이딩 봇 — NAS(DS920+) Docker 배포 최적화 버전.

`connectai-trade-bot`의 DL 모델(torch/SB3) 의존성을 제거하고 rule-based Antifragile 전략만 추출.

---

## Git 브랜치 구조

```
hoyo870/ConnectAI-Trade-BOT
├── main          ← 풀 버전 (DL v17 + Antifragile, torch/SB3 포함)
└── nas-deploy    ← 이 브랜치 — NAS 경량화 (Antifragile 전용, torch 없음)
```

- `main` 브랜치에서 전략 업데이트 → `nas-deploy` 브랜치에 동기화
- NAS는 `nas-deploy` 브랜치만 pull하여 사용

---

## 전략 요약

- **Antifragile AdaptRSI**: 1h EMA 추세 방향에 따라 RSI 임계값 동적 조정
- **ATR Trailing Stop**: 고정 SL 없음, 추세를 따라가며 이익 보호
- **피라미딩**: 유리한 방향으로 0.5×ATR 이동 시마다 포지션 추가 (최대 3회)
- **4종목 동시 운용**: BTC/ETH/SOL/XRP 각 25% 분할
- **레버리지**: 5x (hist 10창 9~10/10 검증, MDD ≤5.6%)

---

## 요구사항

- Python 3.11+
- 패키지: `requirements.txt` (torch/SB3 없음, ~8개)
- 거래소: Bybit (BingX 지원, sandbox는 Bybit만 가능)

---

## 설치

### 직접 실행 (venv)

```bash
git clone -b nas-deploy --single-branch \
  https://github.com/hoyo870/ConnectAI-Trade-BOT.git hy-trade-bot
cd hy-trade-bot

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# .env 파일에 API 키 입력
```

### Docker (DS920+ 권장)

```bash
git clone -b nas-deploy --single-branch \
  https://github.com/hoyo870/ConnectAI-Trade-BOT.git hy-trade-bot
cd hy-trade-bot

cp .env.example .env
# .env 파일에 API 키 입력

docker compose up -d
```

---

## 설정 (.env)

```env
EXCHANGE=bybit           # bybit | bingx
TRADE_MODE=paper         # paper(시뮬) | sandbox(테스트넷) | real(실거래)
STRATEGY=antifragile     # 고정값

# Bybit API 키
BYBIT_REAL_API_KEY=your_key
BYBIT_REAL_API_SECRET=your_secret

# 샌드박스 테스트용
BYBIT_SANDBOX_API_KEY=your_sandbox_key
BYBIT_SANDBOX_API_SECRET=your_sandbox_secret

# 텔레그램 알림 (선택)
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# 시작 자본 (paper 모드, 종목당 SEED/4 USDT)
PAPER_SEED=10000
```

---

## 실행

### 직접 실행

```bash
# paper 모드 (기본)
python live_tools/run.py

# 백그라운드
nohup python live_tools/run.py > logs/run.log 2>&1 &
```

### 대시보드

브라우저에서 접속:
```
http://localhost:8765        # 로컬
http://NAS-IP:8765           # NAS 원격
```

### 텔레그램 명령

| 명령 | 동작 |
|------|------|
| `/stop` | 봇 종료 (포지션 청산 후) |
| `/account` | 현재 계좌 현황 보고 |

---

## NAS(DS920+) 배포

### DSM Container Manager

1. NAS SSH 접속 → repo clone
2. `.env` 파일 작성
3. Container Manager → 가져오기 → Docker Compose

```bash
# NAS SSH
cd /volume1/
git clone -b nas-deploy --single-branch \
  https://github.com/hoyo870/ConnectAI-Trade-BOT.git hy-trade-bot
cd hy-trade-bot
cp .env.example .env
nano .env   # API 키 입력

docker compose up -d
```

### DSM 방화벽 설정

제어판 → 보안 → 방화벽 → 규칙 추가:
- 포트: **8765** (TCP)
- 출발지: 로컬 네트워크 또는 VPN IP

### 자동 시작

`docker-compose.yml`의 `restart: unless-stopped`로 NAS 재부팅 시 자동 실행.

---

## 업데이트

```bash
cd /volume1/hy-trade-bot
git pull
docker compose down
docker compose up -d --build
```

---

## 소액 실거래 테스트 기준

| 코인 | 최소 자본/종목 (10x, rr=0.10) | 비고 |
|------|------------------------------|------|
| BTC | ~62 USDT | 4종목 합계 ~248 USDT |
| ETH | ~2 USDT | |
| SOL | ~1 USDT | |
| XRP | ~0.01 USDT | |

- **90 USDT** 테스트 시: BTC 자동 스킵, ETH/SOL/XRP 3종목 운용
- **250 USDT+**: 4종목 전체 운용 가능

---

## 파라미터 (AF_PARAMS)

`live_tools/live_trader.py`의 `AF_PARAMS` dict에서 수정:

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `leverage` | 5 | 레버리지 |
| `rr_base` | 0.10 | 초기 리스크 비율 |
| `ut_rsi_lo/hi` | 40/85 | 상승추세 RSI 임계값 |
| `trail_atr_init` | 1.0 | 초기 trailing stop 거리 (ATR 배수) |
| `trail_atr_tight` | 1.5 | 피라미딩 후 trailing stop |
