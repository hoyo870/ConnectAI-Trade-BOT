# live_tools — 실계좌 운용 관리 도구

BTC/ETH/SOL/XRP 4종목 자동매매 봇의 실계좌 운용에 필요한 모든 도구를 제공합니다.

---

## 파일 구성

```
live_tools/
├── run.py               # ⭐ 통합 실행 + 웹 대시보드 (여기서 시작)
├── live_trader.py       # 봇 메인 (직접 실행 금지 — run.py가 관리)
├── bot_manage.py        # 수동 관리 CLI (init / preflight / close / watch / fee)
├── exchange_client.py   # 거래소 API (BingX / Bybit)
├── telegram_notifier.py # 텔레그램 알림
├── data_pipeline.py     # 기술적 지표 계산
├── signal_extractor.py  # DL 신호 추출
├── expert_models.py     # TCN+Attention 전문가 모델
└── README.md
```

---

## 빠른 시작 (실계좌 첫 실행)

```bash
# 1. state 파일 초기화 (최초 1회)
python live_tools/bot_manage.py init

# 2. 사전 검증
python live_tools/bot_manage.py preflight

# 3. 통합 실행 (봇 + 감시 + 대시보드 한 번에)
python live_tools/run.py

# 브라우저에서 확인
open http://127.0.0.1:8765
```

---

## `run.py` — 통합 실행 + 웹 대시보드

단 하나의 커맨드로 봇·감시 데몬·대시보드를 모두 시작합니다.

### 실행

```bash
python live_tools/run.py                    # 실계좌 (기본)
python live_tools/run.py --paper            # Paper 모드 (API 조회 + 가상 매매)
python live_tools/run.py --port 8080        # 포트 변경 (기본: 8765)
python live_tools/run.py --no-auto-restart  # 자동 재시작 비활성화
```

### 대시보드 기능 (`http://127.0.0.1:8765`)

| 섹션 | 내용 |
|------|------|
| 포트폴리오 헤더 | 총 자본, 일일 PnL %, 현재 모드 |
| 프로세스 카드 | `trader` / `watchdog` 상태, PID, 가동시간, 재시작 횟수 |
| 시작/정지 버튼 | 각 프로세스 원클릭 제어 |
| 코인별 현황 | BTC/ETH/SOL/XRP 자본 + 포지션 상태 |
| 자본 추이 차트 | Chart.js 시계열 (trade_log 기반, 60초 갱신) |
| 로그 뷰어 | `logs/live_multi.log` 마지막 120줄, 5초 자동 갱신 |

### 자동 재시작 정책

| 상태 | 조건 | 동작 |
|------|------|------|
| **재시작** | 비정상 종료 | 5s → 15s → 30s → 60s 백오프 후 재시작 |
| **HALTED** | 종료 시점 일일 손실 ≥ 5% | 재시작 차단 (watchdog 의도 종료 방어) |
| **CRASH_LOOP** | 10분 내 3회 이상 재시작 | 재시작 중단, 수동 확인 필요 |

> HALTED/CRASH_LOOP 상태는 대시보드의 **▶ 시작** 버튼으로 수동 재시작 가능합니다.

### 내부 동작

```
run.py
├── SupervisorThread (백그라운드)
│   ├── live_tools/live_trader.py  (TRADE_MODE=real EXCHANGE=bingx)
│   └── live_tools/bot_manage.py watch --kill
└── Flask 웹서버 (127.0.0.1:8765)
    ├── GET  /              대시보드 HTML
    ├── GET  /api/status    프로세스 + 포트폴리오 상태 (JSON)
    ├── GET  /api/logs      로그 tail (JSON)
    ├── GET  /api/capital   코인별 자본 시계열 (JSON)
    └── POST /api/action/<start|stop>/<trader|watchdog>
```

### 종료

```bash
Ctrl+C   # 모든 하위 프로세스 SIGTERM 후 종료
```

---

## `bot_manage.py` — 수동 관리 CLI

`run.py` 실행 전후 수동 작업에 사용합니다.

### `init` — State 파일 초기화

실계좌 시작 전 코인별 자본을 올바르게 설정합니다.

```bash
python live_tools/bot_manage.py init           # 기본 (올바른 값이면 건너뜀)
python live_tools/bot_manage.py init --force   # 강제 덮어쓰기
```

- 코인당 250 USDT (총 1,000 USDT ÷ 4코인) 초기화
- 열린 포지션 있으면 중단 (거래소 수동 청산 후 재실행)
- 기존 `trade_log` 보존, 자본 값만 업데이트

| 코인 | State 파일 |
|------|-----------|
| BTC | `logs/live_state.json` |
| ETH | `logs/live_state_eth.json` |
| SOL | `logs/live_state_sol.json` |
| XRP | `logs/live_state_xrp.json` |

---

### `preflight` — 실계좌 시작 전 사전 검증

```bash
python live_tools/bot_manage.py preflight
```

| # | 체크 항목 | 판정 기준 |
|---|----------|----------|
| 1 | 거래소 API 연결 | real 모드 연결 성공 + 지연 측정 |
| 2 | USDT 가용 잔고 | ≥ 1,000 USDT (이체 전이면 ⚠️ 무시) |
| 3 | State 파일 자본 | 코인별 capital > 0 |
| 4 | 포지션 일치 | 거래소 실제 포지션 vs state 파일 일치 |
| 5 | 레버리지 설정 | set_leverage(3) 호출 성공 |

**재시작 시마다 실행 권장** — 포지션 불일치 사전 감지.

---

### `close` — 긴급 청산

봇 크래시 시 거래소에 남은 포지션 강제 청산 (state와 무관하게 API 직접 조회).

```bash
python live_tools/bot_manage.py close            # DRY-RUN (확인만)
python live_tools/bot_manage.py close --execute  # 실제 청산
```

---

### `watch` — 포트폴리오 감시 데몬

> `run.py` 사용 시 자동으로 실행됩니다. 단독 사용 시에만 아래 명령을 씁니다.

```bash
python live_tools/bot_manage.py watch --kill      # 손실 초과 시 봇 자동 종료
python live_tools/bot_manage.py watch --dry-run   # 1회 체크 테스트
```

| 기능 | 조건 | 동작 |
|------|------|------|
| 포트폴리오 손실 경보 | 일일 손실 > 5% | 텔레그램 경고 |
| 봇 자동 종료 | 손실 > 5% + `--kill` | SIGTERM |
| 봇 생존 감시 | state 파일 6분 미갱신 | 텔레그램 경고 |
| 시간 보고 | 매 1시간 | 코인별 현황 텔레그램 발송 |

---

### `resume` — 일일 Halt 해제

```bash
python live_tools/bot_manage.py resume          # halt된 코인 해제
python live_tools/bot_manage.py resume --force  # halt 여부 무관하게 강제 적용
```

일일 손실 한도 초과로 `daily_halt=True` 가 된 코인을 수동 해제합니다.
해제 후 `python live_tools/run.py` 로 봇을 재시작하세요.

---

### `backup` — State 파일 백업

```bash
python live_tools/bot_manage.py backup
```

`logs/backup/<UTC타임스탬프>/` 디렉토리에 현재 state 파일 4개를 복사합니다.

---

### `fee` — 수수료 참고

```bash
python live_tools/bot_manage.py fee
```

| 거래소 | Taker | Maker | 실효 Taker | 왕복 비용 |
|--------|-------|-------|-----------|---------|
| Bybit  | 0.044% | 0.020% | 0.044% | 0.088% |
| BingX  | 0.050% | 0.020% | **0.025%** (50% 페이백) | **0.05%** |

백테스트 `FEE_TOTAL = 0.07%` → 실제보다 보수적 (결과가 현실보다 불리).

---

## 운용 시나리오

### 일반 시작

```bash
python live_tools/bot_manage.py init       # 최초 1회만
python live_tools/bot_manage.py preflight  # 매번 실행 권장
python live_tools/run.py                   # 봇 + 감시 + 대시보드 시작
```

### 봇 크래시 후 재시작

```bash
python live_tools/bot_manage.py close          # 거래소 포지션 확인
python live_tools/bot_manage.py close --execute  # 포지션 있으면 청산
python live_tools/bot_manage.py preflight      # 재검증
python live_tools/run.py                       # 재시작
```

### 자본 재설정 후 재시작

```bash
python live_tools/bot_manage.py close --execute  # 포지션 정리
python live_tools/bot_manage.py init --force     # 250 USDT × 4 재설정
python live_tools/bot_manage.py preflight        # 검증
python live_tools/run.py                         # 시작
```

---

## 설정값

| 항목 | 값 |
|------|---|
| 총 시드 | 1,000 USDT |
| 코인당 자본 | 250 USDT (25%) |
| 대상 코인 | BTC, ETH, SOL, XRP |
| 레버리지 | 3x |
| 포트폴리오 손실 한도 | -5% (일일) |
| 봇 응답 없음 판단 | state 파일 6분 미갱신 |
| 감시 주기 | 30초 |
| 대시보드 포트 | 8765 |
| 크래시루프 임계값 | 10분 내 3회 |

---

## 로그 파일

| 파일 | 내용 |
|------|------|
| `logs/live_multi.log` | 실계좌 봇 실행 로그 |
| `logs/paper_multi.log` | Paper 모드 실행 로그 |
| `logs/watchdog.log` | Watchdog 감시 로그 |
| `logs/supervisor.log` | run.py 슈퍼바이저 로그 |
| `logs/live_state*.json` | 코인별 거래 state |

---

## 환경변수 (`.env` — 프로젝트 루트)

```env
EXCHANGE=bingx              # bybit | bingx
TRADE_MODE=real             # real | paper (run.py --paper 옵션으로 override)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
BINGX_REAL_API_KEY=...
BINGX_REAL_API_SECRET=...
```
