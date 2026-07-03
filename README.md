# ConnectAI Trade Bot

> ⚠️ **[2026-06-24] 이 문서의 백테스트 수익 수치는 무효(과대계상)입니다.**
> 피라미딩 수익률을 최초진입가 기준으로 계산하던 회계 버그로 손실 전략이 큰 수익으로 표시됐습니다.
> 가중평균(거래소 실현손익) 기준 수정 시 예: BTC 2026 OOS가 +수익 → **−48.7%**.
> 피라미딩은 구조적 손실로 비활성화(`add_levels=0`)했고 정직한 단일진입 재베이스라인 진행 중.
> 상세: 메모리 `project_pnl_accounting_fix`, 계획 `~/.claude/plans/cosmic-leaping-goblet.md`.

BTC/USDT · ETH/USDT · SOL/USDT · XRP/USDT 5분봉 암호화폐 선물 자동매매 봇.  
**Antifragile 전략** (AdaptRSI + ATR trailing stop + AFEnsemble ML 필터) — 4종목 동시 운영.

---

## 전략 개요

### Antifragile + ML 필터 (현행 유일 전략)

| 구성 요소 | 내용 |
|----------|------|
| 진입 신호 | AdaptRSI — 1h BB 트렌드(σ=0) 방향별 RSI 임계값 동적 조정 |
| 청산 | ATR trailing stop — 고정 SL/TP 없음, peak 추적 |
| 사이징 | rr_base=0.10 시작 → 유리방향 0.5ATR마다 피라미딩 |
| ML 필터 | AFEnsemble (LightGBM + LSTM, theta=0.300) — 진입 시 추가 검증, **필수** |
| 레버리지 | .env LEVERAGE (기본 5배) |
| 운영 | 단일 프로세스, 4코인 동시, 시드 25%씩 균등 배분 |

---

## 백테스트 성과 (2026 OOS: 2026-01-01 ~ 2026-06-23, ML 필터 포함)

> 엔진: `backtest_af_exact.py` + AFEnsemble (theta=0.300, bb_sigma=0) — live_trader 동일 로직  
> 수수료: FEE_TOTAL=0.061%/side (실거래 49건 실측, Bybit taker 0.055% + 슬리피지 0.006%)

| 코인 | 수익률 | TPD | MDD | WR | PF | Top-5 제거 | 판정 |
|------|-------|-----|-----|-----|-----|-----------|------|
| BTC  | +5,077%    | 2.83 ✅ | 10.7% | 39.3% | 3.431 | +294.0% ✅ | 3/3 ✅ |
| ETH  | +106,282%  | 4.13 ✅ | 17.8% | 41.0% | 3.613 | +606.5% ✅ | 3/3 ✅ |
| SOL  | +1,303,654%| 4.76 ✅ | 14.7% | 39.1% | 4.214 | +839.7% ✅ | 3/3 ✅ |
| XRP  | +55,869%   | 3.84 ✅ | 25.3% | 37.5% | 3.411 | +542.0% ✅ | 3/3 ✅ |

> WR ~40%에도 고수익인 이유: avg_win / avg_loss ≈ 4~5x (ATR trailing stop 기반 손익비)  
> 피라미딩(최대 3레벨)이 수익 포지션을 증폭 — rr_final 비례로 수수료도 자동 스케일

---

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env: API 키, Telegram 토큰, LEVERAGE 설정
```

`models/af_ensemble/saved/` 디렉토리에 학습된 ML 모델 필요 (git 제외, 별도 수령):

```
models/af_ensemble/saved/
├── lgbm.pkl
├── lstm.pt
└── meta.json
```

---

## 실행

### 봇 구동 (권장)

```bash
# run.py — supervisor + live_trader 통합 실행
source .venv/bin/activate
python live_tools/run.py              # real 모드 (기본)
python live_tools/run.py --paper      # paper 모드
```

### 직접 실행

```bash
python live_tools/live_trader.py
```

### 환경 변수 (.env)

```bash
STRATEGY=antifragile          # 고정 (Antifragile 전용, DL v17 제거됨)
TRADE_MODE=real               # real | sandbox | paper
LEVERAGE=10                   # 레버리지
ML_MODEL_DIR=models/af_ensemble/saved  # ML 모델 경로
EXCHANGE=bybit                # bybit | bingx
AF_PARAM_PRESET=              # '' (기본) | aggressive | conservative
```

### 로그 / 상태 파일

```
logs/live_multi.log          ← 공유 로그
logs/live_state.json         ← BTC 상태
logs/live_state_eth.json     ← ETH 상태
logs/live_state_sol.json     ← SOL 상태
logs/live_state_xrp.json     ← XRP 상태
```

---

## 백테스트

유일한 공식 백테스트 엔진: `scripts/backtest_af_exact.py` (ML 필터 항상 포함)

> **live_trader = backtest 전략 로직 완전 동일** — 허용된 차이 4가지만 존재:  
> ① OHLCV 소스(실시간 vs 기록) ② 실행 타이밍(시장가 주문 vs 즉시 체결 가정)  
> ③ 미세 슬리피지/수수료 ④ 시드/자본 기준

```bash
# 2026 OOS 전체 (4코인)
python scripts/backtest_af_exact.py --mode 2026 --coin all

# 랜덤 히스토리 10창 검증
python scripts/backtest_af_exact.py --mode hist --coin all --windows 10 --seed 42

# 특정 실거래 기간 재현 (Jun 08~14 / Jun 15~22 / Jun 전체)
python scripts/backtest_af_exact.py --mode jun0814 --coin all
python scripts/backtest_af_exact.py --mode jun1522 --coin all
python scripts/backtest_af_exact.py --mode jun --coin all
```

Python API:

```python
from strategies.backtest_engine import AntifragileBacktestRunner

runner = AntifragileBacktestRunner.from_saved("models/af_ensemble/saved")
df, df_ml = runner.load_coin("btc", start="2026-01-01", end="2026-06-01")
result = runner.run(df, df_ml)
runner.print_result("BTC 2026 OOS", result, days=151)
```

### 판정 기준 (3/3 통과)

| 기준 | 조건 |
|------|------|
| 수익률 | > 0% |
| TPD | ≥ 1.5 |
| Top-5 제거 후 수익 | > 0% |

---

## 데이터 업데이트

```bash
# 실거래 전 최신 OHLCV 수집 필수
python src/data_fetcher.py --symbol BTC/USDT --start 2026-06-01
python src/data_fetcher.py --symbol ETH/USDT --start 2026-06-01
python src/data_fetcher.py --symbol SOL/USDT --start 2026-06-01
python src/data_fetcher.py --symbol XRP/USDT --start 2026-06-01
```

---

## 디렉토리 구조

```
connectai-trade-bot/
├── live_tools/
│   ├── run.py               # 봇 구동 supervisor (권장 진입점)
│   ├── live_trader.py       # 실거래 루프
│   ├── exchange_client.py   # Bybit / BingX API
│   ├── telegram_notifier.py # 텔레그램 알림
│   ├── bot_manage.py        # 상태 초기화·관리
│   └── data_fetcher.py      # (src/ 공유)
│
├── strategies/
│   ├── antifragile.py       # AntifragileStrategy 클래스 (live_trader 공유)
│   ├── backtest_engine.py   # AntifragileBacktestRunner (공식 백테스트 엔진)
│   └── indicators.py        # 지표 계산 단일 소스
│
├── scripts/
│   ├── backtest_af_exact.py # 공식 백테스트 CLI
│   ├── generate_ml_labels.py# ML 재학습용 레이블 생성
│   └── train_af_ml.py       # ML 앙상블 재학습
│
├── models/
│   └── af_ensemble/
│       ├── saved/           # 학습된 모델 (LightGBM + LSTM)
│       ├── ensemble.py
│       └── feature_extractor.py
│
├── config/
│   └── af_params.py         # 파라미터/프리셋 단일 소스
│
├── src/
│   └── data_fetcher.py      # OHLCV 수집
│
├── data/                    # OHLCV CSV (git 제외)
├── logs/                    # 런타임 로그 (git 제외)
│
├── STRATEGY_ANTIFRAGILE.md  # 전략 상세 규칙
├── SCRIPTS.md               # 스크립트 사용 가이드
├── DEPRECATED_ML_LESS_STRATEGY.md  # 폐기된 엔진 이력
└── MODEL_HISTORY.md         # ML 모델 이력
```

---

## 알림

텔레그램으로 실시간 알림:
- 포지션 진입 / 청산 (trailing SL 도달 포함)
- 피라미딩 추가
- 1시간 주기 상태 보고
- 일일 결산 (거래소 실현 PnL 포함)

```bash
/stop   # 텔레그램에서 봇 긴급 정지
```

---

## 주의사항

- ML 모델(`models/af_ensemble/saved/`) 없으면 봇 시작 불가 (fail-fast)
- `TRADE_MODE=real` 전 반드시 `--paper` 모드 검증 후 전환
- 실거래 전 최신 OHLCV 수집 → `backtest_af_exact.py` 검증 → 봇 시작
- 모든 백테스트 결과는 과거 성과이며 미래 수익을 보장하지 않음
