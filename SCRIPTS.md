# 스크립트 구조

## 디렉토리

```
connectai-trade-bot/
├── src/                        # 핵심 라이브러리
│   ├── data_pipeline.py        # OHLCV → 지표 계산, 스케일러, 레이블
│   ├── expert_models.py        # TCN+MultiHeadAttention 모델 아키텍처
│   ├── hybrid_engine.py        # v17 백테스트 엔진 + compute_metrics
│   ├── signal_extractor.py     # 배치 시그널 추출 (live_trader 의존)
│   ├── live_trader.py          # 실거래 자동매매 루프 (무수정 유지)
│   ├── exchange_client.py      # Bybit API 클라이언트
│   ├── telegram_notifier.py    # 텔레그램 알림
│   ├── data_fetcher.py         # OHLCV 원시 데이터 수집
│   └── rl/                     # RL 보류 (미사용)
│       ├── __init__.py
│       ├── rl_trainer.py
│       └── trading_env.py
│
├── scripts/
│   └── backtest.py             # CLI 백테스트 (아래 참조)
│
└── temp/scripts/
    └── 22_backtest_pullback.py # Pullback 연구용 원본 (참고용)
```

---

## scripts/backtest.py

반복 백테스트를 CLI 파라미터로 실행. 프로젝트 루트에서 실행.

### 파라미터

| 파라미터 | 값 | 기본값 | 설명 |
|---|---|---|---|
| `--coin` | `btc` \| `eth` \| `both` | `both` | 대상 코인 |
| `--mode` | `2026` \| `random` \| `custom` | `2026` | 백테스트 구간 |
| `--strategy` | `instant` \| `pullback` | `instant` | 진입 전략 |
| `--windows` | int | `10` | 랜덤 구간 수 (random 모드) |
| `--seed` | int | `42` | 랜덤 시드 |
| `--start` | `YYYY-MM-DD` | — | 시작일 (custom 모드 필수) |
| `--end` | `YYYY-MM-DD` | — | 종료일 (custom 모드 선택) |

### 사용 예시

```bash
# 2026 프로덕션 검증
python scripts/backtest.py --coin both --mode 2026

# ETH Pullback 전략 테스트
python scripts/backtest.py --coin eth --mode 2026 --strategy pullback

# 히스토리 랜덤 10회
python scripts/backtest.py --coin both --mode random --windows 10 --seed 42

# 특정 구간
python scripts/backtest.py --coin btc --mode custom --start 2024-01-01 --end 2024-04-01
```

### 전략 설명

- `instant` : 시그널 발생 즉시 진입 (`run_v17_backtest`)
- `pullback` : 이전 봉 피크 후 현재 봉 감소 시 진입, tier는 피크 기준

### 판정 기준 (3/3 통과)

- 승률 ≥ 45%
- 일일 거래수(TPD) ≥ 1.5
- 일 평균 수익률 ≥ 1.0%

---

## src/hybrid_engine.py

`run_v17_backtest()` — 즉시 진입 백테스트 엔진  
`compute_metrics()` — equity curve → 성과 지표 딕셔너리

`scripts/backtest.py`와 `temp/scripts/22_backtest_pullback.py` 모두 여기서 `compute_metrics`를 import.

---

## 실행 중인 프로세스

```bash
python src/live_trader.py  # nohup으로 백그라운드 실행 중
```

`live_trader.py`는 수정하지 않음. `signal_extractor.extract_signals_from_df` 의존.
