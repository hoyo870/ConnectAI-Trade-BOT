# 스크립트 사용 가이드

## 구조

```
scripts/
├── backtest_af_exact.py   # 유일한 공식 백테스트 CLI (ML 필수)
├── generate_ml_labels.py  # ML 재학습용 레이블 생성 (필요 시 실행)
└── train_af_ml.py         # ML 앙상블 재학습 (필요 시 실행)

strategies/
├── antifragile.py         # AntifragileStrategy 클래스 (live_trader + 백테스트 공유)
├── backtest_engine.py     # AntifragileBacktestRunner 클래스 (공식 백테스트 엔진)
└── indicators.py          # 지표 계산 단일 소스

config/
└── af_params.py           # 파라미터/프리셋 단일 소스 (backtest + live_trader 공용)

live_tools/
└── live_trader.py         # 실거래 봇 (ML 필터 항상 필수)
```

---

## 핵심 원칙

> **백테스트 ↔ 실거래 단일 진실 소스**
> - `strategies/antifragile.py::AntifragileStrategy` — live_trader와 백테스트가 **동일 클래스** 사용
> - `config/af_params.py::DEFAULT_PARAMS` — 파라미터 단일 소스
> - ML 앙상블 필터(`models/af_ensemble/saved`) — **항상 필수**, 비활성화 불가

---

## 백테스트 실행 (유일한 공식 방법)

```bash
# 2026 OOS 전체 검증 (4코인)
python scripts/backtest_af_exact.py --mode 2026 --coin all

# 랜덤 히스토리 10창 검증
python scripts/backtest_af_exact.py --mode hist --coin all --windows 10 --seed 42

# 특정 실거래 기간 재현
python scripts/backtest_af_exact.py --mode jun1819 --coin all
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--coin` | `all` | `btc` \| `eth` \| `sol` \| `xrp` \| `all` |
| `--mode` | `2026` | `2026` \| `hist` \| `jun1819` |
| `--windows` | `10` | hist 모드 랜덤 검증 창 수 |
| `--seed` | `42` | hist 모드 랜덤 시드 |
| `--model` | `models/af_ensemble/saved` | ML 모델 경로 (변경 불필요) |

### Python API 사용

```python
from strategies.backtest_engine import AntifragileBacktestRunner

runner = AntifragileBacktestRunner.from_saved("models/af_ensemble/saved")
df, df_ml = runner.load_coin("btc", start="2026-01-01", end="2026-06-01")
result = runner.run(df, df_ml)
runner.print_result("BTC 2026 OOS", result, days=151)
```

---

## 판정 기준 (3/3 통과)

| 기준 | 조건 |
|------|------|
| 수익률 | > 0% |
| TPD (일일 거래 수) | ≥ 1.5 |
| Top-5 제거 후 수익 | > 0% |

---

## 데이터 업데이트

```bash
# 4코인 최신 OHLCV 수집 (실거래 전 갱신 필수)
source .venv/bin/activate
python src/data_fetcher.py --symbol BTC/USDT --start 2026-06-19
python src/data_fetcher.py --symbol ETH/USDT --start 2026-06-19
python src/data_fetcher.py --symbol SOL/USDT --start 2026-06-19
python src/data_fetcher.py --symbol XRP/USDT --start 2026-06-19
```

---

## 폐기된 스크립트 (삭제 완료 — 2026-06-22)

| 삭제된 파일 | 대체 |
|------------|------|
| `scripts/backtest_antifragile.py` | `scripts/backtest_af_exact.py` |
| `scripts/backtest_af_ml.py` | `scripts/backtest_af_exact.py` |
| `scripts/batch_backtest.py` | `strategies/backtest_engine.AntifragileBacktestRunner` |
| `temp/batch_backtest.py` | 동일 |

> 폐기 상세 이유: `DEPRECATED_ML_LESS_STRATEGY.md` 참고

---

## 혼동 방지 체크리스트

- [ ] 백테스트는 반드시 `backtest_af_exact.py` 또는 `AntifragileBacktestRunner` 사용
- [ ] ML 모델(`models/af_ensemble/saved`) 존재 확인 후 실거래 시작
- [ ] 파라미터 변경 시 `config/af_params.py`만 수정 (다른 파일 직접 수정 금지)
- [ ] 실거래 전 최신 OHLCV 수집 → 백테스트 검증 → 실거래 시작
