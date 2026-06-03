# 스크립트 사용 가이드

## scripts/ — 프로덕션 스크립트

### backtest_antifragile.py ⭐ (신규)

Antifragile Trailing Stop 전략 백테스트. `STRATEGY=antifragile` 활성화 전 검증용.

```bash
# 2026 기본 검증
python scripts/backtest_antifragile.py --coin btc --mode 2026
python scripts/backtest_antifragile.py --coin eth --mode 2026
python scripts/backtest_antifragile.py --coin both --mode 2026

# 역사적 랜덤 10회 검증
python scripts/backtest_antifragile.py --coin both --mode random --seed 42

# 파라미터 조정
python scripts/backtest_antifragile.py --coin both --mode 2026 --require-bb     # BB 이탈 조건 추가
python scripts/backtest_antifragile.py --coin both --mode 2026 --trail-init 0.3  # 좁은 초기 trail
```

**주요 파라미터:**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--coin` | `btc` | `btc` \| `eth` \| `both` |
| `--mode` | `2026` | `2026` \| `random` \| `both` |
| `--windows` | `10` | 랜덤 검증 횟수 |
| `--seed` | `42` | 랜덤 시드 |
| `--window-days` | `91` | 랜덤 윈도우 길이 (일) |
| `--require-bb` | `False` | BB 밴드 이탈 조건 추가 |
| `--trail-init` | `0.5` | 초기 trailing ATR 배수 |
| `--trail-tight` | `0.8` | 피라미딩 후 trailing ATR 배수 |
| `--add-step` | `0.5` | 피라미딩 트리거 ATR 배수 |

**검증 기준 (trailing 전략용):**
- 수익률 > 0%
- TPD ≥ 1.5
- Top-5 제거 후 수익 > 0%

---

### backtest.py — DL v17 전략

```bash
# 2026 기본 검증
python scripts/backtest.py --coin both --mode 2026

# 역사적 랜덤 10회
python scripts/backtest.py --coin both --mode random --windows 10 --seed 42

# ETH Pullback 전략
python scripts/backtest.py --coin eth --mode 2026 --strategy pullback

# 특정 구간
python scripts/backtest.py --coin btc --mode custom --start 2024-01-01 --end 2024-04-01
```

**주요 파라미터:**

| 파라미터 | 값 | 기본값 | 설명 |
|---|---|---|---|
| `--coin` | `btc` \| `eth` \| `both` | `both` | 대상 코인 |
| `--mode` | `2026` \| `random` \| `custom` | `2026` | 구간 모드 |
| `--strategy` | `instant` \| `pullback` | `instant` | 진입 전략 |
| `--windows` | int | `10` | 랜덤 구간 수 |
| `--seed` | int | `42` | 랜덤 시드 |
| `--start` / `--end` | `YYYY-MM-DD` | — | custom 모드 구간 |

**판정 기준 (3/3 통과):**
- 승률 ≥ 45%
- 일일 거래수(TPD) ≥ 1.5
- 일 평균 수익률 ≥ 1.0%

---

### 분석 스크립트

```bash
# BTC 펀딩비 포함 수익률 분석
python scripts/analyze_funding.py

# ETH 펀딩비 + 하이브리드 게이트 분석
python scripts/analyze_funding_eth.py

# DL v17 Top-N 제거 outlier 분석
python scripts/analyze_top_trades.py
```

---

## src/ — 핵심 라이브러리

| 파일 | 역할 |
|------|------|
| `live_trader.py` | 실거래 루프 (`STRATEGY` 환경변수로 전략 선택) |
| `data_pipeline.py` | OHLCV → 지표 30+개, 스케일링, 레이블 |
| `expert_models.py` | TCN + Multi-Head Attention 아키텍처 |
| `hybrid_engine.py` | DL v17 백테스트 엔진 + `compute_metrics()` |
| `signal_extractor.py` | 배치 DL 시그널 추출 |
| `exchange_client.py` | Bybit 선물 API 래퍼 |
| `telegram_notifier.py` | 텔레그램 알림 + `/stop` kill switch |
| `data_fetcher.py` | OHLCV 원시 데이터 수집 |

---

## temp/scripts/ — 연구용 스크립트 (참고용)

| 파일 | 내용 | 결과 |
|------|------|------|
| `22_backtest_pullback.py` | Pullback 전략 검증 | BTC -72% ❌ |
| `23_pullback_improved.py` | Pullback + BTC Gate 6변형 | 전체 탈락 ❌ |
| `24_style_b_improved.py` | Style B (RSI+BB) 7변형 | AdaptRSI +12.9% ⚠️ |
| `25_metalabel_train.py` | Triple-Barrier meta-label | label 불균형 ❌ |
| `26_style_b_adptrsi_hist.py` | AdaptRSI 역사적 검증 | hist 0/10 ❌ |
| `27_antifragile_trailing.py` | **Antifragile 발견** | BTC +132% ✅ |
| `28_vol_expansion_breakout.py` | 변동성 압축 돌파 | -27% ❌ |
| `29_atr_sl_tp_sweep.py` | ATR SL/TP 스윕 | -2.4% ⚠️ |
| `30_ratio_trade.py` | ETH/BTC 비율 거래 | WR<5% ❌ |
| `31_triple_barrier_lgbm.py` | Triple-Barrier + LightGBM | 51.5% 정확도 ⚠️ |
| `32_antifragile_hist.py` | Antifragile 역사적 검증 | hist 9/10 ✅ → scripts/ 승격 |
| `backtest_style_a~e.py` | A~E 스타일 탐색 | 모두 DL v17 미만 ❌ |

---

## deploy/ — 배포 설정

```bash
# macOS launchd (자동 재시작)
cp deploy/com.connectai.tradebot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.connectai.tradebot.plist

# Linux systemd
sudo cp deploy/live_trader.service /etc/systemd/system/
sudo systemctl enable live_trader
sudo systemctl start live_trader

# nohup (간단)
bash deploy/run_paper.sh
```

---

## 공통 규칙

- **모든 백테스트 결과에 TPD 필수 포함** (기준: TPD ≥ 1.5)
- 신규 스크립트는 `temp/scripts/`에서 개발 후 검증 완료 시 `scripts/`로 승격
- `src/` 파일은 프로덕션 코드 — 신중하게 수정
