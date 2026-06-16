# 스크립트 사용 가이드

## 구조

```
scripts/
├── backtest_antifragile.py   # 핵심 백테스트 엔진
├── backtest.py               # 구버전 DL v17 백테스트 (참고용)
├── analyze_*.py              # 분석 유틸
└── sweeps/                   # 파라미터 스윕 스크립트
    ├── param_sweep.py        # trail × RSI 그리드 스윕
    ├── rsi_sweep.py          # RSI 축별 독립 스윕 (dt/rg/ut)
    ├── preset_sweep.py       # 종목 × 프리셋 비교
    └── leverage_sweep.py     # 레버리지별 수익/MDD 선형성 체크

config/
└── af_params.py              # 프리셋/수수료 상수 단일 소스 (backtest + live_trader 공용)

live_tools/                   # 실거래 봇 (프로덕션)
temp/scripts/                 # 실험용 스크립트 (33~56번, 참고용)
```

---

## 백테스트 기본 규칙

- **2026 OOS 기간**: `2026-01-01 ~ 2026-05-31` (6월 이후는 실거래 기간)
- **수수료**: `FEE_TOTAL=0.111%/side` (Bybit taker 0.055% + 실측 슬리피지 0.056%)
- **판정 기준 3/3**: 수익률 > 0%, TPD ≥ 1.5, Top-5 제거 후 수익 > 0%
- **hist 통과**: 91일 × 10창 중 7창 이상 판정 통과

---

## scripts/backtest_antifragile.py

Antifragile Trailing Stop 전략 핵심 엔진.

```bash
# 단일 코인 2026 OOS
python scripts/backtest_antifragile.py --coin btc --mode 2026
python scripts/backtest_antifragile.py --coin all --mode 2026

# 역사적 검증 (91일 × 10창)
python scripts/backtest_antifragile.py --coin all --mode random --seed 42

# 2026-06 단기 검증
python scripts/backtest_antifragile.py --coin all --mode june2026

# 프리셋 지정
python scripts/backtest_antifragile.py --coin all --mode 2026 --preset stable
python scripts/backtest_antifragile.py --coin all --mode 2026 --preset aggressive
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--coin` | `btc` | `btc` \| `eth` \| `sol` \| `xrp` \| `all` |
| `--mode` | `2026` | `2026` \| `random` \| `both` \| `june2026` |
| `--preset` | _(없음)_ | `prod` \| `stable` \| `aggressive` \| `conservative` |
| `--windows` | `10` | 랜덤 검증 창 수 |
| `--seed` | `42` | 랜덤 시드 |
| `--trail-init` | `1.8` | 초기 trailing ATR 배수 |
| `--trail-tight` | `2.0` | 피라미딩 후 trailing ATR 배수 |

---

## scripts/sweeps/

### preset_sweep.py — 종목 × 프리셋 비교

```bash
python scripts/sweeps/preset_sweep.py              # BTC (기본)
python scripts/sweeps/preset_sweep.py --all        # 4종목 전체
python scripts/sweeps/preset_sweep.py --coin eth   # ETH만
```

### leverage_sweep.py — 레버리지별 수익/MDD 비교

```bash
python scripts/sweeps/leverage_sweep.py                         # 기본 1/3/5/7/10x
python scripts/sweeps/leverage_sweep.py --leverages 5 7 10     # 특정 레버리지만
python scripts/sweeps/leverage_sweep.py --preset stable         # 프리셋 지정
```

### param_sweep.py — trail × RSI 그리드 스윕

```bash
python scripts/sweeps/param_sweep.py --phase 1    # trail_init × trail_tight
python scripts/sweeps/param_sweep.py --phase 2    # RSI 조합
python scripts/sweeps/param_sweep.py --phase all  # 전체
```

### rsi_sweep.py — RSI 축별 독립 스윕

```bash
python scripts/sweeps/rsi_sweep.py --phase 2a   # dt_rsi_lo × dt_rsi_hi
python scripts/sweeps/rsi_sweep.py --phase 2b   # ut_rsi_lo × ut_rsi_hi
python scripts/sweeps/rsi_sweep.py --phase 2c   # rg_rsi_lo × rg_rsi_hi
python scripts/sweeps/rsi_sweep.py --phase 3    # 프리셋별 trail 검증
python scripts/sweeps/rsi_sweep.py --phase all  # 전체
```

---

## config/af_params.py — 프리셋 관리

프리셋 변경 시 이 파일만 수정하면 backtest + live_trader 양쪽에 반영됩니다.

| 프리셋 | dt_lo/hi | ut_lo/hi | trail init/tight | 특성 |
|--------|----------|----------|-----------------|------|
| `prod` | 28/60 | 42/75 | 1.8/2.0 | 기본값 (실거래 기준) |
| `stable` | 30/60 | 42/70 | 1.5/2.0 | 보수적 진입, hist 통과율↑ |
| `aggressive` | 25/60 | 42/78 | 0.8/1.5 | 넓은 진입, 거래빈도↑ |
| `conservative` | 28/70 | 42/78 | 2.0/2.5 | 엄격한 진입, 손절여유↑ |

```python
from config.af_params import get_preset, PRESETS, FEE_TOTAL
cfg = get_preset("prod")   # DEFAULT_PARAMS + prod 오버라이드 병합
```

---

## live_tools/ — 실거래 봇

| 파일 | 역할 |
|------|------|
| `live_trader.py` | 메인 실거래 루프 (5분봉 기반) |
| `exchange_client.py` | Bybit 선물 API 래퍼 |
| `run.py` | 봇 시작/재시작 진입점 |
| `bot_manage.py` | 프로세스 관리 유틸 |
| `telegram_notifier.py` | 텔레그램 알림 + `/stop` kill switch |
| `data_pipeline.py` | OHLCV → 지표 계산 |

```bash
# 실거래 봇 실행
python live_tools/run.py

# 대시보드 확인 (기본 포트 8765)
open http://localhost:8765
```

---

## deploy/ — 배포 설정

```bash
# macOS launchd (자동 재시작)
cp deploy/com.connectai.tradebot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.connectai.tradebot.plist

# Linux systemd
sudo cp deploy/live_trader.service /etc/systemd/system/
sudo systemctl enable live_trader && sudo systemctl start live_trader

# nohup (간단)
bash deploy/run_paper.sh
```

---

## 공통 규칙

- 프리셋/파라미터 변경은 `config/af_params.py` 에서만
- 신규 스크립트는 `temp/scripts/`에서 개발 → 검증 완료 후 `scripts/` 또는 `scripts/sweeps/`로 승격
- **모든 백테스트에 TPD 필수 포함** (기준: TPD ≥ 1.5)
