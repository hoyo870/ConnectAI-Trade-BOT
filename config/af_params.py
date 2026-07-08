"""
Antifragile 전략 공통 파라미터/프리셋 — 단일 소스
strategies/backtest_engine.py + live_tools/live_trader.py 공유.
프리셋 변경 시 이 파일만 수정하면 됨.
"""
import os

# ── 타임프레임 (AF_TIMEFRAME 미설정 시 5m = 기존 동작 불변) ──────────────────
# 5m 기준 상수를 TF에 비례 스케일. TREND_RULE=12×base(5m→1h, 60m→12h),
# FUNDING_BARS_8H=8h/base(5m→96, 60m→8). load_coin_raw가 이 TF로 리샘플.
_TF_MIN = {"5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60}
BASE_TF         = os.getenv("AF_TIMEFRAME", "5m")
TF_MINUTES      = _TF_MIN.get(BASE_TF, 5)
TREND_RULE      = f"{TF_MINUTES * 12}min"   # 5m→60min(=1h), 60m→720min(=12h)
FUNDING_BARS_8H = int(480 / TF_MINUTES)     # 5m→96, 60m→8

# ── 수수료 상수 ──────────────────────────────────────────────────────────────
TRADING_FEE = 0.00055   # Bybit taker 실측 0.055% (per side)
SLIPPAGE    = 0.000060  # measured avg slippage 0.0046% + 20% safety margin (per side)
FEE_TOTAL   = TRADING_FEE + SLIPPAGE  # total per side (≈ 0.061%). 왕복은 2×FEE_TOTAL.

# ── Funding 비용 (무기한 선물) ───────────────────────────────────────────────
# 8h마다 부과되는 funding을 보유봉수 비례로 비용 근사. 방향/부호와 무관하게 보수적
# 비용(절대값)으로 모델링. 실거래는 거래소 realized_pnl에 이미 반영됨.
FUNDING_RATE_8H = 0.0001  # 8시간당 ≈0.01% (BTC perp 평균 근사). cost_mult로 민감도 조정.

# ── 실거래 전용 상수 ─────────────────────────────────────────────────────────
EMERGENCY_SL_ATR = 6.0  # intrabar wick SL 조기 체결 방지용 넓은 SL 배수

# ── 기본 파라미터 (prod 프리셋 = 실거래 기본값) ─────────────────────────────
DEFAULT_PARAMS = {
    # ── RSI 진입 임계값 (STRATEGY_ANTIFRAGILE.md 2026-06-10 기준) ────────────
    "dt_rsi_lo":       22,    # 하락추세: 롱 진입 RSI 임계값
    "dt_rsi_hi":       65,    # 하락추세: 숏 진입 RSI 임계값
    "rg_rsi_lo":       30,    # 횡보:     롱 진입 RSI 임계값
    "rg_rsi_hi":       70,    # 횡보:     숏 진입 RSI 임계값
    "ut_rsi_lo":       40,    # 상승추세: 롱 진입 RSI 임계값 (35→40, 2026-06-10)
    "ut_rsi_hi":       85,    # 상승추세: 숏 진입 RSI 임계값 (78→85, 2026-06-10)
    # ── 청산: ATR Trailing Stop ───────────────────────────────────────────────
    "trail_atr_init":  1.0,   # 초기 trailing stop 거리 (0.5→1.0, 2026-06-09)
    "trail_atr_tight": 1.5,   # 피라미딩 후 tight trailing (0.8→1.5, 2026-06-09)
    # ── 포지션 사이징 ─────────────────────────────────────────────────────────
    "rr_base":         0.10,  # 초기 자본 위험 비율
    "rr_add":          0.15,  # 피라미딩 1회당 추가 비율 (add_levels=0이면 미사용)
    # 피라미딩 비활성 (0): add-to-winner + peak 기준 trailing stop 조합이 가중평균 손실
    # 쪽에서 청산되어 구조적 손실 — project_pnl_accounting_fix 검증. 단일진입으로 기본
    # 신호 엣지부터 정직하게 검증. 재설계 전까지 0 유지 (코드 보존, 값만 되돌리면 복원).
    "add_levels":      0,     # 최대 피라미딩 횟수 (3→0, 2026-06-24)
    "atr_add_step":    0.5,   # 피라미딩 트리거 (유리방향 X×ATR마다)
    # ── ML 필터 ───────────────────────────────────────────────────────────────
    "ml_threshold":    None,  # float 0-1 if ML active, None = disabled
}

# ── 프리셋 (backtest + live_trader 공통) ─────────────────────────────────────
# 각 프리셋은 DEFAULT_PARAMS에서 변경되는 키만 포함.
# live_trader는 bb_sigma=0.5를 추가로 적용 (백테스트 엔진은 bb_sigma 미사용).
PRESETS: dict[str, dict] = {
    "prod": {
        # DEFAULT_PARAMS 그대로 사용
    },
    "candidate": {
        # 2026-06-24 과적합방지 스윕 검증 후보 (project_param_sweep): δ=10 선별강화 + trail 2.0.
        # ML θ=0.45는 .env ML_THRESHOLD로 적용. 4코인 held-out TEST + 비용2배 통과.
        # forward(paper) 검증용. 저레버리지(.env LEVERAGE=3)로 가동 권장.
        "dt_rsi_lo": 12, "dt_rsi_hi": 75,
        "rg_rsi_lo": 20, "rg_rsi_hi": 80,
        "ut_rsi_lo": 30, "ut_rsi_hi": 95,
        "trail_atr_init": 2.0,
        "add_levels": 0,
    },
    "stable": {
        # 보수적 진입 + 타이트한 trail → 안정적 수익, 높은 hist 통과율
        "dt_rsi_lo": 30, "dt_rsi_hi": 60,
        "ut_rsi_lo": 42, "ut_rsi_hi": 70,
        "trail_atr_init": 1.5, "trail_atr_tight": 2.0,
        "add_levels": 0,   # 피라미딩 비활성 (4→0, 2026-06-24) — DEFAULT_PARAMS 주석 참조
    },
    "aggressive": {
        # 넓은 진입 조건 + 빠른 trail → 거래 빈도↑, MDD↑
        "dt_rsi_lo": 15, "dt_rsi_hi": 85,
        "ut_rsi_lo": 25, "ut_rsi_hi": 95,
        "trail_atr_init": 0.8, "trail_atr_tight": 1.25,
        #"rr_base": 0.3, "rr_add": 0.2,
        "add_levels": 0,   # 피라미딩 비활성 (6→0, 2026-06-24) — DEFAULT_PARAMS 주석 참조
    },
    "conservative": {
        # 엄격한 진입 + 넓은 trail → 거래 빈도↓, 손절 여유↑
        "dt_rsi_lo": 28, "dt_rsi_hi": 70,
        "ut_rsi_lo": 42, "ut_rsi_hi": 78,
        "trail_atr_init": 2.0, "trail_atr_tight": 2.5,
        "add_levels": 0,   # 피라미딩 비활성 (3→0, 2026-06-24) — DEFAULT_PARAMS 주석 참조
    },
}


def get_preset(name: str) -> dict:
    """DEFAULT_PARAMS에 프리셋 오버라이드를 병합해서 반환."""
    base = dict(DEFAULT_PARAMS)
    base.update(PRESETS.get(name, {}))
    return base
