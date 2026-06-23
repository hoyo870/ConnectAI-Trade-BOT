"""
Antifragile 전략 공통 파라미터/프리셋 — 단일 소스
strategies/backtest_engine.py + live_tools/live_trader.py 공유.
프리셋 변경 시 이 파일만 수정하면 됨.
"""

# ── 수수료 상수 ──────────────────────────────────────────────────────────────
TRADING_FEE = 0.00055   # Bybit taker 실측 0.055%
SLIPPAGE    = 0.00056   # 실거래 관측 평균 슬리피지 0.056%
FEE_TOTAL   = TRADING_FEE + SLIPPAGE  # 0.111%/side

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
    "rr_add":          0.15,  # 피라미딩 1회당 추가 비율
    "add_levels":      3,     # 최대 피라미딩 횟수
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
    "stable": {
        # 보수적 진입 + 타이트한 trail → 안정적 수익, 높은 hist 통과율
        "dt_rsi_lo": 30, "dt_rsi_hi": 60,
        "ut_rsi_lo": 42, "ut_rsi_hi": 70,
        "trail_atr_init": 1.5, "trail_atr_tight": 2.0,
        "add_levels": 4,
    },
    "aggressive": {
        # 넓은 진입 조건 + 빠른 trail → 거래 빈도↑, MDD↑
        "dt_rsi_lo": 15, "dt_rsi_hi": 85,
        "ut_rsi_lo": 25, "ut_rsi_hi": 95,
        "trail_atr_init": 0.8, "trail_atr_tight": 1.25,
        #"rr_base": 0.3, "rr_add": 0.2,
        "add_levels": 6,
    },
    "conservative": {
        # 엄격한 진입 + 넓은 trail → 거래 빈도↓, 손절 여유↑
        "dt_rsi_lo": 28, "dt_rsi_hi": 70,
        "ut_rsi_lo": 42, "ut_rsi_hi": 78,
        "trail_atr_init": 2.0, "trail_atr_tight": 2.5,
        "add_levels": 3,
    },
}


def get_preset(name: str) -> dict:
    """DEFAULT_PARAMS에 프리셋 오버라이드를 병합해서 반환."""
    base = dict(DEFAULT_PARAMS)
    base.update(PRESETS.get(name, {}))
    return base
