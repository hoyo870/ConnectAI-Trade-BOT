# DEPRECATED: ML-less Antifragile Strategy

> **폐기일**: 2026-06-22  
> **이유**: ML 필터 미포함 전략으로 실거래 운용 시 시드 손실 반복 발생.  
> 실거래는 반드시 `backtest_af_exact.py --model models/af_ensemble/saved/` 기준으로만 검증할 것.

---

## 폐기된 스크립트 목록

| 파일 | 역할 | 폐기 이유 |
|------|------|----------|
| `scripts/backtest_antifragile.py` | ML 없는 순수 rule-based 백테스트 엔진 | `AntifragileStrategy` 미사용, live_trader와 로직 불일치 |
| `scripts/backtest_af_ml.py` | ML 있으나 구형 엔진(`run_antifragile_ml`) | partial/flip/cooling/bb_sigma 불일치 — 2026-06-20 공식 deprecated |
| `scripts/batch_backtest.py` | `run_antifragile_ml` 기반 배치 | 위와 동일한 deprecated 엔진 의존 |
| `temp/batch_backtest.py` | 동일 deprecated 엔진 복사본 | |

---

## ML-less 전략 파라미터 (backtest_antifragile.py 기준)

```python
DEFAULT_PARAMS = {
    # AdaptRSI 임계값 (추세 방향별)
    "dt_rsi_lo": 22, "dt_rsi_hi": 65,   # 하락추세 (downtrend)
    "rg_rsi_lo": 40, "rg_rsi_hi": 85,   # 횡보 (ranging)
    "ut_rsi_lo": 40, "ut_rsi_hi": 85,   # 상승추세 (uptrend)

    # 포지션 사이징
    "rr_base":   0.10,   # 초기 리스크 비율
    "rr_add":    0.05,   # 피라미딩 추가 리스크
    "add_levels": 3,     # 최대 피라미딩 횟수

    # ATR trailing stop
    "trail_atr_init":  2.0,   # 초기 trail SL 배수
    "trail_atr_tight": 1.0,   # 피라미딩 후 tight SL 배수
    "atr_add_step":    0.5,   # 피라미딩 발동 ATR 배수
}
```

## 핵심 로직 (run_antifragile)

1. **진입 신호**: `RSI <= rsi_lo` (롱) / `RSI >= rsi_hi` (숏)
   - 추세(EMA 기반 `_trend_up`/`_trend_down`)에 따라 RSI 임계값 동적 변경
   - ATR < price × 0.15% 시 진입 차단 (횡보 구간 필터)
2. **청산**: ATR trailing stop — 포지션 방향 따라 peak 추적 후 trail_sl 히트 시 청산
3. **피라미딩**: favorable_move ≥ n × atr_add_step 조건 충족 시 rr_add 누적
4. **쿨링**: MDD > max_dd_cb 발동 시 cooling_bars 동안 진입 차단

## ML-less 전략과 실거래 전략의 차이

| 항목 | ML-less (폐기) | 실거래 (현행) |
|------|---------------|-------------|
| ML 진입 필터 | ❌ 없음 | ✅ AFEnsemble (LGBM+LSTM, theta=0.30) |
| 트렌드 판별 | EMA 단순 크로스 | BB σ=0.5 기반 |
| 절반 익절 | ❌ 없음 | ✅ RSI 극단 구간 50% 청산 |
| reverse_flip | ❌ 없음 | ✅ 반대 신호 2봉 연속 즉시 청산 |
| 방향 쿨링 | ❌ 없음 | ✅ 동일 방향 3연속 손실 시 20봉 차단 |
| 엔진 | `run_antifragile()` | `AntifragileStrategy` 클래스 |

## 2026-06-22 추가 정리 (Ralph 세션 수정사항 복원)

Ralph 세션에서 `DEFAULT_PARAMS`에 `flip_bars=1, cooling_bars=10, partial_enabled=True`를 활성화했으나,
이는 미검증 변경으로 live_trader 동작을 `backtest_af_ml.py` 기준에서 벗어나게 만들었음.

**복원 내용:**
- `DEFAULT_PARAMS`에서 flip/cooling/partial 관련 키 완전 제거
- `AntifragileStrategy`에서 flip detection, direction cooling, partial exit, require_trend_stable 코드 블록 제거
- live_trader.py에서 관련 state 필드 및 이벤트 핸들러 제거
- 삭제: `scripts/analyze_*.py`, `scripts/sweeps/`, `src/hybrid_engine.py`

**현행 AntifragileStrategy 로직 (단순화 완료):**
- RSI 기반 진입 (AdaptRSI)
- ATR trailing stop
- 피라미딩
- ML 진입 필터 (필수)
- ATR 필터 (atr < price × 0.15% 차단)

## 현행 유일한 백테스트 기준

```bash
# 반드시 이것만 사용
python scripts/backtest_af_exact.py --mode 2026 --coin all --model models/af_ensemble/saved/

# 특정 기간 직접 슬라이스
python scripts/backtest_af_exact.py --mode hist --coin all --model models/af_ensemble/saved/
```

## 백테스트 성과 기록 (ML-less, 참고용)

- BTC 2026 OOS: +132.4%, PF=8.364, MDD=2.7%
- hist 9/10 통과, 평균 +123.9%/3개월
- **단, ML 필터 미포함으로 실거래 괴리 발생 → 폐기**
