"""
AntifragileStrategy — 순수 신호 로직 (거래소·로깅 없음)

live_tools/live_trader.py와 scripts/backtest_af_exact.py가 이 클래스를 공유하여
실거래 ↔ 백테스트 로직 드리프트를 원천 방지.

반환 events 타입:
  {"type": "close",   "reason": str, "pnl_raw": float, "rr": float, "direction": int}
  {"type": "pyramid", "new_rr": float, "new_avg": float, "add_cnt": int}
  {"type": "entry",   "direction": int, "trail_sl": float, "avg_entry": float}
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.af_params import DEFAULT_PARAMS, FEE_TOTAL


class AntifragileStrategy:
    """
    AdaptRSI + ATR Trailing Stop + 피라미딩 + ML 진입 필터 전략 상태 머신.

    - reset() / load_state()로 초기화
    - process_tick(price, atr, rsi, trend_up, trend_down) → events 기반 결과
    - get_state() → live_trader state JSON 호환 dict
    """

    def __init__(self, params: dict | None = None, ml_filter=None):
        self.p         = {**DEFAULT_PARAMS, **(params or {})}
        self.ml_filter = ml_filter
        self._df_context = None
        self.lev       = self.p.get("leverage", 7)
        self.fee       = self.p.get("fee_total", FEE_TOTAL)
        self.reset()

    # ── 상태 초기화 ────────────────────────────────────────────────────────────

    def reset(self):
        self.pos       = 0
        self.avg_entry = 0.0
        self.entry_atr = 0.0
        self.rr        = self.p["rr_base"]
        self.add_cnt   = 0
        self.trail_sl  = 0.0
        self.peak_px   = 0.0
        self.entry_bar = 0
        self.prev_trend = "RG"
        self._bar_idx  = 0
        self._df_idx   = 0

    # ── state JSON 직렬화 / 복원 ────────────────────────────────────────────────

    def get_state(self) -> dict:
        """live_trader state dict에 병합 가능한 AF 관련 필드 반환."""
        return {
            "avg_entry_price":  self.avg_entry,
            "af_entry_atr":     self.entry_atr,
            "af_current_rr":    self.rr,
            "entry_rr":         self.rr,
            "af_pyramid_count": self.add_cnt,
            "af_trail_sl":      self.trail_sl,
            "af_peak_price":    self.peak_px,
            "entry_bar":        self.entry_bar,
            "last_trend":       self.prev_trend,
        }

    def load_state(self, state: dict):
        """live_trader state dict에서 전략 상태 복원."""
        self.pos       = int(state.get("position", 0))
        self.avg_entry = float(state.get("avg_entry_price") or state.get("entry_price") or 0.0)
        self.entry_atr = float(state.get("af_entry_atr",    0.0) or 0.0)
        self.rr        = float(state.get("af_current_rr")   or state.get("entry_rr") or self.p["rr_base"])
        self.add_cnt   = int(state.get("af_pyramid_count",   0))
        self.trail_sl  = float(state.get("af_trail_sl",      0.0))
        self.peak_px   = float(state.get("af_peak_price",    0.0))
        self.entry_bar = int(state.get("entry_bar",           0))
        self.prev_trend = state.get("last_trend", "RG")
        self._bar_idx  = int(state.get("current_bar", self._bar_idx)) - 1

    # ── ML 필터 컨텍스트 ───────────────────────────────────────────────────────

    def update_context(self, df, idx: int | None = None):
        """ML 컨텍스트 df 제공. idx=None이면 마지막 봉(실거래), 정수면 해당 봉(백테스트)."""
        self._df_context = df
        self._df_idx = idx if idx is not None else len(df) - 1

    # ── 핵심: 한 봉 처리 ──────────────────────────────────────────────────────

    def process_tick(
        self,
        price: float,
        atr: float,
        rsi: float,
        trend_up: bool,
        trend_down: bool,
    ) -> dict:
        """
        하나의 봉을 처리하고 발생 이벤트 목록을 반환.

        Returns:
            {
              "events": list[dict],
              "pos":       int,
              "trail_sl":  float,
              "avg_entry": float,
              "rr":        float,
              "add_cnt":   int,
              "trend":     str,
            }
        """
        self._bar_idx += 1
        bar_idx = self._bar_idx
        events: list[dict] = []

        trend    = self._trend_str(trend_up, trend_down)
        rsi_lo, rsi_hi = self._get_thresholds(trend)

        # ── 청산 / trailing stop + 피라미딩 ───────────────────────────────────
        if self.pos != 0:
            hold     = bar_idx - self.entry_bar
            hit_stop = (self.pos == 1 and price <= self.trail_sl) or \
                       (self.pos == -1 and price >= self.trail_sl)
            timeout  = hold >= self.p.get("max_hold_bars", 288)

            if hit_stop or timeout:
                reason    = "trail_SL" if hit_stop else "timeout"
                pnl_raw   = self.pos * (price - self.avg_entry) / (self.avg_entry + 1e-9)
                rr_closed = self.rr
                closed_pos = self.pos

                events.append({
                    "type": "close", "reason": reason,
                    "pnl_raw": pnl_raw, "rr": rr_closed, "direction": closed_pos,
                    "entry_price": self.avg_entry,
                    "exit_price":  price,
                })

                self.pos = 0; self.avg_entry = 0.0; self.entry_atr = 0.0
                self.rr = self.p["rr_base"]; self.add_cnt = 0
                self.trail_sl = 0.0; self.peak_px = 0.0

            else:
                # Trailing stop 갱신
                eff_atr    = max(atr, self.entry_atr * 0.6)
                trail_mult = self.p["trail_atr_tight"] if self.add_cnt > 0 else self.p["trail_atr_init"]
                if self.pos == 1:
                    self.peak_px  = max(self.peak_px, price)
                    self.trail_sl = max(self.trail_sl, self.peak_px - trail_mult * eff_atr)
                else:
                    self.peak_px  = min(self.peak_px, price)
                    self.trail_sl = min(self.trail_sl, self.peak_px + trail_mult * eff_atr)

                # 피라미딩 (RSI 극단 구간 차단)
                favorable  = self.pos * (price - self.avg_entry) / (self.entry_atr + 1e-9)
                next_lvl   = (self.add_cnt + 1) * self.p["atr_add_step"]
                rsi_ok_pyr = (self.pos == 1 and rsi < rsi_hi) or (self.pos == -1 and rsi > rsi_lo)
                if self.add_cnt < self.p["add_levels"] and favorable >= next_lvl and rsi_ok_pyr:
                    old_qty    = self.rr / self.p["rr_base"]
                    add_rr     = self.p["rr_add"]
                    new_qty    = old_qty + add_rr / self.p["rr_base"]
                    self.avg_entry = (self.avg_entry * old_qty + price * (add_rr / self.p["rr_base"])) / new_qty
                    self.rr       += add_rr
                    self.add_cnt  += 1
                    tight = self.p["trail_atr_tight"]
                    if self.pos == 1: self.trail_sl = max(self.trail_sl, price - tight * atr)
                    else:             self.trail_sl = min(self.trail_sl, price + tight * atr)
                    events.append({
                        "type": "pyramid",
                        "new_rr": self.rr, "new_avg": self.avg_entry, "add_cnt": self.add_cnt,
                    })

        # ── 신규 진입 ────────────────────────────────────────────────────────
        if self.pos == 0:
            long_ok  = rsi <= rsi_lo
            short_ok = (rsi >= rsi_hi) and not long_ok

            direction = 1 if long_ok else (-1 if short_ok else 0)

            if direction != 0 and atr < price * 0.0015:
                direction = 0

            if direction != 0 and self.ml_filter is not None and self._df_context is not None:
                from models.af_ensemble.feature_extractor import extract_snapshot, extract_sequence
                loss_streak = 0
                snap = extract_snapshot(self._df_context, self._df_idx, direction, rsi_lo, rsi_hi, loss_streak)
                seq  = extract_sequence(self._df_context, self._df_idx)
                if not self.ml_filter.should_enter(snap, seq):
                    direction = 0

            if direction != 0:
                self.pos       = direction
                self.avg_entry = price
                self.entry_atr = atr
                self.rr        = self.p["rr_base"]
                self.add_cnt   = 0
                self.peak_px   = price
                self.entry_bar = bar_idx
                init_atr       = self.p["trail_atr_init"]
                self.trail_sl  = (price - init_atr * atr) if direction == 1 \
                                 else (price + init_atr * atr)
                events.append({
                    "type": "entry", "direction": direction,
                    "trail_sl": self.trail_sl, "avg_entry": self.avg_entry,
                })

        self.prev_trend = trend
        return {
            "events":    events,
            "pos":       self.pos,
            "trail_sl":  self.trail_sl,
            "avg_entry": self.avg_entry,
            "rr":        self.rr,
            "add_cnt":   self.add_cnt,
            "trend":     trend,
        }

    # ── 헬퍼 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _trend_str(trend_up: bool, trend_down: bool) -> str:
        return "DN" if trend_down else ("UP" if trend_up else "RG")

    def _get_thresholds(self, trend: str) -> tuple[float, float]:
        p = self.p
        if trend == "DN": return p["dt_rsi_lo"], p["dt_rsi_hi"]
        if trend == "UP": return p["ut_rsi_lo"], p["ut_rsi_hi"]
        return p["rg_rsi_lo"], p["rg_rsi_hi"]
