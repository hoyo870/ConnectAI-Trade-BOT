"""
[RL] 커스텀 트레이딩 환경 (Gymnasium + Action Masking)

Action Space (Discrete 3):
  0 = HOLD   — 포지션 유지 / 대기
  1 = LONG   — 매수 진입 (또는 Short → Long 전환)
  2 = SHORT  — 매도 진입 (또는 Long → Short 전환)

Observation Space:
  [signal_long, signal_short, signal_context,   ← 전문가 시그널 (3)
   position_long, position_short, position_flat,  ← 현재 포지션 원-핫 (3)
   unrealized_pnl,                               ← 미실현 손익률 (1)
   hold_ratio]                                    ← 포지션 보유 비율 (1)
  총 8차원

Action Masking:
  - 이미 Long 포지션 → LONG 행동 불가 (action_mask[1]=0)
  - 이미 Short 포지션 → SHORT 행동 불가 (action_mask[2]=0)
  - signal_long < MASK_THR  → LONG 행동 불가
  - signal_short < MASK_THR → SHORT 행동 불가
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Action Masking 임계값
MASK_THRESHOLD = 0.40     # 0.45 → 0.40: 신호 mean(0.37) 대비 과도한 마스킹 방지

# 거래 비용
TRADING_FEE    = 0.0005   # 0.05% (메이커/테이커 평균)
SLIPPAGE       = 0.0002   # 0.02% 슬리피지 가정

# 보상 설계 파라미터
HOLD_PENALTY   = -0.001   # 포지션 없이 HOLD 시 패널티 (거래 유도)
DRAWDOWN_COEFF = 2.0      # MDD 패널티 강도


def _mask_fn(env) -> np.ndarray:
    """ActionMasker 래퍼에 전달할 마스크 함수."""
    return env.action_masks()


class CryptoTradingEnv(gym.Env):
    """
    5분봉 신호 기반 암호화폐 트레이딩 환경.

    Parameters
    ----------
    df          : signal_long/short/context + close 컬럼을 가진 DataFrame
    window_size : 에피소드 길이 (봉 수). None이면 전체 데이터 사용.
    initial_capital : 초기 자본 (USDT 기준)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int | None = 2000,
        initial_capital: float = 10_000.0,
        risk_ratio: float      = 0.3,    # 시드 투입 비율 (0.1~0.5)
        stop_loss: float       = 0.05,   # 5% pnl_lev — RL이 자체 판단 학습
        take_profit: float     = 0.09,  # 9% pnl_lev — 큰 움직임 포착
        leverage: float        = 1.5,    # 레버리지 1.5x
    ):
        super().__init__()

        self.df              = df.reset_index(drop=True)
        self.window_size     = window_size if window_size else len(df)
        self.initial_capital = initial_capital
        self.risk_ratio      = float(np.clip(risk_ratio, 0.1, 0.5))   # 10%~50% 제한
        self.stop_loss       = stop_loss
        self.take_profit     = take_profit
        self.leverage        = float(np.clip(leverage, 1.0, 5.0))     # 최대 5배

        # ── 공간 정의 ──────────────────────────────────────────────────────────
        self.action_space      = spaces.Discrete(3)  # HOLD / LONG / SHORT
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # 내부 상태 초기화
        self._reset_state(0)

    # ── 초기화 ─────────────────────────────────────────────────────────────────

    def _reset_state(self, start_idx: int):
        self.current_step     = start_idx
        self.end_step         = min(start_idx + self.window_size, len(self.df) - 1)
        self.capital          = self.initial_capital
        self.position         = 0          # -1=Short, 0=Flat, 1=Long
        self.entry_price      = 0.0
        self.hold_steps       = 0
        self.peak_capital     = self.initial_capital
        self.total_trades     = 0
        self.trade_log: list  = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 랜덤 시작점 (학습 다양성)
        max_start = max(0, len(self.df) - self.window_size - 1)
        start_idx = self.np_random.integers(0, max(1, max_start))
        self._reset_state(int(start_idx))

        obs  = self._get_obs()
        info = {}
        return obs, info

    # ── 관측값 ─────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]

        sig_long    = float(row.get("signal_long",    0.5))
        sig_short   = float(row.get("signal_short",   0.5))
        sig_context = float(row.get("signal_context", 0.5))

        pos_long  = float(self.position ==  1)
        pos_short = float(self.position == -1)
        pos_flat  = float(self.position ==  0)

        unrealized_pnl = self._unrealized_pnl()
        hold_ratio     = min(self.hold_steps / 100.0, 1.0)
        returns_1      = float(row.get("returns_1", 0.0))
        atr_pct        = float(row.get("atr_pct",   0.0))

        return np.array([
            sig_long, sig_short, sig_context,
            pos_long, pos_short, pos_flat,
            unrealized_pnl,
            hold_ratio,
            returns_1,
            atr_pct,
        ], dtype=np.float32)

    # ── Action Masking ─────────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """
        MaskablePPO가 호출하는 메서드.
        True = 허용, False = 마스킹(불가).
        """
        row       = self.df.iloc[self.current_step]
        sig_long  = float(row.get("signal_long",  0.5))
        sig_short = float(row.get("signal_short", 0.5))

        mask = np.ones(3, dtype=bool)

        # 이미 동일 포지션 진입 불가
        if self.position ==  1:
            mask[1] = False   # LONG 중복 불가
        if self.position == -1:
            mask[2] = False   # SHORT 중복 불가

        # 시그널이 임계값 미달이면 해당 방향 진입 불가
        if sig_long  < MASK_THRESHOLD:
            mask[1] = False
        if sig_short < MASK_THRESHOLD:
            mask[2] = False

        # 최소 1개 행동은 항상 허용 (HOLD=0은 항상 True)
        return mask

    # ── 스텝 ───────────────────────────────────────────────────────────────────

    def step(self, action: int):
        assert self.action_space.contains(action)

        current_price = float(self.df.iloc[self.current_step]["close"])
        reward        = 0.0

        # ── 행동 처리 ──────────────────────────────────────────────────────────
        if action == 1:   # LONG 진입
            reward += self._close_position(current_price)   # 기존 포지션 청산
            self._open_position(1, current_price)
        elif action == 2: # SHORT 진입
            reward += self._close_position(current_price)
            self._open_position(-1, current_price)
        else:             # HOLD — 포지션 없이 대기하면 패널티
            if self.position == 0:
                reward += HOLD_PENALTY

        # ── 손절/익절/마진콜 자동 청산 (레버리지 반영) ───────────────────────
        if self.position != 0:
            upnl_lev = self._unrealized_pnl() * self.leverage
            margin_call = upnl_lev <= -0.9      # 레버리지 마진콜 (-90%)
            sl_hit      = upnl_lev <= -self.stop_loss
            tp_hit      = upnl_lev >= self.take_profit
            if margin_call or sl_hit or tp_hit:
                reward += self._close_position(current_price)

        # ── 다음 봉으로 이동 ───────────────────────────────────────────────────
        self.current_step += 1
        self.hold_steps   += 1 if self.position != 0 else 0

        next_price = float(self.df.iloc[self.current_step]["close"])

        # ── 미실현 손익 보상 (레버리지 × 투입비율) ────────────────────────────
        if self.position != 0:
            step_ret  = (next_price - current_price) / (current_price + 1e-9)
            reward   += self.position * step_ret * self.risk_ratio * self.leverage

        # ── MDD 패널티 ─────────────────────────────────────────────────────────
        equity = self.capital * (1 + self._unrealized_pnl() * self.risk_ratio * self.leverage)
        self.peak_capital = max(self.peak_capital, equity)
        drawdown = (self.peak_capital - equity) / (self.peak_capital + 1e-9)
        reward  -= DRAWDOWN_COEFF * drawdown * 0.01

        # ── 종료 판단 ──────────────────────────────────────────────────────────
        terminated = (self.current_step >= self.end_step)
        truncated  = (self.capital <= self.initial_capital * 0.1)  # 90% 손실 시 강제 종료

        if terminated or truncated:
            reward += self._close_position(next_price)  # 최종 청산

        obs  = self._get_obs()
        info = {
            "capital":       self.capital,
            "position":      self.position,
            "total_trades":  self.total_trades,
            "step":          self.current_step,
        }
        return obs, float(reward), terminated, truncated, info

    # ── 포지션 관리 ────────────────────────────────────────────────────────────

    def _open_position(self, direction: int, price: float):
        fee = price * TRADING_FEE + price * SLIPPAGE
        self.position    = direction
        self.entry_price = price + fee * direction
        self.hold_steps  = 0

    def _close_position(self, price: float) -> float:
        if self.position == 0:
            return 0.0

        fee   = price * TRADING_FEE + price * SLIPPAGE
        close = price - fee * self.position
        pnl   = self.position * (close - self.entry_price) / (self.entry_price + 1e-9)
        # 실효 손익 = 가격변동률 × 레버리지 × 투입비율 (단, -100% 초과 손실 방지)
        effective_pnl = max(pnl * self.leverage * self.risk_ratio, -self.risk_ratio)

        self.capital    *= (1 + effective_pnl)
        self.position    = 0
        self.entry_price = 0.0
        self.hold_steps  = 0
        self.total_trades += 1

        self.trade_log.append({
            "step":           self.current_step,
            "pnl":            effective_pnl,
            "pnl_raw":        pnl,
            "leverage":       self.leverage,
            "risk_ratio":     self.risk_ratio,
        })
        return float(effective_pnl)

    def _unrealized_pnl(self) -> float:
        if self.position == 0:
            return 0.0
        current_price = float(self.df.iloc[self.current_step]["close"])
        return self.position * (current_price - self.entry_price) / (self.entry_price + 1e-9)

    # ── 렌더링 ─────────────────────────────────────────────────────────────────

    def render(self):
        equity = self.capital * (1 + self._unrealized_pnl())
        pos_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}[self.position]
        print(
            f"Step {self.current_step:5d} | "
            f"Equity: {equity:10.2f} | "
            f"Position: {pos_str:5s} | "
            f"Trades: {self.total_trades}"
        )
