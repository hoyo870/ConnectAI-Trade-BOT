"""
[RL] MaskablePPO 에이전트 학습 로직

의존성:
  pip install stable-baselines3 sb3-contrib gymnasium
"""

import os
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnNoModelImprovement,
    CheckpointCallback, BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from trading_env import CryptoTradingEnv, _mask_fn


# ── 설정 ──────────────────────────────────────────────────────────────────────
MODEL_SAVE_DIR = "models"
LOG_DIR        = "logs/rl"

RL_CONFIG = dict(
    learning_rate   = 3e-4,
    n_steps         = 2048,
    batch_size      = 256,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.05,     # 탐험 촉진 (0.01 → 0.05)
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    verbose         = 1,
)

TOTAL_TIMESTEPS  = 2_000_000
N_ENVS           = 4           # 병렬 환경 수
EVAL_FREQ        = 50_000      # 평가 주기 (timesteps)
EVAL_EPISODES    = 5
WINDOW_SIZE      = 2000        # 에피소드 길이


def make_env(df: pd.DataFrame, rank: int, seed: int = 0):
    """벡터화 환경 생성 팩토리."""
    def _init():
        env = CryptoTradingEnv(df, window_size=WINDOW_SIZE)
        env = ActionMasker(env, _mask_fn)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ── 콜백 ──────────────────────────────────────────────────────────────────────

class TradingMetricsCallback(BaseCallback):
    """에피소드 종료마다 거래 지표(수익률, 거래 수)를 로그에 기록."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self._episode_rewards.append(ep["r"])
                if len(self._episode_rewards) % 50 == 0:
                    avg = np.mean(self._episode_rewards[-50:])
                    self.logger.record("trade/avg_ep_reward_50", avg)
        return True


# ── 학습 ──────────────────────────────────────────────────────────────────────

def train_rl_agent(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    total_timesteps: int = TOTAL_TIMESTEPS,
    n_envs: int          = N_ENVS,
    resume_path: str | None = None,
) -> MaskablePPO:
    """
    MaskablePPO 에이전트를 학습하여 반환.

    Parameters
    ----------
    train_df    : signal_long/short/context + close 포함 DataFrame
    val_df      : 평가용 DataFrame
    resume_path : 기존 체크포인트 경로 (이어서 학습 시)
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RL] 학습 시작 | device={device} | n_envs={n_envs} | "
          f"total_timesteps={total_timesteps:,}")

    # ── 학습 환경 (n_envs>1: 병렬, n_envs=1: 단일) ────────────────────────────
    env_fns = [make_env(train_df, rank=i, seed=42) for i in range(n_envs)]
    if n_envs > 1:
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)
    train_env = VecMonitor(train_env)

    # ── 평가 환경 (단일) ───────────────────────────────────────────────────────
    eval_env_raw = CryptoTradingEnv(val_df, window_size=len(val_df))
    eval_env     = ActionMasker(eval_env_raw, _mask_fn)
    eval_env     = Monitor(eval_env)

    # ── 콜백 ──────────────────────────────────────────────────────────────────
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, min_evals=20, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_SAVE_DIR, "rl_best"),
        log_path=LOG_DIR,
        eval_freq=max(EVAL_FREQ // n_envs, 1),
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        callback_after_eval=stop_callback,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=os.path.join(MODEL_SAVE_DIR, "rl_checkpoints"),
        name_prefix="maskable_ppo",
    )
    metrics_callback = TradingMetricsCallback()

    callbacks = [eval_callback, checkpoint_callback, metrics_callback]

    # ── 모델 생성 / 로드 ───────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=torch.nn.GELU,
    )

    if resume_path and os.path.exists(resume_path):
        print(f"[RL] 체크포인트 로드: {resume_path}")
        model = MaskablePPO.load(
            resume_path,
            env=train_env,
            device=device,
            custom_objects={"learning_rate": RL_CONFIG["learning_rate"]},
        )
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            train_env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=LOG_DIR,
            device=device,
            **RL_CONFIG,
        )

    # ── 학습 ──────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=(resume_path is None),
        progress_bar=False,
    )

    # 최종 저장
    final_path = os.path.join(MODEL_SAVE_DIR, "rl_final")
    model.save(final_path)
    print(f"[RL] 최종 모델 저장: {final_path}.zip")

    train_env.close()
    eval_env.close()
    return model


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

def load_rl_agent(
    path: str = "models/rl_best/best_model",
    env: CryptoTradingEnv | None = None,
) -> MaskablePPO:
    """저장된 MaskablePPO 에이전트 로드."""
    model = MaskablePPO.load(path, env=env)
    print(f"[RL] 에이전트 로드 완료: {path}")
    return model


# ── 단일 에피소드 롤아웃 ──────────────────────────────────────────────────────

def rollout_episode(
    model: MaskablePPO,
    df: pd.DataFrame,
    deterministic: bool = True,
) -> dict:
    """
    학습된 에이전트로 단일 에피소드를 실행하고 결과를 반환.

    Returns
    -------
    dict: capital_history, trade_log, total_return, n_trades
    """
    env = CryptoTradingEnv(df, window_size=len(df))
    env = ActionMasker(env, _mask_fn)

    obs, _     = env.reset()
    done       = False
    capital_history = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic,
                                  action_masks=env.action_masks())
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        capital_history.append(info["capital"])

    final_capital = capital_history[-1] if capital_history else env.unwrapped.initial_capital
    total_return  = (final_capital - env.unwrapped.initial_capital) / env.unwrapped.initial_capital

    return {
        "capital_history": capital_history,
        "trade_log":       env.unwrapped.trade_log,
        "total_return":    total_return,
        "n_trades":        info["total_trades"],
        "final_capital":   final_capital,
    }


# ── 실행 예시 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_signals.parquet"
    val_path   = sys.argv[2] if len(sys.argv) > 2 else "data/val_signals.parquet"

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)

    print(f"[RL] 데이터 로드 | train={len(train_df)} | val={len(val_df)}")

    model = train_rl_agent(train_df, val_df)

    # 검증 롤아웃
    result = rollout_episode(model, val_df)
    print(f"\n[RL] 검증 결과:")
    print(f"  총 수익률: {result['total_return']*100:.2f}%")
    print(f"  최종 자본: {result['final_capital']:.2f}")
    print(f"  총 거래 수: {result['n_trades']}")
