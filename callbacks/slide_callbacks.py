from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)


class SlideDiagnosticsCallback(BaseCallback):
    """Aggregate slide-task diagnostics and publish scalar TensorBoard data."""

    def __init__(self, log_interval_steps: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_interval_steps = max(int(log_interval_steps), 1)
        self._last_log_step = 0
        self._samples: dict[str, list[float]] = defaultdict(list)

    def _append(self, name: str, value: Any) -> None:
        scalar = np.asarray(value, dtype=np.float64)
        if scalar.size == 0:
            return
        value_float = float(np.mean(scalar))
        if np.isfinite(value_float):
            self._samples[name].append(value_float)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            self._append("diagnostics/base_forward_velocity", info.get("base_forward_velocity"))
            self._append("diagnostics/velocity_error", info.get("velocity_error"))
            self._append("diagnostics/base_height", info.get("base_height"))
            self._append("diagnostics/roll", info.get("roll"))
            self._append("diagnostics/pitch", info.get("pitch"))
            self._append("diagnostics/mean_abs_leg_torque", np.abs(info.get("tau_leg", [])))
            self._append("diagnostics/mean_abs_wheel_torque", np.abs(info.get("tau_wheel", [])))
            self._append(
                "diagnostics/mean_leg_joint_velocity",
                info.get("mean_leg_joint_velocity"),
            )
            self._append("diagnostics/terminated_rate", float(info.get("terminated", False)))

            reward_terms = info.get("reward_terms", {})
            if isinstance(reward_terms, dict):
                self._append(
                    "diagnostics/reward_forward_velocity_tracking",
                    reward_terms.get("forward_velocity_tracking"),
                )
                self._append("diagnostics/reward_upright", reward_terms.get("upright"))
                self._append(
                    "diagnostics/penalty_action_rate",
                    reward_terms.get("action_rate_penalty"),
                )
                self._append(
                    "diagnostics/penalty_torque",
                    reward_terms.get("torque_penalty"),
                )

        if self.num_timesteps - self._last_log_step >= self.log_interval_steps:
            for name, values in self._samples.items():
                if values:
                    # record_mean preserves all windows until SB3 dumps its
                    # logger, avoiding per-step scalar overwrites.
                    self.logger.record_mean(name, float(np.mean(values)))
            self._samples.clear()
            self._last_log_step = self.num_timesteps

        return True


def build_slide_callbacks(
    cfg: dict[str, Any],
    eval_env: Any,
    n_envs: int = 1,
) -> CallbackList:
    """Build diagnostics, checkpoint, and evaluation callbacks from YAML."""
    base_dir = Path(cfg.get("_base_dir", ".")).expanduser().resolve()

    def resolve(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else base_dir / path

    logging_cfg = cfg.get("logging", {})
    callback_cfg = cfg.get("callbacks", {})
    checkpoint_dir = resolve(logging_cfg["checkpoint_dir"])
    eval_log_dir = resolve(logging_cfg["eval_log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    divisor = max(int(n_envs), 1)
    checkpoint_freq = max(int(callback_cfg.get("checkpoint_freq", 50000)) // divisor, 1)
    eval_freq = max(int(callback_cfg.get("eval_freq", 50000)) // divisor, 1)

    callbacks: list[BaseCallback] = [
        SlideDiagnosticsCallback(int(logging_cfg.get("log_interval_steps", 1000))),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix=str(logging_cfg.get("run_name", "slide_flat")),
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(eval_log_dir),
            log_path=str(eval_log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=int(callback_cfg.get("n_eval_episodes", 5)),
            deterministic=bool(callback_cfg.get("deterministic_eval", True)),
            render=False,
        ),
    ]
    return CallbackList(callbacks)
