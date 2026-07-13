from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from common.run_manager import RunPaths
from common.task_identity import canonical_task_id_from_config


class SlideDiagnosticsCallback(BaseCallback):
    """Aggregate slide-task diagnostics and publish scalar TensorBoard data."""

    def __init__(
        self,
        log_interval_steps: int,
        *,
        command_debug: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.log_interval_steps = max(int(log_interval_steps), 1)
        self.command_debug = bool(command_debug)
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
            vx_error = info.get("velocity_error")
            yaw_error = info.get("yaw_rate_error")
            self._append("tracking/vx_error", vx_error)
            self._append("tracking/vx_abs_error", None if vx_error is None else abs(float(vx_error)))
            self._append("tracking/yaw_rate_error", yaw_error)
            self._append(
                "tracking/yaw_rate_abs_error",
                None if yaw_error is None else abs(float(yaw_error)),
            )

            self._append("stability/base_height", info.get("base_height"))
            self._append("stability/pitch", info.get("pitch"))
            self._append("stability/roll", info.get("roll"))
            self._append("stability/pitch_rate", info.get("pitch_rate"))
            self._append("stability/roll_rate", info.get("roll_rate"))
            self._append("stability/wheel_longitudinal_offset", info.get("wheel_longitudinal_offset"))

            self._append("smoothness/forward_delta_vx", info.get("forward_delta_vx"))
            self._append("smoothness/wheel_action_rate", info.get("wheel_action_rate"))
            self._append("smoothness/action_rate", info.get("action_rate"))

            self._append("effort/mean_leg_torque", np.abs(info.get("tau_leg", [])))
            self._append("effort/mean_wheel_torque", np.abs(info.get("tau_wheel", [])))

            terminated = bool(info.get("terminated", False))
            truncated = bool(info.get("truncated", False))
            self._append("episode/fall_rate", float(terminated))
            self._append("episode/timeout_rate", float(truncated))
            if terminated or truncated:
                self._append("episode/mean_length", info.get("episode_step"))

            reward_terms = info.get("reward_terms", {})
            if isinstance(reward_terms, dict):
                self._append("effort/torque_penalty", reward_terms.get("torque_penalty"))

            if self.command_debug:
                self._append("command/current_vx_cmd", info.get("command_forward_velocity"))
                self._append("command/current_yaw_rate_cmd", info.get("command_yaw_rate"))
                self._append("command/target_vx_cmd", info.get("target_command_forward_velocity"))
                self._append("command/target_yaw_rate_cmd", info.get("target_command_yaw_rate"))
                self._append(
                    "command/time_to_next_resample",
                    info.get("time_to_next_command_resample"),
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
    run_paths: RunPaths,
    n_envs: int = 1,
) -> CallbackList:
    """Build callbacks whose artifacts stay inside one run directory."""
    logging_cfg = cfg.get("logging", {})
    callback_cfg = cfg.get("callbacks", {})
    task_id = canonical_task_id_from_config(cfg)

    divisor = max(int(n_envs), 1)
    checkpoint_freq = max(int(callback_cfg.get("checkpoint_freq", 50000)) // divisor, 1)
    eval_freq = max(int(callback_cfg.get("eval_freq", 50000)) // divisor, 1)

    callbacks: list[BaseCallback] = [
        SlideDiagnosticsCallback(
            int(logging_cfg.get("log_interval_steps", 1000)),
            command_debug=bool(logging_cfg.get("command_debug", False)),
        ),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(run_paths.checkpoints),
            name_prefix=task_id,
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_paths.models),
            log_path=str(run_paths.eval),
            eval_freq=eval_freq,
            n_eval_episodes=int(callback_cfg.get("n_eval_episodes", 5)),
            deterministic=bool(callback_cfg.get("deterministic_eval", True)),
            render=False,
        ),
    ]
    return CallbackList(callbacks)
