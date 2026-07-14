from __future__ import annotations

from typing import Any

import numpy as np

from envs.slide_variable_velocity_flat_v2 import SlideVariableVelocityFlatV2Env


def _range_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain two finite values")
    if arr[0] > arr[1]:
        raise ValueError(f"{name} lower bound must be <= upper bound")
    return arr


def _non_negative_float(value: Any, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return scalar


class SlideFlatV3Env(SlideVariableVelocityFlatV2Env):
    """Slide-flat task with smooth intra-episode vx and yaw-rate commands."""

    def _configure_command_sampler(self) -> None:
        command_cfg = self.command_cfg
        if str(command_cfg.get("mode", "")).lower() != "dynamic_per_episode":
            raise ValueError("v3 command.mode must be dynamic_per_episode")

        forward_cfg = command_cfg.get("forward_velocity", {})
        yaw_cfg = command_cfg.get("yaw_rate", {})
        interval_cfg = command_cfg.get("resample_interval_s", {})
        initial_cfg = command_cfg.get("initial_command", {})
        if not isinstance(forward_cfg, dict):
            raise ValueError("v3 command.forward_velocity must use a mapping schema")
        if not isinstance(yaw_cfg, dict):
            raise ValueError("v3 command.yaw_rate must use a mapping schema")
        if not isinstance(interval_cfg, dict):
            raise ValueError("v3 command.resample_interval_s must use a mapping schema")
        if not isinstance(initial_cfg, dict):
            raise ValueError("v3 command.initial_command must use a mapping schema")

        self.forward_command_range = _range_array(
            forward_cfg.get("range", []),
            "v3 command.forward_velocity.range",
        )
        self.yaw_command_range = _range_array(
            yaw_cfg.get("range", []),
            "v3 command.yaw_rate.range",
        )
        self.initial_forward_command_range = _range_array(
            initial_cfg.get("forward_velocity_range", self.forward_command_range),
            "v3 command.initial_command.forward_velocity_range",
        )
        self.initial_yaw_command_range = _range_array(
            initial_cfg.get("yaw_rate_range", self.yaw_command_range),
            "v3 command.initial_command.yaw_rate_range",
        )
        self.resample_interval_range = _range_array(
            interval_cfg.get("range", []),
            "v3 command.resample_interval_s.range",
        )
        if self.resample_interval_range[0] <= 0.0:
            raise ValueError("v3 command.resample_interval_s.range lower bound must be positive")

        self.forward_max_delta_per_resample = _non_negative_float(
            forward_cfg.get("max_delta_per_resample"),
            "v3 command.forward_velocity.max_delta_per_resample",
        )
        self.yaw_max_delta_per_resample = _non_negative_float(
            yaw_cfg.get("max_delta_per_resample"),
            "v3 command.yaw_rate.max_delta_per_resample",
        )
        self.command_max_rates = np.array(
            [
                _non_negative_float(
                    forward_cfg.get("max_rate"),
                    "v3 command.forward_velocity.max_rate",
                ),
                _non_negative_float(yaw_cfg.get("max_rate"), "v3 command.yaw_rate.max_rate"),
            ],
            dtype=np.float64,
        )
        self._command_lower_bounds = np.array(
            [self.forward_command_range[0], self.yaw_command_range[0]],
            dtype=np.float64,
        )
        self._command_upper_bounds = np.array(
            [self.forward_command_range[1], self.yaw_command_range[1]],
            dtype=np.float64,
        )
        self._max_command_delta_per_resample = np.array(
            [self.forward_max_delta_per_resample, self.yaw_max_delta_per_resample],
            dtype=np.float64,
        )

        if not (
            self.forward_command_range[0]
            <= self.initial_forward_command_range[0]
            <= self.initial_forward_command_range[1]
            <= self.forward_command_range[1]
        ):
            raise ValueError("v3 initial forward velocity range must be inside command range")
        if not (
            self.yaw_command_range[0]
            <= self.initial_yaw_command_range[0]
            <= self.initial_yaw_command_range[1]
            <= self.yaw_command_range[1]
        ):
            raise ValueError("v3 initial yaw-rate range must be inside command range")

        self.current_command = self.command.copy()
        self.target_command = self.command.copy()
        self.command_schedule_enabled = True
        self.next_command_resample_time = 0.0

    @property
    def command_lower_bounds(self) -> np.ndarray:
        return self._command_lower_bounds.copy()

    @property
    def command_upper_bounds(self) -> np.ndarray:
        return self._command_upper_bounds.copy()

    def _command_override(self, options: dict[str, Any] | None) -> np.ndarray | None:
        if not options or "command" not in options:
            return None
        command = np.asarray(options["command"], dtype=np.float64)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError("reset options.command must contain two finite values [vx, yaw_rate]")
        if not (
            self.forward_command_range[0] <= command[0] <= self.forward_command_range[1]
            and self.yaw_command_range[0] <= command[1] <= self.yaw_command_range[1]
        ):
            raise ValueError(
                "v3 reset options.command must be inside configured forward/yaw ranges"
            )
        return command

    def _sample_resample_interval(self) -> float:
        return float(
            self.np_random.uniform(
                self.resample_interval_range[0],
                self.resample_interval_range[1],
            )
        )

    def _sample_initial_command(self) -> np.ndarray:
        return np.array(
            [
                self.np_random.uniform(
                    self.initial_forward_command_range[0],
                    self.initial_forward_command_range[1],
                ),
                self.np_random.uniform(
                    self.initial_yaw_command_range[0],
                    self.initial_yaw_command_range[1],
                ),
            ],
            dtype=np.float64,
        )

    def _resample_target_command(self) -> None:
        sampled = np.array(
            [
                self.np_random.uniform(
                    self.forward_command_range[0],
                    self.forward_command_range[1],
                ),
                self.np_random.uniform(
                    self.yaw_command_range[0],
                    self.yaw_command_range[1],
                ),
            ],
            dtype=np.float64,
        )
        delta = np.clip(
            sampled - self.target_command,
            -self._max_command_delta_per_resample,
            self._max_command_delta_per_resample,
        )
        self.target_command = np.clip(
            self.target_command + delta,
            self._command_lower_bounds,
            self._command_upper_bounds,
        )

    def _reset_command(self, options: dict[str, Any] | None) -> None:
        override = self._command_override(options)
        initial_command = self._sample_initial_command() if override is None else override
        self.current_command = initial_command.astype(np.float64, copy=True)
        self.target_command = self.current_command.copy()
        self.command[:] = self.current_command
        self.command_schedule_enabled = not bool(options and options.get("fixed_command", False))
        self.next_command_resample_time = self._sample_resample_interval()

    def _post_step_update_command(self) -> None:
        if not self.command_schedule_enabled:
            self.command[:] = self.current_command
            return

        elapsed_s = float(self.episode_step) * self.control_dt
        while elapsed_s + 1e-12 >= self.next_command_resample_time:
            self._resample_target_command()
            self.next_command_resample_time = elapsed_s + self._sample_resample_interval()

        max_step = self.command_max_rates * self.control_dt
        delta = np.clip(self.target_command - self.current_command, -max_step, max_step)
        self.current_command = np.clip(
            self.current_command + delta,
            self._command_lower_bounds,
            self._command_upper_bounds,
        )
        self.command[:] = self.current_command

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        elapsed_s = float(self.episode_step) * self.control_dt
        info.update(
            {
                "target_command_forward_velocity": float(self.target_command[0]),
                "target_command_yaw_rate": float(self.target_command[1]),
                "time_to_next_command_resample": max(
                    float(self.next_command_resample_time - elapsed_s),
                    0.0,
                ),
            }
        )
        return info


SlideDynamicCommandFlatV3Env = SlideFlatV3Env
SlideFlatEnv = SlideFlatV3Env
