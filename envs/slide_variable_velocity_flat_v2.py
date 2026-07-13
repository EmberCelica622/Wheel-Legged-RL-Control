from __future__ import annotations

from typing import Any

import numpy as np

from envs.slide_fixed_velocity_flat_v1 import SlideFixedVelocityFlatV1Env


class SlideVariableVelocityFlatV2Env(SlideFixedVelocityFlatV1Env):
    """Flat slide task with one independently sampled vx command per episode."""

    def __init__(self, cfg: dict[str, Any], render_mode: str | None = None):
        super().__init__(cfg, render_mode=render_mode)
        self._configure_command_sampler()

    def _configure_command_sampler(self) -> None:
        command_cfg = self.command_cfg
        forward_cfg = command_cfg.get("forward_velocity", {})
        yaw_cfg = command_cfg.get("yaw_rate", {})
        if not isinstance(forward_cfg, dict):
            raise ValueError("v2 command.forward_velocity must use the versioned mapping schema")
        if str(forward_cfg.get("mode", "")).lower() != "uniform_per_episode":
            raise ValueError("v2 command.forward_velocity.mode must be uniform_per_episode")

        forward_range = np.asarray(forward_cfg.get("range", []), dtype=np.float64)
        if forward_range.shape != (2,) or not np.isfinite(forward_range).all():
            raise ValueError("v2 command.forward_velocity.range must contain two finite values")
        # if not np.allclose(forward_range, [0.0, 2.0], atol=1e-12):
        #     raise ValueError("slide-flat v2 requires forward velocity range [0.0, 2.0]")
        self.forward_command_range = forward_range

        if not isinstance(yaw_cfg, dict):
            raise ValueError("v2 command.yaw_rate must use the versioned mapping schema")
        if str(yaw_cfg.get("mode", "")).lower() != "fixed":
            raise ValueError("v2 command.yaw_rate.mode must be fixed")
        if not np.isclose(float(yaw_cfg.get("value", 0.0)), 0.0, atol=1e-12):
            raise ValueError("slide-flat v2 requires command.yaw_rate.value to be 0.0")

    def _reset_command(self, options: dict[str, Any] | None) -> None:
        override = self._command_override(options)
        if override is not None:
            lower, upper = self.forward_command_range
            if not lower <= override[0] <= upper:
                raise ValueError(
                    f"v2 reset command vx must be in [{lower}, {upper}], got {override[0]}"
                )
            self.command[:] = override
            return

        self.command[0] = self.np_random.uniform(
            self.forward_command_range[0],
            self.forward_command_range[1],
        )
        self.command[1] = 0.0


SlideFlatV2Env = SlideVariableVelocityFlatV2Env
SlideFlatEnv = SlideVariableVelocityFlatV2Env
