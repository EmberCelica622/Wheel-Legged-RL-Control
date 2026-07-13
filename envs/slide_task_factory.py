from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

import gymnasium as gym
import numpy as np

from common.task_identity import canonical_task_id_from_config, normalize_task_config
from envs.slide_fixed_velocity_flat_v1 import (
    SlideFixedVelocityFlatV1Env,
    load_slide_config as _load_slide_config,
)
from envs.slide_fixed_velocity_flat_v1_legacy import (
    SlideFixedVelocityFlatV1LegacyEnv,
)
from envs.slide_flat_v3 import SlideFlatV3Env
from envs.slide_variable_velocity_flat_v2 import SlideVariableVelocityFlatV2Env


_ENV_CLASSES: dict[str, Type[gym.Env]] = {
    "slide_fixed_velocity_flat_v1_legacy": SlideFixedVelocityFlatV1LegacyEnv,
    "slide_fixed_velocity_flat_v1": SlideFixedVelocityFlatV1Env,
    "slide_variable_velocity_flat_v2": SlideVariableVelocityFlatV2Env,
    "slide_dynamic_command_flat_v3": SlideFlatV3Env,
}


def normalize_slide_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a config whose task block is the only task identity source."""
    return normalize_task_config(cfg)


def load_slide_config(config_path: str | Path) -> dict[str, Any]:
    """Load and normalize a slide task config at the compatibility boundary."""
    return normalize_slide_config(_load_slide_config(config_path))


def slide_task_id(cfg: dict[str, Any]) -> str:
    """Return the validated canonical slide task id."""
    task_id = canonical_task_id_from_config(normalize_slide_config(cfg))
    if task_id not in _ENV_CLASSES:
        choices = ", ".join(sorted(_ENV_CLASSES))
        raise ValueError(f"Unsupported slide task {task_id!r}; expected one of: {choices}")
    return task_id


def create_slide_env(
    cfg: dict[str, Any],
    *,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a fresh slide environment using a strict canonical task whitelist."""
    normalized = normalize_slide_config(cfg)
    env_class = _ENV_CLASSES[slide_task_id(normalized)]
    return env_class(normalized, render_mode=render_mode)


class FixedCommandResetWrapper(gym.Wrapper):
    """Apply the same command override to every automatic episode reset."""

    def __init__(self, env: gym.Env, command: list[float], *, fixed_command: bool = True):
        super().__init__(env)
        self.command = list(command)
        self.fixed_command = bool(fixed_command)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        merged_options = dict(options or {})
        merged_options["command"] = self.command
        merged_options["fixed_command"] = self.fixed_command
        return self.env.reset(seed=seed, options=merged_options)


@dataclass(frozen=True)
class SlideEnvFactory:
    """Spawn-safe callable that constructs MuJoCo state inside one worker."""

    config_path: str
    seed: int
    rank: int

    def __call__(self) -> gym.Env:
        worker_seed = int(self.seed) + int(self.rank)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

        cfg = load_slide_config(Path(self.config_path))
        cfg["seed"] = worker_seed
        # Training workers must never create a Viewer or another render target.
        return create_slide_env(cfg, render_mode=None)
