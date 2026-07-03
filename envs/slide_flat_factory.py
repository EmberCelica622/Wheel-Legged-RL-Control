from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

import gymnasium as gym
import numpy as np

from envs.slide_flat_v1 import SlideFlatEnv as SlideFlatV1Env
from envs.slide_flat_v1 import load_slide_config
from envs.slide_flat_v2 import SlideFlatEnv as SlideFlatV2Env


_ENV_CLASSES: dict[str, Type[gym.Env]] = {
    "v1": SlideFlatV1Env,
    "v2": SlideFlatV2Env,
}


def slide_env_variant(cfg: dict[str, Any]) -> str:
    """Return the validated slide environment variant from a config."""
    variant = str(cfg.get("experiment", {}).get("env_variant", "v1")).lower()
    if variant not in _ENV_CLASSES:
        choices = ", ".join(sorted(_ENV_CLASSES))
        raise ValueError(f"Unsupported experiment.env_variant {variant!r}; expected one of: {choices}")
    return variant


def create_slide_env(
    cfg: dict[str, Any],
    *,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a fresh v1 or v2 environment using a strict variant whitelist."""
    env_class = _ENV_CLASSES[slide_env_variant(cfg)]
    return env_class(cfg, render_mode=render_mode)


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

