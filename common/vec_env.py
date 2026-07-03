from __future__ import annotations

import os
from pathlib import Path

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
)

from envs.slide_flat_factory import SlideEnvFactory


def _validate_parallel_config(n_envs: int, start_method: str) -> None:
    if n_envs < 1:
        raise ValueError("training.n_envs must be at least 1")
    if os.name == "nt" and start_method != "spawn":
        raise ValueError("Windows slide training requires training.start_method: spawn")


def create_training_vec_env(
    config_path: str | Path,
    *,
    seed: int,
    n_envs: int,
    start_method: str = "spawn",
    monitor_path: str | Path | None = None,
) -> VecEnv:
    """Create seeded training environments without sharing MuJoCo objects."""
    n_envs = int(n_envs)
    start_method = str(start_method).lower()
    _validate_parallel_config(n_envs, start_method)

    config_path = str(Path(config_path).expanduser().resolve())
    factories = [SlideEnvFactory(config_path, int(seed), rank) for rank in range(n_envs)]
    if n_envs == 1:
        base_env: VecEnv = DummyVecEnv(factories)
    else:
        base_env = SubprocVecEnv(factories, start_method=start_method)

    filename = None if monitor_path is None else str(Path(monitor_path).expanduser().resolve())
    env = VecMonitor(base_env, filename=filename)
    # BaseVecEnv assigns seed + rank on the next reset.
    env.seed(int(seed))
    return env


def create_evaluation_vec_env(
    config_path: str | Path,
    *,
    seed: int,
) -> VecEnv:
    """Create an independently seeded, single-environment evaluation VecEnv."""
    config_path = str(Path(config_path).expanduser().resolve())
    base_env = DummyVecEnv([SlideEnvFactory(config_path, int(seed), 0)])
    env = VecMonitor(base_env)
    env.seed(int(seed))
    return env
