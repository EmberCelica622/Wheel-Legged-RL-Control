from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.slide_task_factory import create_slide_env, load_slide_config


def test_v1_command_is_fixed() -> None:
    cfg = load_slide_config(REPO_ROOT / "configs" / "slide_fixed_velocity_flat_v1.yaml")
    env = create_slide_env(cfg)
    try:
        for seed in range(8):
            obs, info = env.reset(seed=seed)
            assert np.array_equal(env.command, [0.8, 0.0])
            assert np.array_equal(obs[9:11], env.command.astype(np.float32))
            assert info["command_forward_velocity"] == 0.8
            assert info["command_yaw_rate"] == 0.0
    finally:
        env.close()


def test_v2_command_range_episode_stability_and_override() -> None:
    cfg = load_slide_config(REPO_ROOT / "configs" / "slide_variable_velocity_flat_v2.yaml")
    env = create_slide_env(cfg)
    try:
        for seed in range(100):
            obs, _ = env.reset(seed=seed)
            command = env.command.copy()
            assert 0.0 <= command[0] <= 2.0
            assert command[1] == 0.0
            assert np.array_equal(obs[9:11], command.astype(np.float32))
            for _ in range(10):
                env.step(np.zeros(6, dtype=np.float32))
                assert np.array_equal(env.command, command)

        obs, _ = env.reset(seed=9, options={"command": [0.8, 0.0]})
        assert np.array_equal(env.command, [0.8, 0.0])
        assert np.array_equal(obs[9:11], np.array([0.8, 0.0], dtype=np.float32))
    finally:
        env.close()


def _fixed_action_rollout(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = load_slide_config(REPO_ROOT / "configs" / "slide_variable_velocity_flat_v2.yaml")
    env = create_slide_env(cfg)
    try:
        obs, _ = env.reset(seed=seed)
        command = env.command.copy()
        trajectory = []
        action = np.zeros(6, dtype=np.float32)
        for _ in range(50):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            trajectory.append(np.concatenate([next_obs, [reward]]))
            if terminated or truncated:
                break
        return command, obs, np.asarray(trajectory)
    finally:
        env.close()


def test_v2_same_seed_is_reproducible() -> None:
    first = _fixed_action_rollout(1234)
    second = _fixed_action_rollout(1234)
    for first_value, second_value in zip(first, second):
        assert np.array_equal(first_value, second_value)
