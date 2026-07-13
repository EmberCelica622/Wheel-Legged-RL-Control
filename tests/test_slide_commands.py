from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.slide_task_factory import create_slide_env, load_slide_config


def _fast_v3_config() -> dict:
    cfg = copy.deepcopy(load_slide_config(REPO_ROOT / "configs" / "slide_flat_v3.yaml"))
    cfg["command"]["resample_interval_s"]["range"] = [0.02, 0.02]
    return cfg


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
        lower, upper = cfg["command"]["forward_velocity"]["range"]
        for seed in range(100):
            obs, _ = env.reset(seed=seed)
            command = env.command.copy()
            assert lower <= command[0] <= upper
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


def test_v3_shapes_match_v2() -> None:
    v2_env = create_slide_env(load_slide_config(REPO_ROOT / "configs" / "slide_variable_velocity_flat_v2.yaml"))
    v3_env = create_slide_env(load_slide_config(REPO_ROOT / "configs" / "slide_flat_v3.yaml"))
    try:
        assert v3_env.observation_space.shape == v2_env.observation_space.shape == (28,)
        assert v3_env.action_space.shape == v2_env.action_space.shape == (6,)
    finally:
        v2_env.close()
        v3_env.close()


def test_v3_command_range_rate_limit_and_yaw_signs() -> None:
    cfg = _fast_v3_config()
    env = create_slide_env(cfg)
    try:
        lower = np.array(
            [
                cfg["command"]["forward_velocity"]["range"][0],
                cfg["command"]["yaw_rate"]["range"][0],
            ],
            dtype=np.float64,
        )
        upper = np.array(
            [
                cfg["command"]["forward_velocity"]["range"][1],
                cfg["command"]["yaw_rate"]["range"][1],
            ],
            dtype=np.float64,
        )
        max_step = np.array(
            [
                cfg["command"]["forward_velocity"]["max_rate"],
                cfg["command"]["yaw_rate"]["max_rate"],
            ],
            dtype=np.float64,
        ) * env.control_dt
        yaw_signs = {"zero"}
        action = np.zeros(6, dtype=np.float32)

        for seed in range(80):
            obs, _ = env.reset(seed=seed)
            assert np.allclose(obs[9:11], env.command.astype(np.float32))
            previous_command = env.command.copy()
            assert np.all(previous_command >= lower)
            assert np.all(previous_command <= upper)
            if np.isclose(previous_command[1], 0.0):
                yaw_signs.add("zero")

            for _ in range(40):
                obs, _, terminated, truncated, _ = env.step(action)
                command = env.command.copy()
                assert np.all(command >= lower - 1e-12)
                assert np.all(command <= upper + 1e-12)
                assert np.all(np.abs(command - previous_command) <= max_step + 1e-12)
                assert np.allclose(obs[9:11], command.astype(np.float32))
                if command[1] > 1e-6:
                    yaw_signs.add("positive")
                if command[1] < -1e-6:
                    yaw_signs.add("negative")
                previous_command = command
                if terminated or truncated:
                    break

        assert {"negative", "positive", "zero"} <= yaw_signs
    finally:
        env.close()


def _v3_command_trace(seed: int) -> np.ndarray:
    cfg = _fast_v3_config()
    env = create_slide_env(cfg)
    try:
        env.reset(seed=seed)
        trace = [env.command.copy()]
        action = np.zeros(6, dtype=np.float32)
        for _ in range(80):
            env.step(action)
            trace.append(env.command.copy())
        return np.asarray(trace)
    finally:
        env.close()


def test_v3_same_seed_command_schedule_is_reproducible() -> None:
    assert np.array_equal(_v3_command_trace(2026), _v3_command_trace(2026))
