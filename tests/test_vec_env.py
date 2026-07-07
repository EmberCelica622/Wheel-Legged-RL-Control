from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.vec_env import create_training_vec_env
from envs.slide_task_factory import load_slide_config
from rl.ppo import resolve_ppo_rollout_config


def test_spawn_vec_env_has_independent_finite_workers() -> None:
    config_path = REPO_ROOT / "configs" / "slide_variable_velocity_flat_v2.yaml"
    env = create_training_vec_env(
        config_path,
        seed=123,
        n_envs=2,
        start_method="spawn",
    )
    try:
        obs = env.reset()
        assert obs.shape == (2, 28)
        assert np.isfinite(obs).all()
        assert not np.array_equal(obs[0], obs[1])
        assert env.get_attr("render_mode") == [None, None]

        actions = np.zeros((2, 6), dtype=np.float32)
        for _ in range(32):
            obs, rewards, dones, infos = env.step(actions)
            assert np.isfinite(obs).all()
            assert np.isfinite(rewards).all()
            assert dones.shape == (2,)
            assert len(infos) == 2
    finally:
        env.close()


def test_parallel_ppo_rollout_sizes_match() -> None:
    cfg = load_slide_config(REPO_ROOT / "configs" / "slide_variable_velocity_flat_v2.yaml")
    for n_envs, expected_n_steps in ((1, 2048), (4, 512), (8, 256)):
        resolved = resolve_ppo_rollout_config(cfg, n_envs)
        assert resolved["n_steps"] == expected_n_steps
        assert resolved["rollout_batch_size"] == 2048
