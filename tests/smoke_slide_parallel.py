from __future__ import annotations

import copy
import multiprocessing
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3.common.env_checker import check_env
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from callbacks.slide_callbacks import build_slide_callbacks
from common.run_manager import create_run
from common.vec_env import create_evaluation_vec_env, create_training_vec_env
from envs.slide_flat_factory import create_slide_env, load_slide_config
from rl.ppo import create_ppo_model, load_ppo_model


EXPECTED_DIAGNOSTICS = {
    "diagnostics/wheel_longitudinal_offset",
    "diagnostics/wheel_longitudinal_offset_abs_m",
    "diagnostics/wheel_longitudinal_offset_excess_m",
    "diagnostics/penalty_wheel_longitudinal_offset",
    "diagnostics/straight_stance_gate",
}


def run_random_rollout(cfg: dict, steps: int = 1000) -> None:
    env = create_slide_env(cfg)
    try:
        seed = int(cfg.get("seed", 1))
        obs, _ = env.reset(seed=seed)
        assert obs.shape == (28,)
        assert np.isfinite(obs).all()
        check_env(env, warn=True, skip_render_check=True)

        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        for _ in range(int(steps)):
            action = rng.uniform(-1.0, 1.0, size=(6,)).astype(np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            assert np.isfinite(env.data.qpos).all()
            assert np.isfinite(env.data.qvel).all()
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()


def run_parallel_training_smoke(cfg: dict, output_root: Path) -> None:
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["output"] = {"root_dir": str(output_root), "run_id": "parallel_smoke"}
    smoke_cfg["experiment"]["name"] = "parallel_smoke"
    smoke_cfg["training"].update({"total_timesteps": 1024, "n_envs": 8})
    smoke_cfg["ppo"].update(
        {
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 1,
            "verbose": 0,
            "policy_kwargs": {"net_arch": [32, 32]},
        }
    )
    smoke_cfg["logging"]["log_interval_steps"] = 64
    smoke_cfg["callbacks"].update(
        {
            "checkpoint_freq": 1024,
            "eval_freq": 1024,
            "n_eval_episodes": 1,
            "deterministic_eval": True,
        }
    )

    run_paths = create_run(smoke_cfg, run_id="parallel_smoke")
    seed = int(smoke_cfg["seed"])

    single_cfg = copy.deepcopy(smoke_cfg)
    single_cfg["training"]["n_envs"] = 1
    single_cfg["ppo"]["n_steps"] = 2048
    single_env = create_training_vec_env(run_paths.config, seed=seed, n_envs=1)
    try:
        single_model = create_ppo_model(single_env, single_cfg)
        single_checkpoint = run_paths.models / "single_env_checkpoint"
        single_model.save(str(single_checkpoint))
    finally:
        single_env.close()

    train_env = create_training_vec_env(
        run_paths.config,
        seed=seed,
        n_envs=8,
        start_method="spawn",
        monitor_path=run_paths.tensorboard / "monitor.csv",
    )
    eval_env = create_evaluation_vec_env(run_paths.config, seed=seed + 10000)
    try:
        assert train_env.num_envs == 8
        assert train_env.get_attr("render_mode") == [None] * 8
        initial_obs = train_env.reset()
        assert initial_obs.shape == (8, 28)
        assert np.isfinite(initial_obs).all()
        assert len({row.tobytes() for row in initial_obs}) == 8
        model = load_ppo_model(
            single_checkpoint.with_suffix(".zip"),
            env=train_env,
            cfg=smoke_cfg,
            tensorboard_log=run_paths.tensorboard,
            override_ppo_config=True,
        )
        assert model.n_envs == 8
        assert model.n_steps == 128
        assert model.batch_size == 64
        callbacks = build_slide_callbacks(
            smoke_cfg,
            eval_env=eval_env,
            run_paths=run_paths,
            n_envs=8,
        )
        model.learn(total_timesteps=1024, callback=callbacks, tb_log_name="parallel_smoke")
        model.save(str(run_paths.models / "final_model"))
    finally:
        train_env.close()
        eval_env.close()

    assert (run_paths.models / "final_model.zip").is_file()
    assert (run_paths.models / "best_model.zip").is_file()
    assert (run_paths.eval / "evaluations.npz").is_file()
    assert list(run_paths.checkpoints.glob("*.zip"))

    event_files = list(run_paths.tensorboard.rglob("events.out.tfevents.*"))
    assert event_files
    accumulator = EventAccumulator(str(event_files[0].parent))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    missing = EXPECTED_DIAGNOSTICS - scalar_tags
    assert not missing, f"Missing TensorBoard diagnostics: {sorted(missing)}"


def main() -> None:
    multiprocessing.freeze_support()
    cfg = load_slide_config(REPO_ROOT / "configs" / "slide_flat_v2.yaml")
    run_random_rollout(cfg)
    print("v2 check_env and 1000-step random rollout passed")

    with tempfile.TemporaryDirectory(prefix="slide_parallel_smoke_") as temp_dir:
        run_parallel_training_smoke(cfg, Path(temp_dir))
    print("8-worker spawn PPO, checkpoint, eval, and TensorBoard smoke passed")


if __name__ == "__main__":
    main()
