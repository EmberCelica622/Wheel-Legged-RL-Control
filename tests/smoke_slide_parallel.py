from __future__ import annotations

import argparse
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
from envs.slide_task_factory import create_slide_env, load_slide_config, slide_task_id
from rl.ppo import create_ppo_model, warm_start_ppo_policy


EXPECTED_DIAGNOSTICS = {
    "tracking/vx_error",
    "tracking/vx_abs_error",
    "tracking/yaw_rate_error",
    "tracking/yaw_rate_abs_error",
    "stability/base_height",
    "stability/pitch",
    "stability/roll",
    "stability/pitch_rate",
    "stability/roll_rate",
    "stability/wheel_longitudinal_offset",
    "smoothness/forward_delta_vx",
    "smoothness/wheel_action_rate",
    "smoothness/action_rate",
    "effort/mean_leg_torque",
    "effort/mean_wheel_torque",
    "effort/torque_penalty",
    "episode/fall_rate",
    "episode/timeout_rate",
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


def run_parallel_training_smoke(
    cfg: dict,
    output_root: Path,
    *,
    n_envs: int,
    timesteps: int,
) -> None:
    if timesteps < 64 or timesteps % n_envs != 0:
        raise ValueError("Smoke timesteps must be at least 64 and divisible by n_envs")
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["output"] = {"root_dir": str(output_root), "run_id": "parallel_smoke"}
    smoke_cfg["experiment"]["name"] = "parallel_smoke"
    smoke_cfg["training"].update(
        {
            "total_timesteps": timesteps,
            "n_envs": n_envs,
            "rollout_batch_size": timesteps,
        }
    )
    smoke_cfg["ppo"].update(
        {
            "batch_size": 64,
            "n_epochs": 1,
            "verbose": 0,
            "policy_kwargs": {"net_arch": [32, 32]},
        }
    )
    smoke_cfg["logging"]["log_interval_steps"] = 64
    smoke_cfg["callbacks"].update(
        {
            "checkpoint_freq": timesteps,
            "eval_freq": timesteps,
            "n_eval_episodes": 1,
            "deterministic_eval": True,
        }
    )

    run_paths = create_run(smoke_cfg, run_id="parallel_smoke")
    seed = int(smoke_cfg["seed"])

    # Build a genuine v1 checkpoint, then policy-only warm-start the selected
    # target task. With the default v2 target this exercises the v1 -> v2 path.
    single_cfg = load_slide_config(REPO_ROOT / "configs" / "slide_fixed_velocity_flat_v1.yaml")
    single_cfg["output"] = {"root_dir": str(output_root), "run_id": "v1_source"}
    single_cfg["experiment"]["name"] = "warm_start_source"
    single_cfg["training"].update({"n_envs": 1, "rollout_batch_size": 2048})
    single_cfg["ppo"].update(copy.deepcopy(smoke_cfg["ppo"]))
    source_paths = create_run(single_cfg, run_id="v1_source")
    single_env = create_training_vec_env(source_paths.config, seed=seed, n_envs=1)
    try:
        single_model = create_ppo_model(single_env, single_cfg)
        single_checkpoint = run_paths.models / "single_env_checkpoint"
        single_model.save(str(single_checkpoint))
    finally:
        single_env.close()

    train_env = create_training_vec_env(
        run_paths.config,
        seed=seed,
        n_envs=n_envs,
        start_method="spawn",
        monitor_path=run_paths.tensorboard / "monitor.csv",
    )
    eval_env = create_evaluation_vec_env(run_paths.config, seed=seed + 10000)
    try:
        assert train_env.num_envs == n_envs
        assert train_env.get_attr("render_mode") == [None] * n_envs
        initial_obs = train_env.reset()
        assert initial_obs.shape == (n_envs, 28)
        assert np.isfinite(initial_obs).all()
        assert len({row.tobytes() for row in initial_obs}) == n_envs
        dynamic_task_ids = {
            "slide_variable_velocity_flat_v2",
            "slide_dynamic_command_flat_v3",
        }
        if slide_task_id(smoke_cfg) in dynamic_task_ids:
            assert len({float(value) for value in initial_obs[:, 9]}) == n_envs
        else:
            assert np.allclose(initial_obs[:, 9:11], [0.8, 0.0])
        model = create_ppo_model(train_env, smoke_cfg, tensorboard_log=run_paths.tensorboard)
        source = warm_start_ppo_policy(model, single_checkpoint.with_suffix(".zip"), device="cpu")
        for name, value in model.policy.state_dict().items():
            assert np.array_equal(value.detach().cpu().numpy(), source.policy.state_dict()[name].detach().cpu().numpy())
        assert not model.policy.optimizer.state
        assert model.n_envs == n_envs
        assert model.n_steps == timesteps // n_envs
        assert model.batch_size == 64
        callbacks = build_slide_callbacks(
            smoke_cfg,
            eval_env=eval_env,
            run_paths=run_paths,
            n_envs=n_envs,
        )
        model.learn(total_timesteps=timesteps, callback=callbacks, tb_log_name="parallel_smoke")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless slide task vector smoke test.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "slide_variable_velocity_flat_v2.yaml"),
    )
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--random-steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    multiprocessing.freeze_support()
    args = parse_args()
    cfg = load_slide_config(args.config)
    run_random_rollout(cfg, steps=args.random_steps)
    print(f"check_env and {args.random_steps}-step random rollout passed")

    with tempfile.TemporaryDirectory(prefix="slide_parallel_smoke_") as temp_dir:
        run_parallel_training_smoke(
            cfg,
            Path(temp_dir),
            n_envs=args.n_envs,
            timesteps=args.timesteps,
        )
    print(f"{args.n_envs}-worker spawn PPO, warm-start, checkpoint, eval, and TensorBoard smoke passed")


if __name__ == "__main__":
    main()
