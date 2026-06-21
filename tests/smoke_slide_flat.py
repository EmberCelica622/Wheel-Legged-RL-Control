from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from callbacks.slide_callbacks import build_slide_callbacks
from envs.slide_flat import SlideFlatEnv, load_slide_config
from rl.ppo import create_ppo_model, load_ppo_model


def resolve_path(cfg: dict, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(cfg.get("_base_dir", REPO_ROOT)).expanduser().resolve() / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for SlideFlatEnv and PPO startup.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "slide_flat.yaml"),
        help="Path to slide flat YAML config.",
    )
    parser.add_argument("--random-steps", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_slide_config(args.config)

    xml_path = resolve_path(cfg, cfg["env"]["xml_path"])
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    print(f"XML loaded: {xml_path}")
    print(f"MuJoCo dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}, nsensordata={model.nsensordata}")

    env = SlideFlatEnv(cfg)
    obs, info = env.reset(seed=int(cfg.get("env", {}).get("seed", 1)))
    assert obs.shape == (28,), obs.shape
    assert env.observation_space.shape == (28,), env.observation_space.shape
    assert np.isfinite(obs).all()
    assert env.action_space.shape == (6,), env.action_space.shape
    print(f"reset() observation shape: {obs.shape}")
    print(f"reset() info keys: {sorted(info.keys())}")

    check_env(env, warn=True, skip_render_check=True)
    print("stable_baselines3.common.env_checker.check_env passed")

    obs, info = env.reset(seed=int(cfg.get("env", {}).get("seed", 1)))
    rng = np.random.default_rng(0)
    for _ in range(args.random_steps):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print(f"random action rollout passed: {args.random_steps} steps")

    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg.setdefault("ppo", {}).update(
        {
            "total_timesteps": int(args.train_steps),
            "n_steps": 128,
            "batch_size": 32,
            "n_epochs": 1,
            "verbose": 0,
            "policy_kwargs": {"net_arch": [32, 32]},
        }
    )
    run_id = f"run_{time.time_ns()}"
    artifact_root = f"tests/_smoke_artifacts/{run_id}"
    smoke_cfg.setdefault("logging", {}).update(
        {
            "tensorboard_log_dir": f"{artifact_root}/tensorboard",
            "checkpoint_dir": f"{artifact_root}/checkpoints",
            "eval_log_dir": f"{artifact_root}/eval",
            "model_dir": f"{artifact_root}/models",
            "run_name": "slide_flat_smoke",
            "log_interval_steps": 64,
            "final_model_name": "smoke_model",
        }
    )
    callback_freq = max(min(int(args.train_steps) // 4, 256), 1)
    smoke_cfg.setdefault("callbacks", {}).update(
        {
            "checkpoint_freq": callback_freq,
            "eval_freq": callback_freq,
            "n_eval_episodes": 2,
            "deterministic_eval": True,
        }
    )

    smoke_log_dir = resolve_path(smoke_cfg, smoke_cfg["logging"]["tensorboard_log_dir"])
    smoke_model_dir = resolve_path(smoke_cfg, smoke_cfg["logging"]["model_dir"])
    checkpoint_dir = resolve_path(smoke_cfg, smoke_cfg["logging"]["checkpoint_dir"])
    eval_log_dir = resolve_path(smoke_cfg, smoke_cfg["logging"]["eval_log_dir"])
    smoke_log_dir.mkdir(parents=True, exist_ok=True)
    smoke_model_dir.mkdir(parents=True, exist_ok=True)

    train_env = Monitor(SlideFlatEnv(smoke_cfg), filename=str(smoke_log_dir / "monitor.csv"))
    eval_env = Monitor(SlideFlatEnv(smoke_cfg))
    ppo_model = create_ppo_model(train_env, smoke_cfg)
    callbacks = build_slide_callbacks(smoke_cfg, eval_env=eval_env)
    ppo_model.learn(
        total_timesteps=int(args.train_steps),
        callback=callbacks,
        tb_log_name=smoke_cfg["logging"]["run_name"],
    )
    save_path = smoke_model_dir / smoke_cfg["logging"]["final_model_name"]
    ppo_model.save(str(save_path))
    train_env.close()
    eval_env.close()

    saved_zip = save_path if save_path.suffix == ".zip" else save_path.with_suffix(".zip")
    assert saved_zip.exists(), saved_zip
    event_files = list(smoke_log_dir.rglob("events.out.tfevents.*"))
    checkpoint_files = list(checkpoint_dir.glob("*.zip"))
    evaluation_file = eval_log_dir / "evaluations.npz"
    best_model_file = eval_log_dir / "best_model.zip"
    assert event_files, smoke_log_dir
    assert checkpoint_files, checkpoint_dir
    assert evaluation_file.exists(), evaluation_file
    assert best_model_file.exists(), best_model_file

    load_env = SlideFlatEnv(smoke_cfg)
    loaded_model = load_ppo_model(saved_zip, env=load_env, cfg=smoke_cfg)
    load_obs, _ = load_env.reset(seed=7)
    loaded_action, _ = loaded_model.predict(load_obs, deterministic=True)
    assert loaded_action.shape == (6,), loaded_action.shape
    assert np.isfinite(loaded_action).all()
    load_env.close()

    print(f"PPO smoke training saved model: {saved_zip}")
    print(f"TensorBoard event files: {len(event_files)}")
    print(f"Checkpoints: {len(checkpoint_files)}")
    print(f"Evaluation results: {evaluation_file}")
    print(f"Best model: {best_model_file}")
    print("Saved model reload/predict passed (same loader used by play script)")


if __name__ == "__main__":
    main()
