from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3.common.monitor import Monitor

from callbacks.slide_callbacks import build_slide_callbacks
from envs.slide_flat import SlideFlatEnv, load_slide_config
from rl.ppo import PPOTrainer


def resolve_path(cfg: dict, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(cfg.get("_base_dir", REPO_ROOT)).expanduser().resolve() / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flat-ground sliding policy with PPO.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "slide_flat.yaml"),
        help="Path to slide flat YAML config.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override training.total_timesteps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override env.seed and ppo.seed.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional PPO checkpoint path to resume from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_slide_config(args.config)

    if args.seed is not None:
        cfg.setdefault("env", {})["seed"] = args.seed
        cfg.setdefault("ppo", {})["seed"] = args.seed

    tensorboard_log_dir = resolve_path(cfg, cfg["logging"]["tensorboard_log_dir"])
    model_dir = resolve_path(cfg, cfg["logging"]["model_dir"])
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    run_name = str(cfg.get("logging", {}).get("run_name", "slide_flat"))
    env = Monitor(SlideFlatEnv(cfg), filename=str(tensorboard_log_dir / "monitor.csv"))
    eval_env = Monitor(SlideFlatEnv(cfg))
    seed = int(cfg.get("env", {}).get("seed", cfg.get("ppo", {}).get("seed", 1)))
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1)

    print("Observation schema: slide-flat v2 (28D). Existing 25D PPO checkpoints are incompatible.")

    if args.resume:
        checkpoint = Path(args.resume).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        try:
            trainer = PPOTrainer.load(checkpoint, env=env, cfg=cfg)
        except ValueError as exc:
            raise ValueError(
                "Checkpoint is incompatible with the 28D slide-flat observation schema; "
                "start a new training run instead."
            ) from exc
    else:
        trainer = PPOTrainer(env=env, cfg=cfg)

    n_envs = int(getattr(trainer.model.get_env(), "num_envs", 1))
    callback = build_slide_callbacks(cfg, eval_env=eval_env, n_envs=n_envs)

    total_timesteps = args.timesteps
    if total_timesteps is None:
        total_timesteps = int(cfg.get("training", {}).get("total_timesteps", 1000000))

    try:
        trainer.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=run_name,
        )
        final_name = cfg.get("logging", {}).get("final_model_name", "final_model")
        final_path = model_dir / final_name
        trainer.save(final_path)
        saved_path = final_path if final_path.suffix == ".zip" else final_path.with_suffix(".zip")
        print(f"Training complete. Saved model: {saved_path}")
        print(f"TensorBoard logs: {tensorboard_log_dir}")
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
