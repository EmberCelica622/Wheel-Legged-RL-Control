from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3.common.monitor import Monitor

from callbacks.slide_callbacks import build_slide_callbacks
from common.reproducibility import seed_everything
from common.run_manager import create_run, resolve_existing_file
from envs.slide_flat_v2 import SlideFlatEnv, load_slide_config
from rl.ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flat-ground sliding policy with PPO.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "slide_flat_v2.yaml"),
        help="Path to the experiment YAML config.",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Override training.total_timesteps.")
    parser.add_argument("--seed", type=int, default=None, help="Override the top-level experiment seed.")
    parser.add_argument("--run-id", default=None, help="Explicit unique run id; existing runs are rejected.")
    parser.add_argument("--resume", default=None, help="Optional PPO checkpoint to resume into a new run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_slide_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.timesteps is not None:
        cfg.setdefault("training", {})["total_timesteps"] = int(args.timesteps)

    seed = int(cfg.get("seed", 1))
    seed_everything(seed)
    try:
        run_paths = create_run(cfg, run_id=args.run_id)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(f"Unable to create run: {exc}") from exc

    os.environ["TMP"] = str(run_paths.tmp)
    os.environ["TEMP"] = str(run_paths.tmp)
    print(f"Run directory: {run_paths.run_dir}")

    env = Monitor(SlideFlatEnv(cfg), filename=str(run_paths.tensorboard / "monitor.csv"))
    eval_env = Monitor(SlideFlatEnv(cfg))
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1)

    print("Observation schema: slide-flat v2 (28D). Existing 25D PPO checkpoints are incompatible.")
    try:
        if args.resume:
            checkpoint = resolve_existing_file(args.resume, field="Resume checkpoint")
            try:
                trainer = PPOTrainer.load(
                    checkpoint,
                    env=env,
                    cfg=cfg,
                    tensorboard_log=run_paths.tensorboard,
                )
            except ValueError as exc:
                raise ValueError(
                    "Checkpoint is incompatible with the 28D slide-flat observation schema; "
                    "start a new training run instead."
                ) from exc
        else:
            trainer = PPOTrainer(env=env, cfg=cfg, tensorboard_log=run_paths.tensorboard)

        n_envs = int(getattr(trainer.model.get_env(), "num_envs", 1))
        callback = build_slide_callbacks(cfg, eval_env=eval_env, run_paths=run_paths, n_envs=n_envs)
        trainer.learn(
            total_timesteps=int(cfg.get("training", {}).get("total_timesteps", 1000000)),
            callback=callback,
            tb_log_name=run_paths.run_dir.name,
        )
        final_path = run_paths.models / "final_model"
        trainer.save(final_path)
        print(f"Training complete. Saved model: {final_path.with_suffix('.zip')}")
        print(f"TensorBoard logs: {run_paths.tensorboard}")
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
