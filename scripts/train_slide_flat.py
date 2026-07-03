from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from callbacks.slide_callbacks import build_slide_callbacks
from common.reproducibility import seed_everything
from common.run_manager import create_run, resolve_existing_file
from common.vec_env import create_evaluation_vec_env, create_training_vec_env
from envs.slide_flat_factory import load_slide_config, slide_env_variant
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
    parser.add_argument("--n-envs", type=int, default=None, help="Override training.n_envs.")
    parser.add_argument("--run-id", default=None, help="Explicit unique run id; existing runs are rejected.")
    parser.add_argument("--resume", default=None, help="Optional PPO checkpoint to resume into a new run.")
    return parser.parse_args()


def main() -> None:
    multiprocessing.freeze_support()
    args = parse_args()
    cfg = load_slide_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.timesteps is not None:
        cfg.setdefault("training", {})["total_timesteps"] = int(args.timesteps)
    if args.n_envs is not None:
        cfg.setdefault("training", {})["n_envs"] = int(args.n_envs)

    seed = int(cfg.get("seed", 1))
    training_cfg = cfg.setdefault("training", {})
    n_envs = int(training_cfg.get("n_envs", 1))
    start_method = str(training_cfg.get("start_method", "spawn")).lower()
    eval_seed = seed + int(training_cfg.get("eval_seed_offset", 10000))
    seed_everything(seed)
    try:
        run_paths = create_run(cfg, run_id=args.run_id)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(f"Unable to create run: {exc}") from exc

    os.environ["TMP"] = str(run_paths.tmp)
    os.environ["TEMP"] = str(run_paths.tmp)
    print(f"Run directory: {run_paths.run_dir}")
    print(f"Environment: {slide_env_variant(cfg)}, workers: {n_envs}, start method: {start_method}")

    env = None
    eval_env = None
    try:
        env = create_training_vec_env(
            run_paths.config,
            seed=seed,
            n_envs=n_envs,
            start_method=start_method,
            monitor_path=run_paths.tensorboard / "monitor.csv",
        )
        eval_env = create_evaluation_vec_env(run_paths.config, seed=eval_seed)

        print("Observation schema: slide-flat 28D. Existing 25D PPO checkpoints are incompatible.")
        if args.resume:
            checkpoint = resolve_existing_file(args.resume, field="Resume checkpoint")
            try:
                trainer = PPOTrainer.load(
                    checkpoint,
                    env=env,
                    cfg=cfg,
                    tensorboard_log=run_paths.tensorboard,
                    override_ppo_config=True,
                )
            except ValueError as exc:
                raise ValueError(
                    "Unable to resume checkpoint with the current environment and PPO rollout "
                    f"configuration: {exc}"
                ) from exc
        else:
            trainer = PPOTrainer(env=env, cfg=cfg, tensorboard_log=run_paths.tensorboard)

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
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
