from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from callbacks.slide_callbacks import build_slide_callbacks
from common.reproducibility import seed_everything
from common.run_manager import RunPaths, create_run, open_run, resolve_existing_file
from common.terminal_logging import tee_terminal, terminal_log_path
from common.vec_env import create_evaluation_vec_env, create_training_vec_env
from envs.slide_task_factory import load_slide_config, slide_task_id
from rl.ppo import PPOTrainer, resolve_ppo_rollout_config, warm_start_ppo_policy


_CHECKPOINT_STEP = re.compile(r"_(\d+)_steps\.zip$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configured slide task policy with PPO.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "slide_fixed_velocity_flat_v1.yaml"),
        help="Experiment YAML used for scratch or warm-start training.",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Override target/additional timesteps.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for a new run.")
    parser.add_argument("--n-envs", type=int, default=None, help="Override n_envs for a new run.")
    parser.add_argument("--run-id", default=None, help="Explicit id for a new run; collisions are rejected.")

    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume-run", help="Existing run directory to continue in place.")
    initialization.add_argument("--warm-start", help="Checkpoint whose policy initializes a new run.")
    parser.add_argument(
        "--resume-checkpoint",
        help="Checkpoint used with --resume-run; defaults to the highest-step checkpoint.",
    )
    return parser.parse_args()


def _latest_resume_checkpoint(paths: RunPaths) -> Path:
    candidates = list(paths.checkpoints.glob("*.zip"))
    if candidates:
        def key(path: Path) -> tuple[int, float]:
            match = _CHECKPOINT_STEP.search(path.name)
            return (int(match.group(1)) if match else -1, path.stat().st_mtime)

        return max(candidates, key=key)

    final_model = paths.models / "final_model.zip"
    if final_model.is_file():
        return final_model
    raise FileNotFoundError(f"No checkpoint or final model found in run: {paths.run_dir}")


def _prepare_new_run(args: argparse.Namespace) -> tuple[dict, RunPaths, Path | None, str]:
    cfg = load_slide_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.timesteps is not None:
        cfg.setdefault("training", {})["total_timesteps"] = int(args.timesteps)
    if args.n_envs is not None:
        cfg.setdefault("training", {})["n_envs"] = int(args.n_envs)

    training_cfg = cfg.setdefault("training", {})
    n_envs = int(training_cfg.get("n_envs", 1))
    resolve_ppo_rollout_config(cfg, n_envs)

    configured_warm_start = training_cfg.get("warm_start_checkpoint")
    warm_value = args.warm_start or configured_warm_start
    warm_checkpoint = None
    initialization = "scratch"
    if warm_value:
        warm_checkpoint = resolve_existing_file(warm_value, field="Warm-start checkpoint")
        initialization = "warm_start"
        training_cfg["warm_start_checkpoint"] = str(warm_checkpoint)
    else:
        training_cfg["warm_start_checkpoint"] = None
    training_cfg["initialization"] = initialization

    paths = create_run(cfg, run_id=args.run_id)
    return cfg, paths, warm_checkpoint, initialization


def _prepare_resume(args: argparse.Namespace) -> tuple[dict, RunPaths, Path, str]:
    if args.run_id or args.seed is not None or args.n_envs is not None:
        raise ValueError("--resume-run uses the frozen run config; do not pass --run-id, --seed, or --n-envs")

    paths = open_run(args.resume_run)
    cfg = load_slide_config(paths.config)
    n_envs = int(cfg.get("training", {}).get("n_envs", 1))
    resolve_ppo_rollout_config(cfg, n_envs)
    checkpoint = (
        resolve_existing_file(args.resume_checkpoint, field="Resume checkpoint")
        if args.resume_checkpoint
        else _latest_resume_checkpoint(paths)
    )
    return cfg, paths, checkpoint, "resume"


def _train(
    cfg: dict,
    run_paths: RunPaths,
    checkpoint: Path | None,
    initialization: str,
    timesteps_override: int | None,
) -> None:
    seed = int(cfg.get("seed", 1))
    training_cfg = cfg.setdefault("training", {})
    n_envs = int(training_cfg.get("n_envs", 1))
    start_method = str(training_cfg.get("start_method", "spawn")).lower()
    eval_seed = seed + int(training_cfg.get("eval_seed_offset", 10000))
    rollout = resolve_ppo_rollout_config(cfg, n_envs)
    seed_everything(seed)

    print(f"Run directory: {run_paths.run_dir}")
    task_id = slide_task_id(cfg)
    print(f"Task: {task_id}")
    print(f"Initialization: {initialization}")
    print(f"n_envs: {rollout['n_envs']}")
    print(f"n_steps: {rollout['n_steps']}")
    print(f"rollout_batch_size: {rollout['rollout_batch_size']}")
    print(f"batch_size: {rollout['batch_size']}")

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

        if initialization == "resume":
            assert checkpoint is not None
            trainer = PPOTrainer.load(
                checkpoint,
                env=env,
                cfg=cfg,
                tensorboard_log=run_paths.tensorboard,
                override_ppo_config=True,
            )
            target_timesteps = int(training_cfg.get("total_timesteps", 1000000))
            learn_timesteps = (
                int(timesteps_override)
                if timesteps_override is not None
                else max(target_timesteps - int(trainer.model.num_timesteps), 0)
            )
            reset_num_timesteps = False
            print(f"Resume checkpoint: {checkpoint}")
            print(f"Checkpoint timesteps: {trainer.model.num_timesteps}")
        else:
            trainer = PPOTrainer(env=env, cfg=cfg, tensorboard_log=run_paths.tensorboard)
            if initialization == "warm_start":
                assert checkpoint is not None
                source = warm_start_ppo_policy(
                    trainer.model,
                    checkpoint,
                    device=str(cfg.get("ppo", {}).get("device", "auto")),
                )
                print(f"Warm-start checkpoint: {checkpoint}")
                print(f"Source timesteps: {source.num_timesteps}")
                print("Warm-start copied policy weights; optimizer and schedules were reset.")
            learn_timesteps = int(training_cfg.get("total_timesteps", 1000000))
            reset_num_timesteps = True

        if learn_timesteps <= 0:
            print("Configured target timesteps already reached; no learning steps were run.")
            return

        callback = build_slide_callbacks(cfg, eval_env=eval_env, run_paths=run_paths, n_envs=n_envs)
        trainer.learn(
            total_timesteps=learn_timesteps,
            callback=callback,
            tb_log_name=task_id,
            reset_num_timesteps=reset_num_timesteps,
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


def main() -> int:
    multiprocessing.freeze_support()
    args = parse_args()
    if args.resume_checkpoint and not args.resume_run:
        raise SystemExit("--resume-checkpoint requires --resume-run")
    if args.resume_run and args.warm_start:
        raise SystemExit("--resume-run and --warm-start are mutually exclusive")

    try:
        if args.resume_run:
            cfg, run_paths, checkpoint, initialization = _prepare_resume(args)
        else:
            cfg, run_paths, checkpoint, initialization = _prepare_new_run(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Unable to prepare training run: {exc}") from exc

    os.environ["TMP"] = str(run_paths.tmp)
    os.environ["TEMP"] = str(run_paths.tmp)
    log_path = terminal_log_path(run_paths.console, "train_terminal")
    with tee_terminal(log_path):
        print(f"Terminal log: {log_path}")
        try:
            _train(cfg, run_paths, checkpoint, initialization, args.timesteps)
        except KeyboardInterrupt:
            print("Training interrupted by user.", file=sys.stderr)
            return 130
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
