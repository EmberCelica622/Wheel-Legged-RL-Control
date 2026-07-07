from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from common.reproducibility import seed_everything
from common.run_manager import resolve_model_selection
from envs.slide_task_factory import (
    FixedCommandResetWrapper,
    create_slide_env,
    load_slide_config,
    slide_task_id,
)
from rl.ppo import load_ppo_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a slide task PPO model without rendering.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="Run directory containing config.yaml and models/.")
    source.add_argument("--model", help="PPO model .zip path; requires --config.")
    parser.add_argument("--config", help="Config path used with --model.")
    parser.add_argument("--model-kind", choices=("best", "final"), default="final")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--command-vx", type=float, default=None)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        selection = resolve_model_selection(
            run=args.run,
            model=args.model,
            config=args.config,
            model_kind=args.model_kind,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Unable to resolve evaluation input: {exc}") from exc

    cfg = load_slide_config(selection.config)
    task_id = slide_task_id(cfg)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1))
    seed_everything(seed)
    base_env = create_slide_env(cfg)
    if args.command_vx is not None:
        base_env = FixedCommandResetWrapper(base_env, [float(args.command_vx), 0.0])
    env = Monitor(base_env)
    env.reset(seed=seed)
    model = load_ppo_model(selection.model, env=env, cfg=cfg)
    try:
        rewards, lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=max(int(args.episodes), 1),
            deterministic=bool(args.deterministic),
            return_episode_rewards=True,
        )
    finally:
        env.close()

    selection.eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = selection.eval_dir / "metrics.json"
    metrics = {
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "model": str(selection.model),
        "config": str(selection.config),
        "task": task_id,
        "seed": seed,
        "deterministic": bool(args.deterministic),
        "episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "episode_rewards": [float(value) for value in rewards],
        "episode_lengths": [int(value) for value in lengths],
    }
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Mean reward: {metrics['mean_reward']:.3f} +/- {metrics['std_reward']:.3f}")
    print(f"Task: {task_id}")
    print(f"Evaluation metrics: {metrics_path}")


if __name__ == "__main__":
    main()
