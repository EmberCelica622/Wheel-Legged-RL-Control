from __future__ import annotations

import argparse
import copy
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.reproducibility import seed_everything
from common.vec_env import create_training_vec_env
from envs.slide_flat_factory import load_slide_config
from rl.ppo import create_ppo_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark slide-flat vector sampling and short PPO training.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "slide_flat_v2.yaml"))
    parser.add_argument("--n-envs", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--sample-transitions", type=int, default=8192)
    parser.add_argument("--warmup-transitions", type=int, default=1024)
    parser.add_argument("--train-timesteps", type=int, default=4096)
    parser.add_argument("--rollout-size", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=1)
    return parser.parse_args()


def vector_steps(transitions: int, n_envs: int) -> int:
    return max((int(transitions) + int(n_envs) - 1) // int(n_envs), 1)


def main() -> None:
    multiprocessing.freeze_support()
    args = parse_args()
    base_cfg = load_slide_config(args.config)
    config_path = Path(args.config).expanduser().resolve()
    seed = int(base_cfg.get("seed", 1))
    seed_everything(seed)

    print("n_envs startup_s sample_transitions_per_s ppo_fps actual_train_timesteps")
    for n_envs in args.n_envs:
        n_envs = int(n_envs)
        if args.rollout_size % n_envs != 0:
            raise ValueError(f"rollout-size {args.rollout_size} must be divisible by n_envs {n_envs}")

        startup_begin = time.perf_counter()
        env = create_training_vec_env(
            config_path,
            seed=seed,
            n_envs=n_envs,
            start_method="spawn",
        )
        startup_seconds = time.perf_counter() - startup_begin
        try:
            env.reset()
            actions = np.zeros((n_envs, 6), dtype=np.float32)
            for _ in range(vector_steps(args.warmup_transitions, n_envs)):
                env.step(actions)

            sample_steps = vector_steps(args.sample_transitions, n_envs)
            sample_begin = time.perf_counter()
            for _ in range(sample_steps):
                env.step(actions)
            sample_seconds = time.perf_counter() - sample_begin
            sampled_transitions = sample_steps * n_envs
            sample_fps = sampled_transitions / max(sample_seconds, 1e-9)

            cfg = copy.deepcopy(base_cfg)
            cfg["ppo"]["n_steps"] = args.rollout_size // n_envs
            cfg["ppo"]["n_epochs"] = int(args.n_epochs)
            model = create_ppo_model(env, cfg)
            train_begin = time.perf_counter()
            model.learn(total_timesteps=int(args.train_timesteps))
            train_seconds = time.perf_counter() - train_begin
            train_fps = model.num_timesteps / max(train_seconds, 1e-9)
            print(
                f"{n_envs:6d} {startup_seconds:9.3f} {sample_fps:24.1f} "
                f"{train_fps:8.1f} {model.num_timesteps:22d}"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
