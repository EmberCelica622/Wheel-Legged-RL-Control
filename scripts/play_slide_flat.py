from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco.viewer

from common.reproducibility import seed_everything
from common.run_manager import resolve_model_selection
from envs.slide_flat_factory import create_slide_env, load_slide_config
from rl.ppo import load_ppo_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained flat sliding PPO policy.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="Run directory containing config.yaml and models/.")
    source.add_argument("--model", help="PPO model .zip path; requires --config.")
    parser.add_argument("--config", help="Config path used with --model.")
    parser.add_argument("--model-kind", choices=("best", "final"), default="final")
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=None,
        help="Use deterministic policy actions.",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic policy actions.",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=None,
        help="Playback speed multiplier. Defaults to play.realtime_factor in YAML.",
    )
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
        raise SystemExit(f"Unable to resolve playback input: {exc}") from exc

    cfg = load_slide_config(selection.config)
    deterministic = args.deterministic
    if deterministic is None:
        deterministic = bool(cfg.get("play", {}).get("deterministic", True))

    realtime_factor = args.realtime_factor
    if realtime_factor is None:
        realtime_factor = float(cfg.get("play", {}).get("realtime_factor", 1.0))
    realtime_factor = max(float(realtime_factor), 1e-6)

    seed = int(cfg.get("seed", 1))
    seed_everything(seed)
    env = create_slide_env(cfg)
    obs, _ = env.reset(seed=seed)
    model = load_ppo_model(selection.model, cfg=cfg)
    target_dt = env.control_dt / realtime_factor

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            frame_start = time.perf_counter()
            viewer.sync()
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            viewer.sync()

            if terminated or truncated:
                print(f"Episode reset: {info.get('termination_reason', 'done')}")
                obs, _ = env.reset()
                viewer.sync()

            elapsed = time.perf_counter() - frame_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    env.close()


if __name__ == "__main__":
    main()
