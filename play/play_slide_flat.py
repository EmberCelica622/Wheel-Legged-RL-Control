from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco.viewer

from envs.slide_flat import SlideFlatEnv, load_slide_config
from rl.ppo import load_ppo_model


def resolve_path(cfg: dict, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(cfg.get("_base_dir", REPO_ROOT)).expanduser().resolve() / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained flat sliding PPO policy.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "slide_flat.yaml"),
        help="Path to slide flat YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional PPO checkpoint path. Defaults to play.checkpoint in YAML.",
    )
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
    cfg = load_slide_config(args.config)

    checkpoint_value = args.checkpoint or cfg.get("play", {}).get("checkpoint")
    if not checkpoint_value:
        raise ValueError("No checkpoint specified. Set play.checkpoint in YAML or pass --checkpoint.")
    checkpoint = resolve_path(cfg, checkpoint_value)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    deterministic = args.deterministic
    if deterministic is None:
        deterministic = bool(cfg.get("play", {}).get("deterministic", True))

    realtime_factor = args.realtime_factor
    if realtime_factor is None:
        realtime_factor = float(cfg.get("play", {}).get("realtime_factor", 1.0))
    realtime_factor = max(float(realtime_factor), 1e-6)

    env = SlideFlatEnv(cfg)
    obs, _ = env.reset(seed=int(cfg.get("env", {}).get("seed", 1)))
    model = load_ppo_model(checkpoint, cfg=cfg)

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
