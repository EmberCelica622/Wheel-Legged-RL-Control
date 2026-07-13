from __future__ import annotations

import argparse
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco.viewer

from common.reproducibility import seed_everything
from common.run_manager import ModelSelection, resolve_model_selection
from common.terminal_logging import tee_terminal, terminal_log_path
from envs.slide_task_factory import create_slide_env, load_slide_config, slide_task_id
from rl.ppo import load_ppo_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained slide task PPO policy.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="Run directory containing config.yaml and models/.")
    source.add_argument("--model", help="PPO model .zip path; requires --config.")
    parser.add_argument("--config", help="Config path used with --model.")
    parser.add_argument("--model-kind", choices=("best", "final"), default="final")
    parser.add_argument("--seed", type=int, default=None, help="Override play.seed.")
    parser.add_argument("--command-vx", type=float, default=None, help="Fix vx for every reset.")
    parser.add_argument("--command-yaw-rate", type=float, default=0.0, help="Yaw-rate used with --command-vx.")
    parser.add_argument(
        "--repeat-same-initial-state",
        dest="repeat_same_initial_state",
        action="store_true",
        default=None,
        help="Reuse play.seed on every reset, including the sampled command.",
    )
    parser.add_argument(
        "--vary-initial-state",
        dest="repeat_same_initial_state",
        action="store_false",
        help="Seed the first reset only, then continue the environment RNG stream.",
    )
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=None,
        help="Use deterministic policy actions.",
    )
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--realtime-factor", type=float, default=None)
    return parser.parse_args()


def _play(args: argparse.Namespace, selection: ModelSelection) -> None:
    cfg = load_slide_config(selection.config)
    play_cfg = cfg.get("play", {})
    deterministic = (
        bool(play_cfg.get("deterministic", True))
        if args.deterministic is None
        else bool(args.deterministic)
    )
    repeat_same = (
        bool(play_cfg.get("repeat_same_initial_state", False))
        if args.repeat_same_initial_state is None
        else bool(args.repeat_same_initial_state)
    )
    realtime_factor = (
        float(play_cfg.get("realtime_factor", 1.0))
        if args.realtime_factor is None
        else float(args.realtime_factor)
    )
    realtime_factor = max(realtime_factor, 1e-6)
    play_seed = int(args.seed if args.seed is not None else play_cfg.get("seed", cfg.get("seed", 1)))
    reset_options = (
        None
        if args.command_vx is None
        else {
            "command": [float(args.command_vx), float(args.command_yaw_rate)],
            "fixed_command": True,
        }
    )

    seed_everything(play_seed)
    env = create_slide_env(cfg)
    model = load_ppo_model(selection.model, cfg=cfg)
    target_dt = env.control_dt / realtime_factor
    first_reset = True

    print(f"Task: {slide_task_id(cfg)}")
    print(f"Model: {selection.model}")
    print(f"Play seed: {play_seed}, repeat_same_initial_state: {repeat_same}")
    if reset_options:
        print(
            "Command mode: fixed override "
            f"vx={reset_options['command'][0]}, yaw_rate={reset_options['command'][1]}"
        )
    else:
        print("Command mode: environment config")

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            def reset_episode() -> tuple[object, float]:
                nonlocal first_reset
                reset_seed = play_seed if first_reset or repeat_same else None
                obs, _ = env.reset(seed=reset_seed, options=reset_options)
                first_reset = False
                viewer.sync()
                return obs, float(env.data.time)

            obs, last_sim_time = reset_episode()
            while viewer.is_running():
                frame_start = time.perf_counter()
                viewer.sync()

                current_sim_time = float(env.data.time)
                if current_sim_time + 1e-9 < last_sim_time:
                    print("Viewer reset detected. Performing full environment reset.")
                    obs, last_sim_time = reset_episode()
                    continue

                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _, terminated, truncated, info = env.step(action)
                last_sim_time = float(env.data.time)
                viewer.sync()

                if terminated or truncated:
                    print(f"Episode reset: {info.get('termination_reason', 'done')}")
                    obs, last_sim_time = reset_episode()

                elapsed = time.perf_counter() - frame_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
    finally:
        env.close()


def main() -> int:
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

    log_context = nullcontext(None)
    if selection.run is not None:
        log_context = tee_terminal(terminal_log_path(selection.run.console, "play_terminal"))

    with log_context as log_path:
        if log_path is not None:
            print(f"Terminal log: {log_path}")
        try:
            _play(args, selection)
        except KeyboardInterrupt:
            print("Playback interrupted by user.", file=sys.stderr)
            return 130
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
