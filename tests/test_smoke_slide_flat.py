from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
from stable_baselines3.common.env_checker import check_env
from callbacks.slide_callbacks import build_slide_callbacks
from common.run_manager import create_run, resolve_config_path, resolve_model_selection
from common.vec_env import create_evaluation_vec_env, create_training_vec_env
from envs.slide_flat_factory import create_slide_env, load_slide_config, slide_env_variant
from rl.ppo import create_ppo_model, load_ppo_model


def _smoke_config(tmp_path: Path) -> dict:
    cfg = copy.deepcopy(load_slide_config(REPO_ROOT / "configs" / "slide_flat_v2.yaml"))
    cfg["output"] = {"root_dir": str(tmp_path), "run_id": "pytest_smoke"}
    cfg["experiment"]["name"] = "pytest_smoke"
    cfg["training"]["total_timesteps"] = 512
    cfg["training"]["n_envs"] = 1
    cfg["logging"]["log_interval_steps"] = 64
    cfg["ppo"].update(
        {
            "n_steps": 128,
            "batch_size": 32,
            "n_epochs": 1,
            "verbose": 0,
            "policy_kwargs": {"net_arch": [32, 32]},
        }
    )
    cfg["callbacks"].update(
        {
            "checkpoint_freq": 128,
            "eval_freq": 128,
            "n_eval_episodes": 2,
            "deterministic_eval": True,
        }
    )
    return cfg


def test_slide_flat_smoke(tmp_path: Path) -> None:
    cfg = _smoke_config(tmp_path)
    seed = int(cfg["seed"])
    xml_path = resolve_config_path(cfg, cfg["env"]["xml_path"])
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert (model.nq, model.nv, model.nu) == (13, 12, 6)

    env = create_slide_env(cfg)
    obs, _ = env.reset(seed=seed)
    assert obs.shape == (28,)
    assert env.action_space.shape == (6,)
    assert np.isfinite(obs).all()
    check_env(env, warn=True, skip_render_check=True)

    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert np.isfinite(env.data.qpos).all()
        assert np.isfinite(env.data.qvel).all()
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    run_paths = create_run(cfg, run_id="pytest_smoke")
    train_env = create_training_vec_env(
        run_paths.config,
        seed=seed,
        n_envs=1,
        monitor_path=run_paths.tensorboard / "monitor.csv",
    )
    eval_env = create_evaluation_vec_env(run_paths.config, seed=seed + 10000)
    ppo_model = create_ppo_model(train_env, cfg, tensorboard_log=run_paths.tensorboard)
    callbacks = build_slide_callbacks(cfg, eval_env=eval_env, run_paths=run_paths)
    ppo_model.learn(total_timesteps=512, callback=callbacks, tb_log_name=run_paths.run_dir.name)
    final_path = run_paths.models / "final_model"
    ppo_model.save(str(final_path))
    train_env.close()
    eval_env.close()

    final_zip = final_path.with_suffix(".zip")
    assert final_zip.is_file()
    assert list(run_paths.tensorboard.rglob("events.out.tfevents.*"))
    assert list(run_paths.checkpoints.glob("*.zip"))
    assert (run_paths.eval / "evaluations.npz").is_file()
    assert (run_paths.models / "best_model.zip").is_file()
    assert run_paths.config.is_file()
    assert run_paths.metadata.is_file()

    frozen_cfg = load_slide_config(run_paths.config)
    frozen_env = create_slide_env(frozen_cfg)
    loaded_model = load_ppo_model(final_zip, env=frozen_env, cfg=frozen_cfg)
    loaded_obs, _ = frozen_env.reset(seed=7)
    loaded_action, _ = loaded_model.predict(loaded_obs, deterministic=True)
    assert loaded_action.shape == (6,)
    assert np.isfinite(loaded_action).all()
    frozen_env.close()


def test_slide_env_factory_selects_v1_and_v2() -> None:
    for variant in ("v1", "v2"):
        cfg = load_slide_config(REPO_ROOT / "configs" / f"slide_flat_{variant}.yaml")
        assert slide_env_variant(cfg) == variant
        env = create_slide_env(cfg)
        try:
            obs, _ = env.reset(seed=int(cfg["seed"]))
            assert obs.shape == (28,)
            assert np.isfinite(obs).all()
        finally:
            env.close()


def test_run_manager_layout_and_collision(tmp_path: Path) -> None:
    cfg = _smoke_config(tmp_path)
    cfg["experiment"]["name"] = "unsafe/name"
    paths = create_run(cfg, run_id="debug/seed0")

    assert paths.run_dir == tmp_path / "slide_flat" / "v2" / "unsafe_name" / "debug_seed0"
    assert all(path.is_dir() for path in paths.artifact_directories())
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["seed"] == cfg["seed"]
    assert metadata["python_version"]
    assert metadata["torch_version"]
    assert metadata["mujoco_version"]
    assert metadata["stable_baselines3_version"]

    (paths.models / "final_model.zip").touch()
    selection = resolve_model_selection(
        run=paths.run_dir,
        model=None,
        config=None,
        model_kind="final",
    )
    assert selection.config == paths.config
    assert selection.eval_dir == paths.eval

    try:
        create_run(cfg, run_id="debug/seed0")
    except FileExistsError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("Existing explicit run id was overwritten")
