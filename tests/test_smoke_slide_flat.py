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
from common.task_identity import canonical_task_id_from_config
from common.vec_env import create_evaluation_vec_env, create_training_vec_env
from envs.slide_task_factory import (
    create_slide_env,
    load_slide_config,
    normalize_slide_config,
    slide_task_id,
)
from rl.ppo import create_ppo_model, load_ppo_model


ACTIVE_REFERENCE_ROOTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "callbacks",
    REPO_ROOT / "common",
    REPO_ROOT / "configs",
    REPO_ROOT / "envs",
    REPO_ROOT / "scripts",
    REPO_ROOT / "tests",
)


def _smoke_config(tmp_path: Path) -> dict:
    cfg = copy.deepcopy(
        load_slide_config(REPO_ROOT / "configs" / "slide_variable_velocity_flat_v2.yaml")
    )
    cfg["output"] = {"root_dir": str(tmp_path), "run_id": "pytest_smoke"}
    cfg["experiment"]["name"] = "pytest_smoke"
    cfg["training"]["total_timesteps"] = 512
    cfg["training"]["n_envs"] = 1
    cfg["training"]["rollout_batch_size"] = 128
    cfg["logging"]["log_interval_steps"] = 64
    cfg["ppo"].update(
        {
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
    ppo_model.learn(
        total_timesteps=512,
        callback=callbacks,
        tb_log_name=canonical_task_id_from_config(cfg),
    )
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


def test_slide_task_factory_selects_current_tasks() -> None:
    cases = {
        "slide_fixed_velocity_flat_v1_legacy": "slide_fixed_velocity_flat_v1_legacy.yaml",
        "slide_fixed_velocity_flat_v1": "slide_fixed_velocity_flat_v1.yaml",
        "slide_variable_velocity_flat_v2": "slide_variable_velocity_flat_v2.yaml",
        "slide_dynamic_command_flat_v3": "slide_flat_v3.yaml",
    }
    for expected_task_id, filename in cases.items():
        cfg = load_slide_config(REPO_ROOT / "configs" / filename)
        assert slide_task_id(cfg) == expected_task_id
        env = create_slide_env(cfg)
        try:
            obs, _ = env.reset(seed=int(cfg["seed"]))
            assert obs.shape == (28,)
            assert np.isfinite(obs).all()
        finally:
            env.close()


def test_deprecated_config_fields_normalize_to_task_metadata() -> None:
    cfg = copy.deepcopy(
        load_slide_config(REPO_ROOT / "configs" / "slide_variable_velocity_flat_v2.yaml")
    )
    cfg.pop("task")
    cfg["experiment"]["task"] = "slide_flat"
    cfg["experiment"]["env_variant"] = "v2"
    normalized = normalize_slide_config(cfg)
    assert slide_task_id(normalized) == "slide_variable_velocity_flat_v2"
    assert "env_variant" not in normalized["experiment"]

    fixed_cfg = copy.deepcopy(load_slide_config(REPO_ROOT / "configs" / "slide_fixed_velocity_flat_v1.yaml"))
    fixed_cfg.pop("task")
    fixed_cfg["experiment"]["task"] = "slide_flat"
    fixed_cfg["experiment"]["env_variant"] = "v2"
    normalized_fixed = normalize_slide_config(fixed_cfg)
    assert slide_task_id(normalized_fixed) == "slide_fixed_velocity_flat_v1"

    v3_cfg = copy.deepcopy(load_slide_config(REPO_ROOT / "configs" / "slide_flat_v3.yaml"))
    v3_cfg.pop("task")
    v3_cfg.setdefault("env", {})["version"] = "v3"
    normalized_v3 = normalize_slide_config(v3_cfg)
    assert slide_task_id(normalized_v3) == "slide_dynamic_command_flat_v3"

    mismatched_cfg = copy.deepcopy(load_slide_config(REPO_ROOT / "configs" / "slide_flat_v3.yaml"))
    mismatched_cfg["env"]["version"] = "v2"
    try:
        normalize_slide_config(mismatched_cfg)
    except ValueError as exc:
        assert "env.version" in str(exc)
    else:
        raise AssertionError("Mismatched env.version was accepted")


def test_task_metadata_rejects_invalid_version() -> None:
    cfg = copy.deepcopy(load_slide_config(REPO_ROOT / "configs" / "slide_fixed_velocity_flat_v1.yaml"))
    cfg["task"]["version"] = "1"
    try:
        normalize_slide_config(cfg)
    except ValueError as exc:
        assert "task.version" in str(exc)
    else:
        raise AssertionError("Invalid task version was accepted")


def test_runtime_requires_task_metadata() -> None:
    try:
        slide_task_id({"experiment": {}})
    except ValueError as exc:
        assert "task metadata" in str(exc)
    else:
        raise AssertionError("Missing task metadata was accepted")


def test_deprecated_specific_task_ids_are_not_active() -> None:
    deprecated_tokens = [f"slide_flat_{suffix}" for suffix in ("v1", "v2")]
    active_files: list[Path] = []
    for root in ACTIVE_REFERENCE_ROOTS:
        if root.is_file():
            active_files.append(root)
        else:
            active_files.extend(
                path
                for path in root.rglob("*")
                if path.suffix in {".md", ".py", ".yaml", ".yml"}
                and "__pycache__" not in path.parts
            )

    offenders: list[str] = []
    for path in active_files:
        text = path.read_text(encoding="utf-8")
        for token in deprecated_tokens:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    assert not offenders


def test_fixed_velocity_legacy_and_v1_stance_semantics_differ() -> None:
    legacy_cfg = load_slide_config(
        REPO_ROOT / "configs" / "slide_fixed_velocity_flat_v1_legacy.yaml"
    )
    v1_cfg = load_slide_config(REPO_ROOT / "configs" / "slide_fixed_velocity_flat_v1.yaml")
    legacy_env = create_slide_env(legacy_cfg)
    v1_env = create_slide_env(v1_cfg)
    try:
        legacy_env.reset(seed=1)
        v1_env.reset(seed=1)
        assert slide_task_id(legacy_cfg) == "slide_fixed_velocity_flat_v1_legacy"
        assert slide_task_id(v1_cfg) == "slide_fixed_velocity_flat_v1"
        assert not getattr(legacy_env, "stance_regularization_enabled", False)
        assert v1_env.stance_regularization_enabled
        assert "wheel_longitudinal_offset_penalty" not in legacy_env.reward_weights
        assert v1_env.reward_weights["wheel_longitudinal_offset_penalty"] == 0.10
        assert np.array_equal(legacy_env.command, [0.8, 0.0])
        assert np.array_equal(v1_env.command, [0.8, 0.0])
    finally:
        legacy_env.close()
        v1_env.close()


def test_run_manager_layout_and_collision(tmp_path: Path) -> None:
    cfg = _smoke_config(tmp_path)
    cfg["experiment"]["name"] = "unsafe/name"
    paths = create_run(cfg, run_id="debug/seed0")

    assert paths.run_dir == tmp_path / "slide_variable_velocity_flat_v2" / "unsafe_name" / "debug_seed0"
    assert all(path.is_dir() for path in paths.artifact_directories())
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["seed"] == cfg["seed"]
    assert metadata["task_id"] == "slide_variable_velocity_flat_v2"
    assert metadata["python_version"]
    assert metadata["torch_version"]
    assert metadata["mujoco_version"]
    assert metadata["stable_baselines3_version"]

    v1_cfg = load_slide_config(REPO_ROOT / "configs" / "slide_fixed_velocity_flat_v1.yaml")
    v1_cfg["output"] = {"root_dir": str(tmp_path), "run_id": "v1_debug"}
    v1_cfg["experiment"]["name"] = "unsafe/name"
    v1_paths = create_run(v1_cfg, run_id="v1_debug")
    assert v1_paths.run_dir == tmp_path / "slide_fixed_velocity_flat_v1" / "unsafe_name" / "v1_debug"
    assert v1_paths.run_dir.parent.parent != paths.run_dir.parent.parent

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
