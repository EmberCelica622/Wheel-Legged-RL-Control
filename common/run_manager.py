from __future__ import annotations

import copy
import json
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import yaml

from common.task_identity import canonical_task_id_from_config, normalize_task_config


_UNSAFE_COMPONENT = re.compile(r"[^\w.-]+", flags=re.UNICODE)


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoints: Path
    models: Path
    eval: Path
    tensorboard: Path
    console: Path
    tmp: Path
    videos: Path
    config: Path
    metadata: Path

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "RunPaths":
        root = Path(run_dir).expanduser().resolve()
        return cls(
            run_dir=root,
            checkpoints=root / "checkpoints",
            models=root / "models",
            eval=root / "eval",
            tensorboard=root / "logs" / "tensorboard",
            console=root / "logs" / "console",
            tmp=root / "logs" / "tmp",
            videos=root / "videos",
            config=root / "config.yaml",
            metadata=root / "metadata.json",
        )

    def artifact_directories(self) -> tuple[Path, ...]:
        return (
            self.checkpoints,
            self.models,
            self.eval,
            self.tensorboard,
            self.console,
            self.tmp,
            self.videos,
        )


@dataclass(frozen=True)
class ModelSelection:
    model: Path
    config: Path
    eval_dir: Path
    run: RunPaths | None


def sanitize_component(value: Any, field: str) -> str:
    raw = str(value).strip()
    cleaned = _UNSAFE_COMPONENT.sub("_", raw).strip(" ._")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"{field} must contain at least one safe character")
    return cleaned[:120]


def resolve_config_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base_dir = Path(cfg.get("_base_dir", Path.cwd())).expanduser().resolve()
    return (base_dir / path).resolve()


def resolve_existing_file(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} not found: {path}")
    return path


def _distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _resolved_config(
    cfg: dict[str, Any],
    *,
    base_dir: Path,
    output_root: Path,
    run_id: str,
    seed: int,
    experiment_name: str,
) -> dict[str, Any]:
    resolved = normalize_task_config(
        copy.deepcopy({key: value for key, value in cfg.items() if not key.startswith("_")})
    )
    resolved.pop("_task_id", None)
    resolved["seed"] = seed
    resolved.setdefault("experiment", {})["name"] = experiment_name
    resolved.setdefault("output", {}).update(
        {"root_dir": str(output_root), "run_id": run_id}
    )

    xml_value = resolved.get("env", {}).get("xml_path")
    if xml_value:
        xml_path = Path(xml_value).expanduser()
        if not xml_path.is_absolute():
            xml_path = base_dir / xml_path
        resolved["env"]["xml_path"] = str(xml_path.resolve())
    return resolved


def create_run(
    cfg: dict[str, Any],
    *,
    run_id: str | None = None,
    now: datetime | None = None,
) -> RunPaths:
    normalized_cfg = normalize_task_config(cfg)
    seed = int(normalized_cfg.get("seed", 1))
    experiment = normalized_cfg.get("experiment", {})
    output = normalized_cfg.get("output", {})
    task_id = sanitize_component(canonical_task_id_from_config(normalized_cfg), "task")
    experiment_name = sanitize_component(experiment.get("name", "ppo"), "experiment.name")

    base_dir = Path(normalized_cfg.get("_base_dir", Path.cwd())).expanduser().resolve()
    output_root = Path(output.get("root_dir", "runs")).expanduser()
    if not output_root.is_absolute():
        output_root = base_dir / output_root
    output_root = output_root.resolve()
    run_parent = output_root / task_id / experiment_name

    requested_id = run_id if run_id is not None else output.get("run_id", "auto")
    if requested_id in {None, "", "auto"}:
        timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S")
        base_run_id = f"{timestamp}_seed{seed}"
        for suffix in range(100):
            actual_run_id = base_run_id if suffix == 0 else f"{base_run_id}_{suffix:02d}"
            run_dir = run_parent / actual_run_id
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(f"Could not allocate a unique run under: {run_parent}")
    else:
        actual_run_id = sanitize_component(requested_id, "output.run_id")
        run_dir = run_parent / actual_run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(f"Run already exists; refusing to overwrite: {run_dir}") from exc

    paths = RunPaths.from_run_dir(run_dir)
    for directory in paths.artifact_directories():
        directory.mkdir(parents=True, exist_ok=False)

    resolved_cfg = _resolved_config(
        normalized_cfg,
        base_dir=base_dir,
        output_root=output_root,
        run_id=actual_run_id,
        seed=seed,
        experiment_name=experiment_name,
    )
    with paths.config.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved_cfg, stream, sort_keys=False)

    created_at = (now or datetime.now().astimezone()).astimezone().isoformat()
    metadata = {
        "created_at": created_at,
        "seed": seed,
        "task_id": task_id,
        "initialization": str(normalized_cfg.get("training", {}).get("initialization", "scratch")),
        "python_version": platform.python_version(),
        "torch_version": _distribution_version("torch"),
        "mujoco_version": _distribution_version("mujoco"),
        "stable_baselines3_version": _distribution_version("stable-baselines3"),
    }
    warm_start_checkpoint = normalized_cfg.get("training", {}).get("warm_start_checkpoint")
    if warm_start_checkpoint:
        metadata["warm_start_checkpoint"] = str(warm_start_checkpoint)
    commit = _git_commit(base_dir)
    if commit:
        metadata["git_commit"] = commit
    with paths.metadata.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return paths


def open_run(run_dir: str | Path) -> RunPaths:
    path = Path(run_dir).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    paths = RunPaths.from_run_dir(path)
    if not paths.run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {paths.run_dir}")
    if not paths.config.is_file():
        raise FileNotFoundError(f"Run config not found: {paths.config}")
    return paths


def resolve_model_selection(
    *,
    run: str | Path | None,
    model: str | Path | None,
    config: str | Path | None,
    model_kind: str = "final",
) -> ModelSelection:
    if (run is None) == (model is None):
        raise ValueError("Specify exactly one of --run or --model")
    if model_kind not in {"best", "final"}:
        raise ValueError("model_kind must be 'best' or 'final'")

    if run is not None:
        if config is not None:
            raise ValueError("--config is only valid with --model; --run uses <run>/config.yaml")
        paths = open_run(run)
        model_path = paths.models / f"{model_kind}_model.zip"
        if not model_path.is_file():
            raise FileNotFoundError(f"Run model not found: {model_path}")
        return ModelSelection(model=model_path, config=paths.config, eval_dir=paths.eval, run=paths)

    if config is None:
        raise ValueError("--model requires --config so the environment can be reconstructed")
    model_path = resolve_existing_file(model, field="Model")
    config_path = resolve_existing_file(config, field="Config")

    inferred_run: RunPaths | None = None
    if model_path.parent.name == "models":
        candidate = RunPaths.from_run_dir(model_path.parent.parent)
        if candidate.config.is_file():
            inferred_run = candidate
    eval_dir = inferred_run.eval if inferred_run is not None else model_path.parent / "eval"
    return ModelSelection(model=model_path, config=config_path, eval_dir=eval_dir, run=inferred_run)
