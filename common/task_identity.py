from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


_TASK_COMPONENT = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_TASK_VERSION = re.compile(r"^v[0-9]+$")


@dataclass(frozen=True)
class TaskIdentity:
    capability: str
    condition: str
    complexity: str
    version: str
    legacy: bool = False

    @classmethod
    def from_mapping(cls, value: Any) -> "TaskIdentity":
        if not isinstance(value, dict):
            raise ValueError("task must be a mapping")
        missing = [
            key
            for key in ("capability", "condition", "complexity", "version")
            if key not in value
        ]
        if missing:
            raise ValueError(f"task is missing required field(s): {', '.join(missing)}")
        identity = cls(
            capability=str(value["capability"]).strip().lower(),
            condition=str(value["condition"]).strip().lower(),
            complexity=str(value["complexity"]).strip().lower(),
            version=str(value["version"]).strip().lower(),
            legacy=bool(value.get("legacy", False)),
        )
        identity.validate()
        return identity

    def validate(self) -> None:
        for field, value in (
            ("task.capability", self.capability),
            ("task.condition", self.condition),
            ("task.complexity", self.complexity),
        ):
            if not _TASK_COMPONENT.fullmatch(value):
                raise ValueError(f"{field} must be a lowercase task component, got {value!r}")
        if not _TASK_VERSION.fullmatch(self.version):
            raise ValueError(f"task.version must look like 'v1', got {self.version!r}")

    @property
    def canonical_id(self) -> str:
        base = "_".join((self.capability, self.condition, self.complexity, self.version))
        return f"{base}_legacy" if self.legacy else base

    def as_mapping(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "condition": self.condition,
            "complexity": self.complexity,
            "version": self.version,
            "legacy": self.legacy,
        }


def canonical_task_id_from_config(cfg: dict[str, Any]) -> str:
    return task_identity_from_config(cfg).canonical_id


def normalize_task_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a config whose task block is the only task identity source."""
    normalized = copy.deepcopy(cfg)
    identity = task_identity_from_config(normalized)
    env_version = _env_version_from_config(normalized)
    if env_version is not None and env_version != identity.version:
        raise ValueError(
            f"env.version {env_version!r} conflicts with task.version {identity.version!r}"
        )
    normalized["task"] = identity.as_mapping()

    experiment = normalized.get("experiment")
    if isinstance(experiment, dict):
        experiment.pop("task", None)
        experiment.pop("env_variant", None)

    env_cfg = normalized.get("env")
    if isinstance(env_cfg, dict):
        env_cfg["version"] = identity.version

    normalized["_task_id"] = identity.canonical_id
    return normalized


def task_identity_from_config(cfg: dict[str, Any]) -> TaskIdentity:
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    if "task" in cfg:
        return TaskIdentity.from_mapping(cfg["task"])
    return _infer_deprecated_task_identity(cfg)


def _infer_deprecated_task_identity(cfg: dict[str, Any]) -> TaskIdentity:
    # DEPRECATED_TASK_COMPATIBILITY: old run snapshots used experiment.task plus
    # experiment.env_variant or env.version. Keep loading them without rewriting.
    experiment = cfg.get("experiment", {})
    env_cfg = cfg.get("env", {})
    if not isinstance(experiment, dict):
        raise ValueError("experiment must be a mapping")
    if not isinstance(env_cfg, dict):
        raise ValueError("env must be a mapping")
    if not any(key in experiment for key in ("task", "env_variant")) and "version" not in env_cfg:
        raise ValueError("Missing required task metadata")

    legacy_task = str(experiment.get("task", "slide_flat")).strip().lower()
    if legacy_task != "slide_flat":
        raise ValueError(
            "Missing task metadata and unsupported deprecated experiment.task "
            f"{legacy_task!r}"
        )

    legacy_variant = experiment.get("env_variant", env_cfg.get("version", "v1"))
    variant = str(legacy_variant).strip().lower()
    command_cfg = cfg.get("command", {})
    forward_cfg = command_cfg.get("forward_velocity") if isinstance(command_cfg, dict) else None

    if variant == "v2" and not _is_uniform_per_episode_forward_command(forward_cfg):
        variant = "v1"

    if variant == "v2":
        return TaskIdentity("slide", "variable_velocity", "flat", "v2", False)
    if variant == "v3":
        return TaskIdentity("slide", "dynamic_command", "flat", "v3", False)
    if variant == "v1":
        return TaskIdentity("slide", "fixed_velocity", "flat", "v1", _has_legacy_marker(cfg))

    raise ValueError(
        "Missing task metadata and unsupported deprecated experiment.env_variant "
        f"{variant!r}"
    )


def _is_uniform_per_episode_forward_command(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and str(value.get("mode", "")).strip().lower() == "uniform_per_episode"
    )


def _env_version_from_config(cfg: dict[str, Any]) -> str | None:
    env_cfg = cfg.get("env", {})
    if not isinstance(env_cfg, dict) or "version" not in env_cfg:
        return None
    version = str(env_cfg["version"]).strip().lower()
    if not _TASK_VERSION.fullmatch(version):
        raise ValueError(f"env.version must look like 'v1', got {version!r}")
    return version


def _has_legacy_marker(cfg: dict[str, Any]) -> bool:
    config_path = str(cfg.get("_config_path", "")).lower()
    experiment = cfg.get("experiment", {})
    experiment_name = ""
    if isinstance(experiment, dict):
        experiment_name = str(experiment.get("name", "")).lower()
    return "legacy" in config_path or "legacy" in experiment_name
