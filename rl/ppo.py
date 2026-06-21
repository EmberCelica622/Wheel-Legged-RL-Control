from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO


PPO_KWARGS = {
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "clip_range_vf",
    "normalize_advantage",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "use_sde",
    "sde_sample_freq",
    "target_kl",
    "stats_window_size",
    "tensorboard_log",
    "policy_kwargs",
    "verbose",
    "seed",
    "device",
}


def _resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(cfg.get("_base_dir", ".")).expanduser().resolve() / path


def _ppo_kwargs_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    ppo_cfg = dict(cfg.get("ppo", {}))
    kwargs = {key: value for key, value in ppo_cfg.items() if key in PPO_KWARGS}

    if "tensorboard_log" in kwargs:
        kwargs["tensorboard_log"] = str(_resolve_path(cfg, kwargs["tensorboard_log"]))
    else:
        log_dir = cfg.get("logging", {}).get("tensorboard_log_dir")
        if log_dir:
            kwargs["tensorboard_log"] = str(_resolve_path(cfg, log_dir))

    return kwargs


def create_ppo_model(env: Any, cfg: dict[str, Any]) -> PPO:
    """Create a Stable-Baselines3 PPO model from the YAML config."""
    ppo_cfg = cfg.get("ppo", {})
    policy = ppo_cfg.get("policy", "MlpPolicy")
    kwargs = _ppo_kwargs_from_cfg(cfg)
    return PPO(policy, env, **kwargs)


def load_ppo_model(path: str | Path, env: Any | None = None, cfg: dict[str, Any] | None = None) -> PPO:
    """Load a Stable-Baselines3 PPO checkpoint."""
    load_kwargs: dict[str, Any] = {}
    if env is not None:
        load_kwargs["env"] = env
    if cfg is not None:
        device = cfg.get("ppo", {}).get("device")
        if device:
            load_kwargs["device"] = device
    return PPO.load(str(path), **load_kwargs)


class PPOTrainer:
    """Small wrapper around Stable-Baselines3 PPO for the slide task."""

    def __init__(self, env: Any, cfg: dict[str, Any], model: PPO | None = None):
        self.env = env
        self.cfg = cfg
        self.model = model if model is not None else create_ppo_model(env, cfg)

    @classmethod
    def load(cls, checkpoint: str | Path, env: Any, cfg: dict[str, Any]) -> "PPOTrainer":
        model = load_ppo_model(checkpoint, env=env, cfg=cfg)
        return cls(env=env, cfg=cfg, model=model)

    def learn(
        self,
        total_timesteps: int | None = None,
        callback: Any | None = None,
        tb_log_name: str = "PPO",
    ) -> PPO:
        if total_timesteps is None:
            total_timesteps = int(self.cfg.get("training", {}).get("total_timesteps", 1000000))
        self.model.learn(
            total_timesteps=int(total_timesteps),
            callback=callback,
            tb_log_name=tb_log_name,
        )
        return self.model

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
