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
    "policy_kwargs",
    "verbose",
    "device",
}


def validate_ppo_rollout_config(cfg: dict[str, Any], n_envs: int) -> None:
    """Validate that PPO minibatches partition the vector rollout exactly."""
    ppo_cfg = cfg.get("ppo", {})
    n_steps = int(ppo_cfg.get("n_steps", 2048))
    batch_size = int(ppo_cfg.get("batch_size", 64))
    n_envs = int(n_envs)
    if n_steps < 1 or batch_size < 1 or n_envs < 1:
        raise ValueError("ppo.n_steps, ppo.batch_size, and training.n_envs must be positive")

    rollout_size = n_steps * n_envs
    if batch_size > rollout_size:
        raise ValueError(
            f"ppo.batch_size ({batch_size}) exceeds rollout size "
            f"ppo.n_steps * n_envs ({n_steps} * {n_envs} = {rollout_size})"
        )
    if rollout_size % batch_size != 0:
        raise ValueError(
            f"PPO rollout size {rollout_size} must be divisible by batch_size {batch_size}"
        )


def _ppo_kwargs_from_cfg(
    cfg: dict[str, Any],
    tensorboard_log: str | Path | None = None,
) -> dict[str, Any]:
    ppo_cfg = dict(cfg.get("ppo", {}))
    kwargs = {key: value for key, value in ppo_cfg.items() if key in PPO_KWARGS}
    kwargs["seed"] = int(cfg.get("seed", 1))
    if tensorboard_log is not None:
        kwargs["tensorboard_log"] = str(Path(tensorboard_log).expanduser().resolve())
    return kwargs


def create_ppo_model(
    env: Any,
    cfg: dict[str, Any],
    tensorboard_log: str | Path | None = None,
) -> PPO:
    """Create a Stable-Baselines3 PPO model from the YAML config."""
    validate_ppo_rollout_config(cfg, int(getattr(env, "num_envs", 1)))
    policy = cfg.get("ppo", {}).get("policy", "MlpPolicy")
    kwargs = _ppo_kwargs_from_cfg(cfg, tensorboard_log=tensorboard_log)
    return PPO(policy, env, **kwargs)


def load_ppo_model(
    path: str | Path,
    env: Any | None = None,
    cfg: dict[str, Any] | None = None,
    tensorboard_log: str | Path | None = None,
    override_ppo_config: bool = False,
) -> PPO:
    """Load a Stable-Baselines3 PPO checkpoint."""
    load_kwargs: dict[str, Any] = {}
    if env is not None:
        load_kwargs["env"] = env
    if cfg is not None:
        device = cfg.get("ppo", {}).get("device")
        if device:
            load_kwargs["device"] = device
        if override_ppo_config:
            n_envs = int(getattr(env, "num_envs", 1))
            validate_ppo_rollout_config(cfg, n_envs)
            ppo_cfg = cfg.get("ppo", {})
            load_kwargs["n_steps"] = int(ppo_cfg.get("n_steps", 2048))
            load_kwargs["batch_size"] = int(ppo_cfg.get("batch_size", 64))
            load_kwargs["seed"] = int(cfg.get("seed", 1))
    if tensorboard_log is not None:
        load_kwargs["tensorboard_log"] = str(Path(tensorboard_log).expanduser().resolve())
    return PPO.load(str(path), **load_kwargs)


class PPOTrainer:
    """Small wrapper around Stable-Baselines3 PPO for the slide task."""

    def __init__(
        self,
        env: Any,
        cfg: dict[str, Any],
        model: PPO | None = None,
        tensorboard_log: str | Path | None = None,
    ):
        self.env = env
        self.cfg = cfg
        self.model = model if model is not None else create_ppo_model(
            env,
            cfg,
            tensorboard_log=tensorboard_log,
        )

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        env: Any,
        cfg: dict[str, Any],
        tensorboard_log: str | Path | None = None,
        override_ppo_config: bool = False,
    ) -> "PPOTrainer":
        model = load_ppo_model(
            checkpoint,
            env=env,
            cfg=cfg,
            tensorboard_log=tensorboard_log,
            override_ppo_config=override_ppo_config,
        )
        return cls(env=env, cfg=cfg, model=model, tensorboard_log=tensorboard_log)

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
