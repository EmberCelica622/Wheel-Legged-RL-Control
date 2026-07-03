from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces


def load_slide_config(config_path: str | Path) -> dict[str, Any]:
    """Load a slide-flat YAML config and attach a base directory for paths."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    cfg["_config_path"] = str(path)
    cfg["_base_dir"] = str(path.parent.parent if path.parent.name == "configs" else path.parent)
    return cfg


def _as_array(value: Any, length: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(length, float(arr), dtype=np.float64)
    if arr.shape != (length,):
        raise ValueError(f"{name} must be a scalar or length-{length} list, got {arr.shape}")
    return arr.astype(np.float64)


def _quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_roll_pitch_yaw(quat: np.ndarray) -> tuple[float, float, float]:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return 0.0, 0.0, 0.0
    w, x, y, z = q / norm

    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(sin_pitch, -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class SlideFlatEnv(gym.Env):
    """Flat-ground sliding task for a two-wheel-legged robot.

    The RL policy outputs a normalized 6D action. The environment maps that
    action into low-level PD targets, computes torques in Python, clips them,
    and writes torques to MuJoCo motor actuators.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict[str, Any], render_mode: str | None = None):
        super().__init__()

        self.cfg = copy.deepcopy(cfg)
        self.render_mode = render_mode

        base_dir = Path(self.cfg.get("_base_dir", ".")).expanduser().resolve()
        xml_path = Path(self.cfg["env"]["xml_path"]).expanduser()
        if not xml_path.is_absolute():
            xml_path = base_dir / xml_path
        self.xml_path = xml_path.resolve()

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        sim_cfg = self.cfg.get("sim", {})
        if "timestep" in sim_cfg:
            self.model.opt.timestep = float(sim_cfg["timestep"])
        self.control_decimation = int(sim_cfg.get("control_decimation", 10))
        self.max_episode_steps = int(sim_cfg.get("episode_length", 1000))
        self.reset_noise_scale = float(sim_cfg.get("reset_noise_scale", 0.0))

        robot_cfg = self.cfg.get("robot", {})
        self.base_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            robot_cfg.get("base_body", "base"),
        )

        # v2 加入轮子位置查询  +++++++++++++++++
        self.left_wheel_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            robot_cfg["left_wheel_body"],
        )
        self.right_wheel_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            robot_cfg["right_wheel_body"],
        )

        
        # v2  -----------------

        self.leg_joint_names = robot_cfg.get(
            "leg_joints",
            ["left_hip_pitch", "left_knee_pitch", "right_hip_pitch", "right_knee_pitch"],
        )
        self.wheel_joint_names = robot_cfg.get(
            "wheel_joints",
            ["left_wheel_joint", "right_wheel_joint"],
        )
        self.leg_actuator_names = robot_cfg.get(
            "leg_actuators",
            ["left_hip_motor", "left_knee_motor", "right_hip_motor", "right_knee_motor"],
        )
        self.wheel_actuator_names = robot_cfg.get(
            "wheel_actuators",
            ["left_wheel_motor", "right_wheel_motor"],
        )

        self.leg_joint_ids = self._required_ids(mujoco.mjtObj.mjOBJ_JOINT, self.leg_joint_names)
        self.wheel_joint_ids = self._required_ids(mujoco.mjtObj.mjOBJ_JOINT, self.wheel_joint_names)
        self.leg_actuator_ids = self._required_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, self.leg_actuator_names)
        self.wheel_actuator_ids = self._required_ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            self.wheel_actuator_names,
        )

        self.leg_qpos_addr = np.array([self.model.jnt_qposadr[jid] for jid in self.leg_joint_ids])
        self.leg_qvel_addr = np.array([self.model.jnt_dofadr[jid] for jid in self.leg_joint_ids])
        self.wheel_qvel_addr = np.array([self.model.jnt_dofadr[jid] for jid in self.wheel_joint_ids])

        self.leg_joint_range = self.model.jnt_range[self.leg_joint_ids].copy()
        self.clip_joint_targets = bool(self.cfg.get("control", {}).get("clip_joint_targets", True))

        self.default_qpos = self._load_default_qpos(robot_cfg.get("stand_keyframe", "stand"))
        self.default_qvel = np.zeros(self.model.nv, dtype=np.float64)
        self.default_leg_pos = self.default_qpos[self.leg_qpos_addr].copy()

        control_cfg = self.cfg.get("control", {})
        self.joint_action_scale = _as_array(control_cfg.get("joint_action_scale", 0.35), 4, "joint_action_scale")
        self.wheel_vel_scale = _as_array(control_cfg.get("wheel_vel_scale", 25.0), 2, "wheel_vel_scale")
        self.wheel_vel_bias = _as_array(control_cfg.get("wheel_vel_bias", 0.0), 2, "wheel_vel_bias")
        self.leg_kp = _as_array(control_cfg.get("leg_kp", 55.0), 4, "leg_kp")
        self.leg_kd = _as_array(control_cfg.get("leg_kd", 3.0), 4, "leg_kd")
        self.wheel_kp = _as_array(control_cfg.get("wheel_kp", 1.2), 2, "wheel_kp")
        self.leg_torque_limit = _as_array(control_cfg.get("leg_torque_limit", 60.0), 4, "leg_torque_limit")
        self.wheel_torque_limit = _as_array(
            control_cfg.get("wheel_torque_limit", 25.0),
            2,
            "wheel_torque_limit",
        )

        command_cfg = self.cfg.get("command", {})
        self.command = np.array(
            [
                float(command_cfg.get("forward_velocity", 0.8)),
                float(command_cfg.get("yaw_rate", 0.0)),
            ],
            dtype=np.float64,
        )

        self.reward_cfg = self.cfg.get("reward", {})
        self.reward_weights = self.reward_cfg.get("weights", {})
        self.termination_cfg = self.cfg.get("termination", {})

        # v2: stance regularization
        stance_cfg = self.reward_cfg.get("straight_stance", {})
        self.stance_regularization_enabled = bool(stance_cfg.get("enabled", False))
        self.stance_free_offset_m = float(
            stance_cfg.get("free_longitudinal_offset_m", 0.02)
        )
        self.stance_offset_scale_m = float(
            stance_cfg.get("longitudinal_offset_scale_m", 0.04)
        )
        self.stance_yaw_rate_threshold = float(
            stance_cfg.get("yaw_rate_threshold", 0.05)
        )
        # 加入 stance_offset_scale_m 的正数检查
        if self.stance_offset_scale_m <= 0.0:
            raise ValueError("longitudinal_offset_scale_m must be positive.")

        obs_clip = float(self.cfg.get("observation", {}).get("clip", 100.0))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # Observation v2 is 28D. PPO checkpoints trained with the previous
        # 25D observation schema are intentionally incompatible.
        self.observation_space = spaces.Box(low=-obs_clip, high=obs_clip, shape=(28,), dtype=np.float32)

        self.base_linvel_sensor = self._optional_sensor(robot_cfg.get("base_linear_velocity_sensor", "base_linear_velocity"))
        self.base_angvel_sensor = self._optional_sensor(robot_cfg.get("base_angular_velocity_sensor", "base_angular_velocity"))

        self.episode_step = 0
        self.prev_action = np.zeros(6, dtype=np.float64)
        self.last_action = np.zeros(6, dtype=np.float64)
        self.last_leg_torque = np.zeros(4, dtype=np.float64)
        self.last_wheel_torque = np.zeros(2, dtype=np.float64)
        self.last_q_des = self.default_leg_pos.copy()
        self.last_wheel_vel_des = np.zeros(2, dtype=np.float64)

    @property
    def control_dt(self) -> float:
        return float(self.model.opt.timestep * self.control_decimation)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.episode_step = 0
        self.prev_action[:] = 0.0
        self.last_action[:] = 0.0
        self.last_leg_torque[:] = 0.0
        self.last_wheel_torque[:] = 0.0
        self.last_q_des = self.default_leg_pos.copy()
        self.last_wheel_vel_des[:] = 0.0

        mujoco.mj_resetData(self.model, self.data)
        qpos = self.default_qpos.copy()
        qvel = self.default_qvel.copy()

        reset_noise_scale = self.reset_noise_scale
        if options and "reset_noise_scale" in options:
            reset_noise_scale = float(options["reset_noise_scale"])
        if reset_noise_scale > 0.0:
            qpos[self.leg_qpos_addr] += self.np_random.uniform(
                low=-reset_noise_scale,
                high=reset_noise_scale,
                size=4,
            )
            qvel[self.leg_qvel_addr] += self.np_random.uniform(
                low=-reset_noise_scale,
                high=reset_noise_scale,
                size=4,
            )

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.episode_step += 1

        old_action = self.prev_action.copy()
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # The policy target is held for one control interval, while torque is
        # recomputed from the latest joint state at every MuJoCo timestep.
        q_des, wheel_vel_des = self._action_to_targets(action)
        for _ in range(self.control_decimation):
            self._apply_pd_targets(q_des, wheel_vel_des)
            mujoco.mj_step(self.model, self.data)

        reward, reward_terms = self._compute_reward(action, old_action)
        terminated, termination_reason = self._is_terminated()
        truncated = (not terminated) and self.episode_step >= self.max_episode_steps

        self.prev_action = action.copy()
        self.last_action = action.copy()

        obs = self._get_obs()
        info = self._get_info()
        info["reward_terms"] = reward_terms
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        if termination_reason:
            info["termination_reason"] = termination_reason
        if truncated and not terminated:
            info["termination_reason"] = "timeout"

        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        return None

    def _required_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return int(obj_id)

    def _required_ids(self, obj_type: mujoco.mjtObj, names: list[str]) -> np.ndarray:
        return np.array([self._required_id(obj_type, name) for name in names], dtype=np.int32)

    def _optional_sensor(self, name: str | None) -> tuple[int, int, int] | None:
        if not name:
            return None
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            return None
        adr = int(self.model.sensor_adr[sensor_id])
        dim = int(self.model.sensor_dim[sensor_id])
        return int(sensor_id), adr, dim

    def _load_default_qpos(self, keyframe_name: str | None) -> np.ndarray:
        if keyframe_name:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
            if key_id >= 0:
                return self.model.key_qpos[key_id].copy()
        return self.model.qpos0.copy()

    def _sensor_data(self, sensor: tuple[int, int, int] | None) -> np.ndarray | None:
        if sensor is None:
            return None
        _, adr, dim = sensor
        return self.data.sensordata[adr : adr + dim].copy()

    def _base_quat_wxyz(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    def _base_rotmat(self) -> np.ndarray:
        return _quat_wxyz_to_rotmat(self._base_quat_wxyz())

    def _base_linear_velocity_body(self) -> np.ndarray:
        sensor_vel = self._sensor_data(self.base_linvel_sensor)
        if sensor_vel is not None and sensor_vel.shape[0] == 3:
            return sensor_vel
        return self._base_rotmat().T @ self.data.qvel[0:3]

    def _base_angular_velocity_body(self) -> np.ndarray:
        sensor_vel = self._sensor_data(self.base_angvel_sensor)
        if sensor_vel is not None and sensor_vel.shape[0] == 3:
            return sensor_vel
        return self._base_rotmat().T @ self.data.qvel[3:6]

    def _projected_gravity(self) -> np.ndarray:
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return self._base_rotmat().T @ world_gravity

    def _base_height(self) -> float:
        return float(self.data.xpos[self.base_body_id, 2])

    def _leg_pos(self) -> np.ndarray:
        return self.data.qpos[self.leg_qpos_addr].copy()

    def _leg_vel(self) -> np.ndarray:
        return self.data.qvel[self.leg_qvel_addr].copy()

    def _wheel_vel(self) -> np.ndarray:
        return self.data.qvel[self.wheel_qvel_addr].copy()
    
    # v2
    def _wheel_longitudinal_offset(self) -> float:
        """Signed left-right wheel offset along the base-frame forward axis."""
        p_left_world = self.data.xpos[self.left_wheel_body_id]
        p_right_world = self.data.xpos[self.right_wheel_body_id]

        delta_world = p_left_world - p_right_world
        delta_body = self._base_rotmat().T @ delta_world

        # body x-axis is the same forward direction used by vx command tracking.
        return float(delta_body[0])

    def _action_to_targets(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Policy action[0:4] is a normalized offset around the nominal stand
        # pose. The target remains fixed for the full control interval.
        q_des = self.default_leg_pos + action[:4] * self.joint_action_scale
        if self.clip_joint_targets:
            q_des = np.clip(q_des, self.leg_joint_range[:, 0], self.leg_joint_range[:, 1])

        # Policy action[4:6] is a normalized wheel angular velocity command.
        wheel_vel_des = self.wheel_vel_bias + action[4:6] * self.wheel_vel_scale

        self.last_q_des = q_des.copy()
        self.last_wheel_vel_des = wheel_vel_des.copy()
        return q_des, wheel_vel_des

    def _apply_pd_targets(self, q_des: np.ndarray, wheel_vel_des: np.ndarray) -> None:
        """Update torque PD from current state at the MuJoCo physics rate."""
        leg_pos = self._leg_pos()
        leg_vel = self._leg_vel()
        wheel_vel = self._wheel_vel()

        tau_leg = self.leg_kp * (q_des - leg_pos) - self.leg_kd * leg_vel
        tau_leg = np.clip(tau_leg, -self.leg_torque_limit, self.leg_torque_limit)

        # Wheel velocity error is converted to motor torque each physics step.
        tau_wheel = self.wheel_kp * (wheel_vel_des - wheel_vel)
        tau_wheel = np.clip(tau_wheel, -self.wheel_torque_limit, self.wheel_torque_limit)

        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.leg_actuator_ids] = tau_leg
        self.data.ctrl[self.wheel_actuator_ids] = tau_wheel

        self.last_leg_torque = tau_leg.copy()
        self.last_wheel_torque = tau_wheel.copy()

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate(
            [
                self._base_linear_velocity_body(),
                self._base_angular_velocity_body(),
                self._projected_gravity(),
                self.command,
                self._leg_pos() - self.default_leg_pos,
                self._leg_vel(),
                self._wheel_vel(),
                np.array([self._base_height()], dtype=np.float64),
                self.prev_action,
            ]
        )
        obs_clip = float(self.cfg.get("observation", {}).get("clip", 100.0))
        obs = np.clip(obs, -obs_clip, obs_clip)
        return obs.astype(np.float32)

    def _compute_reward(self, action: np.ndarray, old_action: np.ndarray) -> tuple[float, dict[str, float]]:
        base_lin_vel = self._base_linear_velocity_body()
        base_ang_vel = self._base_angular_velocity_body()
        projected_gravity = self._projected_gravity()

        tracking_sigma = float(self.reward_cfg.get("tracking_sigma", 0.35))
        yaw_sigma = float(self.reward_cfg.get("yaw_sigma", 0.5))
        upright_sigma = float(self.reward_cfg.get("upright_sigma", 0.25))
        target_base_height = float(self.reward_cfg.get("target_base_height", self.default_qpos[2]))

        forward_error = base_lin_vel[0] - self.command[0]
        yaw_error = base_ang_vel[2] - self.command[1]
        tilt_error = projected_gravity[0] ** 2 + projected_gravity[1] ** 2
        height_error = self._base_height() - target_base_height

        all_torque = np.concatenate(
            [
                self.last_leg_torque / np.maximum(self.leg_torque_limit, 1e-6),
                self.last_wheel_torque / np.maximum(self.wheel_torque_limit, 1e-6),
            ]
        )

        wheel_longitudinal_offset = self._wheel_longitudinal_offset()

        straight_gate = float(
            self.stance_regularization_enabled
            and abs(self.command[1]) <= self.stance_yaw_rate_threshold
        )

        offset_excess_m = max(
            abs(wheel_longitudinal_offset) - self.stance_free_offset_m,
            0.0,
        )

        wheel_longitudinal_offset_penalty = straight_gate * (
            offset_excess_m / self.stance_offset_scale_m
        ) ** 2

        terms = {
            "forward_velocity_tracking": math.exp(-(forward_error**2) / max(tracking_sigma**2, 1e-6)),
            "yaw_rate_tracking": math.exp(-(yaw_error**2) / max(yaw_sigma**2, 1e-6)),
            "upright": math.exp(-tilt_error / max(upright_sigma, 1e-6)),
            "base_height_penalty": height_error**2,
            "action_penalty": float(np.mean(action**2)),
            "action_rate_penalty": float(np.mean((action - old_action) ** 2)),
            "joint_velocity_penalty": float(np.mean(self._leg_vel() ** 2)),
            "torque_penalty": float(np.mean(all_torque**2)),
            "wheel_longitudinal_offset_abs_m": abs(wheel_longitudinal_offset),
            "wheel_longitudinal_offset_excess_m": offset_excess_m,
            "wheel_longitudinal_offset_penalty": wheel_longitudinal_offset_penalty,
            "straight_stance_gate": straight_gate,
        }

        weights = self.reward_weights
        reward = (
            float(weights.get("forward_velocity_tracking", 1.5)) * terms["forward_velocity_tracking"]
            + float(weights.get("yaw_rate_tracking", 0.4)) * terms["yaw_rate_tracking"]
            + float(weights.get("upright", 0.5)) * terms["upright"]
            - float(weights.get("base_height_penalty", 2.0)) * terms["base_height_penalty"]
            - float(weights.get("action_penalty", 0.01)) * terms["action_penalty"]
            - float(weights.get("action_rate_penalty", 0.03)) * terms["action_rate_penalty"]
            - float(weights.get("joint_velocity_penalty", 0.001)) * terms["joint_velocity_penalty"]
            - float(weights.get("torque_penalty", 0.002)) * terms["torque_penalty"]
            - float(weights.get("wheel_longitudinal_offset_penalty", 0.0)) * terms["wheel_longitudinal_offset_penalty"]
        )

        return float(reward), {key: float(value) for key, value in terms.items()}

    def _is_terminated(self) -> tuple[bool, str | None]:
        min_base_height = float(self.termination_cfg.get("min_base_height", 0.25))
        max_roll = float(self.termination_cfg.get("max_roll", 0.75))
        max_pitch = float(self.termination_cfg.get("max_pitch", 0.75))

        if self._base_height() < min_base_height:
            return True, "base_height_too_low"

        roll, pitch, _ = _quat_wxyz_to_roll_pitch_yaw(self._base_quat_wxyz())
        if abs(roll) > max_roll:
            return True, "roll_too_large"
        if abs(pitch) > max_pitch:
            return True, "pitch_too_large"

        return False, None

    def _get_info(self) -> dict[str, Any]:
        roll, pitch, yaw = _quat_wxyz_to_roll_pitch_yaw(self._base_quat_wxyz())
        return {
            "episode_step": self.episode_step,
            "time": float(self.data.time),
            "base_height": self._base_height(),
            "base_forward_velocity": float(self._base_linear_velocity_body()[0]),
            "velocity_error": float(self._base_linear_velocity_body()[0] - self.command[0]),
            "base_yaw_rate": float(self._base_angular_velocity_body()[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "q_des": self.last_q_des.copy(),
            "wheel_vel_des": self.last_wheel_vel_des.copy(),
            "tau_leg": self.last_leg_torque.copy(),
            "tau_wheel": self.last_wheel_torque.copy(),
            "mean_leg_joint_velocity": float(np.mean(np.abs(self._leg_vel()))),
            "wheel_longitudinal_offset": self._wheel_longitudinal_offset(),
        }
