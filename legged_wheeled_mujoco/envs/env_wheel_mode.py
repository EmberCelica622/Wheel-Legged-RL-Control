import os
from typing import Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import utils, spaces
from numpy.typing import NDArray
from numpy.typing import NDArray
from gymnasium.envs.mujoco import mujoco_env
from scipy.spatial.transform import Rotation


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class RollingEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    MuJoCo + Gymnasium version of the original RollingEnv.

    Core idea:
    - Randomly sample a target position.
    - Randomly sample robot initial position and yaw.
    - Robot moves by rolling wheels.
    - Reward is mainly based on distance reduction to the target.
    - Episode terminates if the robot falls, bumps base/thigh, or reaches target.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(
        self,
        xml_file=None,
        frame_skip=5,
        ctrl_cost_weight=0.0001,
        healthy_reward=0.1,
        healthy_z_range=0.05,
        reset_noise_scale=0.1,
        max_episode_steps=2000,
        render_mode=None,
        width=480,
        height=480,
    ):
        if xml_file is None:
            xml_file = os.path.join(
                os.path.dirname(__file__),
                "asset",
                "wheel_model.xml",
            )

        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            ctrl_cost_weight,
            healthy_reward,
            healthy_z_range,
            reset_noise_scale,
            max_episode_steps,
            render_mode,
            width,
            height,
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self.max_episode_steps = max_episode_steps

        self.c_step = 0
        self.re = 0.1

        # self.obs keeps the unnormalized raw observation.
        self.obs = None

        # New Gymnasium MujocoEnv signature requires observation_space.
        # We can pass None first, then define it after model/data are loaded.
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path=xml_file,
            frame_skip=frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            width=width,
            height=height,
        )

        # Build observation space after self.model and self.data exist.
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float64,
        )

    @property
    def healthy_reward(self):
        """
        Give standing reward only during the early part of the episode.
        This follows your original logic:
        if c_step < 500, reward health; otherwise no health reward.
        """
        if self.c_step < 500:
            return float(self.is_healthy) * self._healthy_reward
        return 0.0

    def control_cost(self, action):
        """
        Penalize control effort.

        Your old code ignores action index 2 and 5 because they are later used
        as transformed wheel commands.
        """
        action = np.asarray(action)
        used_indices = [i for i in range(len(action)) if i not in (2, 5)]
        return self._ctrl_cost_weight * np.sum(np.abs(action[used_indices]))

    @property
    def is_healthy(self):
        """
        Whether the robot is still considered valid/alive.

        In your original code:
        - base_link z must be higher than a threshold
        - base/thigh should not touch the ground
        """
        base_z = self.get_body_com("base_link")[2]
        return (base_z > self._healthy_z_range) and (not self.bump_base())

    @property
    def terminated(self):
        """
        True termination: robot physically failed.
        """
        return not self.is_healthy

    def bump_base(self):
        """
        Check whether base or thigh geoms have contact.

        Old API:
            self.sim.data.contact[i]
            self.sim.model.geom_id2name(...)

        New API:
            self.data.contact[i]
            self.model.geom(geom_id).name
        """
        protected_geoms = {
            "base1", "base2", "base3", "base4",
            "left_thigh1", "left_thigh2", "left_thigh3",
            "right_thigh1", "right_thigh2", "right_thigh3",
        }

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            geom1_name = self.model.geom(int(contact.geom1)).name
            geom2_name = self.model.geom(int(contact.geom2)).name

            if geom1_name in protected_geoms or geom2_name in protected_geoms:
                return True

        return False

    def get_xydistance(self) -> float:
        """
        Distance between target xy and robot xy.

        self.obs is the raw, unnormalized observation.
        Based on your original observation layout:
            obs[0:2] = target xy
            obs[2:4] = robot xy
        """
        if self.obs is None:
            self._get_obs()
        assert self.obs is not None

        target_pos = self.obs[0:2]
        robot_pos = self.obs[2:4]
        xy_diff = target_pos - robot_pos
        return float(np.linalg.norm(xy_diff))

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
        """
        One RL step.

        New Gymnasium API return:
            obs, reward, terminated, truncated, info

        terminated:
            task ended naturally, e.g. success or robot fell.

        truncated:
            episode stopped due to time limit.
        """
        info = {}
        self.c_step += 1

        action = np.asarray(action, dtype=np.float32).copy()

        # Observe before action.
        self._get_obs()
        d_before = self.get_xydistance()

        # Your original action remapping:
        # t = action[2], r = action[5]
        # action[2] = (t - r * 0.5) / 1.5
        # action[5] = (t + r * 0.5) / 1.5
        #
        # Important: use .copy() above so we do not mutate the policy's action array.
        if action.shape[0] >= 6:
            t = action[2]
            r = action[5]
            action[2] = (t - r * 0.5) / 1.5
            action[5] = (t + r * 0.5) / 1.5

        # This still works in Gymnasium's MujocoEnv.
        # Internally it writes self.data.ctrl[:] = action and calls mujoco.mj_step().
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        d_after = self.get_xydistance()

        approach_distance = d_before - d_after
        approaching_reward = 100.0 * approach_distance

        punishment = 0.0

        terminated = False
        truncated = False

        reward = self.re * (
            approaching_reward
            + self.healthy_reward
            - punishment
        )

        # Failure termination.
        if not self.is_healthy:
            terminated = True
            reward -= 10.0
            info["is_success"] = False

        # Success termination.
        elif d_after < 1.0:
            terminated = True
            reward = self.re * 100.0
            info["is_success"] = True

        # Time-limit truncation.
        elif self.c_step >= self.max_episode_steps:
            truncated = True
            info["is_success"] = False

        if self.render_mode == "human":
            self.render()

        reward = np.float64(reward)

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Observation layout from your original code:

        raw observation:
            0-1    target x,y
            2-4    base pos x,y,z
            5-8    base ori quat
            9/11   left/right hip joint pos
            10/12  left/right knee joint pos
            13/14  left/right wheel velocity
            15-17  base local linear velocity
            18-20  base local linear acceleration
            21-23  base local angular velocity

        New API:
            self.data.sensordata
            self.data.qpos
            self.data.qvel
        """
        raw_obs = np.array(self.data.sensordata, dtype=np.float64).copy()

        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # Preserve your old behavior:
        # replace first 8 sensor entries with qpos[:8].
        # This assumes your XML sensor layout and qpos layout match the old code.
        n = min(8, len(raw_obs), len(qpos))
        raw_obs[:n] = qpos[:n]

        self.obs = raw_obs.copy()

        obs = raw_obs.copy()

        # Normalization from your original code.
        if len(obs) >= 4:
            obs[:2] /= 10.0       # target xy
            obs[2:4] /= 10.0      # robot xy

        if len(obs) >= 15:
            obs[13:15] /= 30.0    # wheel velocity

        return obs

    def reset_model(self):
        """
        Reset MuJoCo state.

        Important fix:
        Your old code had:
            qpos = self.init_qpos
            qvel = self.init_qvel

        That mutates init_qpos/init_qvel in-place.
        New version should use .copy().
        """
        self.c_step = 0

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        rng = self.np_random

        random_pos = True

        # Reset target.
        qpos[0] = rng.uniform(-10.0, 10.0)
        qpos[1] = rng.uniform(-10.0, 10.0)

        # Reset initial robot position.
        qpos[2] = rng.uniform(-10.0, 10.0) if random_pos else 0.0
        qpos[3] = rng.uniform(-10.0, 10.0) if random_pos else 0.0

        # Reset robot yaw.
        #
        # MuJoCo free joint quaternion order is usually:
        #     w, x, y, z
        #
        # scipy Rotation.as_quat() returns:
        #     x, y, z, w
        #
        # So we convert xyzw -> wxyz before assigning into qpos.
        yaw = rng.uniform(0.0, 2.0 * np.pi)
        quat_xyzw = Rotation.from_euler("z", yaw).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )

        if len(qpos) >= 9:
            qpos[5:9] = quat_wxyz

        # Ensure target and robot initial position are not too close.
        while np.linalg.norm(qpos[0:2] - qpos[2:4]) < 1.0:
            qpos[0] = rng.uniform(-10.0, 10.0)
            qpos[1] = rng.uniform(-10.0, 10.0)

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_reset_info(self):
        """
        Optional reset info returned by Gymnasium reset().
        """
        if self.obs is None:
            return {}
    
        return {
            "distance_to_target": self.get_xydistance()
        }


if __name__ == "__main__":
    # Direct construction.
    env = RollingEnv(render_mode="human")

    obs, info = env.reset(seed=0)
    print("Initial obs:", obs)
    print("Reset info:", info)

    terminated = False
    truncated = False

    # while not (terminated or truncated):
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # print(reward, terminated, truncated, info)

    env.close()