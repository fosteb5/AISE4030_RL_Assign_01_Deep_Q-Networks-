"""
Mario environment setup and wrappers for AISE 4030 Assignment 01.
"""

from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import gym
import gym_super_mario_bros
import numpy as np
from gym import spaces
from nes_py.wrappers import JoypadSpace

RIGHT_ONLY = [
    ["right"],
    ["right", "A"],
]


def _reset_compat(env: gym.Env, **kwargs) -> Tuple[np.ndarray, dict]:
    """
    Resets an environment across old and new Gym-style APIs.
    """
    try:
        result = env.reset(**kwargs)
    except TypeError:
        seed = kwargs.get("seed", None)
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()

    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return obs, info

    return result, {}


def _step_compat(env: gym.Env, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """
    Steps an environment across old and new Gym-style APIs.
    """
    result = env.step(action)

    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated, truncated, info

    obs, reward, done, info = result
    return obs, reward, bool(done), False, info


class MarioRewardWrapper(gym.Wrapper):
    """
    Shapes the Mario reward using x-position progress, a time penalty,
    and a death penalty, then clips the result to [-15, 15].
    """

    def __init__(
        self,
        env: gym.Env,
        progress_scale: float = 0.03,
        time_penalty: float = -0.005,
        flag_reward: float = 50.0,
        death_penalty: float = -15.0,
    ) -> None:
        """
        Initializes the reward wrapper.

        Args:
            env (gym.Env): The base environment.
            progress_scale (float): Reward multiplier for forward x-position progress.
            time_penalty (float): Small penalty applied every step to encourage speed.
            flag_reward (float): Bonus reward applied when Mario reaches the flag.
            death_penalty (float): Penalty applied when Mario dies.
        """
        super().__init__(env)
        self.progress_scale = progress_scale
        self.time_penalty = time_penalty
        self.flag_reward = flag_reward
        self.death_penalty = death_penalty
        self.prev_x = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment and internal reward state.

        Returns:
            Tuple[np.ndarray, dict]: Reset observation and info dictionary.
        """
        obs, info = _reset_compat(self.env, **kwargs)
        self.prev_x = int(info.get("x_pos", 0))
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Steps the environment and computes shaped reward.

        Args:
            action (int): Selected action.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]:
                Observation, shaped reward, terminated, truncated, info.
        """
        obs, _, terminated, truncated, info = _step_compat(self.env, action)

        current_x = int(info.get("x_pos", self.prev_x))
        delta_x = current_x - self.prev_x
        self.prev_x = current_x

        progress_reward = self.progress_scale * float(delta_x)
        reward = max(progress_reward, 0.0) if delta_x > 0 else progress_reward
        reward += self.time_penalty

        died = terminated and not bool(info.get("flag_get", False))
        if died:
            reward += self.death_penalty

        reward = float(np.clip(reward, -15.0, 15.0))

        if bool(info.get("flag_get", False)):
            reward += self.flag_reward
        return obs, reward, terminated, truncated, info


class StagnationTerminationWrapper(gym.Wrapper):
    """
    Truncates an episode if Mario fails to make forward progress for too long.
    """

    def __init__(
        self,
        env: gym.Env,
        max_stagnation_steps: int = 150,
        stagnation_penalty: float = -10.0,
    ) -> None:
        """
        Initializes the stagnation termination wrapper.

        Args:
            env (gym.Env): The base environment.
            max_stagnation_steps (int): Max consecutive agent steps without forward progress.
            stagnation_penalty (float): Penalty applied when truncating for stagnation.
        """
        super().__init__(env)
        self.max_stagnation_steps = max_stagnation_steps
        self.stagnation_penalty = stagnation_penalty
        self.prev_x = 0
        self.stagnation_steps = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment and stagnation tracking state.

        Returns:
            Tuple[np.ndarray, dict]: Reset observation and info dictionary.
        """
        obs, info = _reset_compat(self.env, **kwargs)
        self.prev_x = int(info.get("x_pos", 0))
        self.stagnation_steps = 0
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Steps the environment and truncates if Mario stagnates for too long.

        Args:
            action (int): Selected action.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]:
                Observation, reward, terminated, truncated, info.
        """
        obs, reward, terminated, truncated, info = _step_compat(self.env, action)

        current_x = int(info.get("x_pos", self.prev_x))
        if current_x > self.prev_x:
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1
        self.prev_x = current_x

        if not terminated and not truncated and self.stagnation_steps >= self.max_stagnation_steps:
            truncated = True
            reward = float(reward) + self.stagnation_penalty
            info = dict(info)
            info["stagnation_terminated"] = True

        return obs, float(reward), terminated, truncated, info


class SkipFrame(gym.Wrapper):
    """
    Repeats the same action for a fixed number of frames and sums rewards.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        """
        Initializes the frame skip wrapper.

        Args:
            env (gym.Env): The base environment.
            skip (int): Number of repeated frames per action.
        """
        super().__init__(env)
        self.skip = skip

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Repeats the action and sums rewards.

        Args:
            action (int): Selected action.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]:
                Observation, total reward, terminated, truncated, info.
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = _step_compat(self.env, action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """
    Converts RGB observations to grayscale.
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Initializes the grayscale wrapper.

        Args:
            env (gym.Env): The base environment.
        """
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Converts an RGB frame to grayscale.

        Args:
            observation (np.ndarray): RGB observation.

        Returns:
            np.ndarray: Grayscale observation.
        """
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return gray.astype(np.uint8)


class ResizeObservation(gym.ObservationWrapper):
    """
    Resizes a grayscale observation to a target square size and normalizes it.
    """

    def __init__(self, env: gym.Env, shape: int = 84) -> None:
        """
        Initializes the resize wrapper.

        Args:
            env (gym.Env): The base environment.
            shape (int): Target height and width.
        """
        super().__init__(env)
        self.shape = shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(shape, shape),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Resizes and normalizes the observation.

        Args:
            observation (np.ndarray): Grayscale observation.

        Returns:
            np.ndarray: Resized normalized observation.
        """
        resized = cv2.resize(observation, (self.shape, self.shape), interpolation=cv2.INTER_AREA)
        resized = resized.astype(np.float32) / 255.0
        return resized


class FrameStackObservation(gym.Wrapper):
    """
    Stacks the most recent N observations along a new first dimension.
    """

    def __init__(self, env: gym.Env, num_stack: int = 4) -> None:
        """
        Initializes the frame stack wrapper.

        Args:
            env (gym.Env): The base environment.
            num_stack (int): Number of frames to stack.
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.frames: Deque[np.ndarray] = deque(maxlen=num_stack)
        self.stacked_obs: Optional[np.ndarray] = None

        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_stack, *obs_shape),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """
        Builds the stacked observation.

        Returns:
            np.ndarray: Stacked frame tensor with shape (num_stack, H, W).
        """
        if self.stacked_obs is None:
            self.stacked_obs = np.stack(list(self.frames), axis=0).astype(np.float32)
        return self.stacked_obs

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment and fills the stack with the first frame.

        Returns:
            Tuple[np.ndarray, dict]: Stacked observation and info dictionary.
        """
        obs, info = _reset_compat(self.env, **kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        self.stacked_obs = np.repeat(
            np.asarray(obs, dtype=np.float32)[np.newaxis, ...],
            self.num_stack,
            axis=0,
        )
        return self._get_observation(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Steps the environment and updates the frame stack.

        Args:
            action (int): Selected action.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]:
                Stacked observation, reward, terminated, truncated, info.
        """
        obs, reward, terminated, truncated, info = _step_compat(self.env, action)
        self.frames.append(obs)
        if self.stacked_obs is None:
            self.stacked_obs = np.stack(list(self.frames), axis=0).astype(np.float32)
        else:
            self.stacked_obs[:-1] = self.stacked_obs[1:]
            self.stacked_obs[-1] = np.asarray(obs, dtype=np.float32)
        return self._get_observation(), reward, terminated, truncated, info


def make_mario_env(
    env_id: str = "SuperMarioBros-1-1-v3",
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    frame_skip: int = 4,
):
    """
    Creates the fully wrapped Mario environment.

    Args:
        env_id (str): Mario environment ID.
        render_mode (Optional[str]): Render mode passed to the environment.
        seed (Optional[int]): Optional random seed.
        frame_skip (int): Number of frames to repeat each selected action.

    Returns:
        tuple:
            env (gym.Env): Wrapped Mario environment.
            observation_shape (tuple): Final observation shape.
            action_size (int): Number of discrete actions.
    """
    env = gym_super_mario_bros.make(
        env_id,
        apply_api_compatibility=True,
        render_mode=render_mode,
    )

    env = JoypadSpace(env, RIGHT_ONLY)
    env = MarioRewardWrapper(env)
    env = SkipFrame(env, skip=frame_skip)
    env = StagnationTerminationWrapper(env)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStackObservation(env, num_stack=4)

    if seed is not None:
        _reset_compat(env, seed=seed)

    observation_shape = env.observation_space.shape
    action_size = env.action_space.n
    return env, observation_shape, action_size
