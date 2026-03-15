"""
Uniform experience replay buffer.
"""

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity uniform replay buffer for storing transitions.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initializes the replay buffer.

        Args:
            capacity (int): Maximum number of transitions stored.
        """
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Adds a transition to the replay buffer.

        Args:
            state (np.ndarray): Current state.
            action (int): Selected action.
            reward (float): Received reward.
            next_state (np.ndarray): Next state.
            done (bool): Terminal flag.
        """
        transition = (
            np.array(state, copy=True, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, copy=True, dtype=np.float32),
            float(done),
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        """
        Samples a random mini batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple:
                A tuple containing (states (np.ndarray), actions (np.ndarray), rewards (np.ndarray), next_states (np.ndarray), dones (np.ndarray)).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    def state_dict(self) -> dict:
        """
        Serializes the replay buffer for checkpointing.

        Returns:
            dict: Replay buffer state.
        """
        return {
            "capacity": self.capacity,
            "buffer": list(self.buffer),
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restores the replay buffer from a checkpoint.

        Args:
            state (dict): Serialized replay buffer state.
        """
        self.capacity = int(state["capacity"])
        self.buffer = deque(state["buffer"], maxlen=self.capacity)

    def __len__(self) -> int:
        """
        Returns the current number of stored transitions.

        Returns:
            int: Current replay buffer size.
        """
        return len(self.buffer)
