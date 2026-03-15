"""
Prioritized experience replay buffer using a Sum Tree.
"""

import random
from typing import List, Optional, Tuple

import numpy as np


class SumTree:
    """
    Sum Tree data structure for proportional priority sampling.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initializes the Sum Tree.

        Args:
            capacity (int): Maximum number of stored transitions.
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    @property
    def total_priority(self) -> float:
        """
        Returns the total priority stored in the root.

        Returns:
            float: Total priority.
        """
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """
        Returns the maximum leaf priority currently stored.

        Returns:
            float: Maximum priority.
        """
        if self.size == 0:
            return 1.0
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.size
        max_value = np.max(self.tree[leaf_start:leaf_end])
        return float(max(max_value, 1.0e-6))

    def add(self, priority: float, data: tuple) -> None:
        """
        Adds a transition with a given priority.

        Args:
            priority (float): Priority value.
            data (tuple): Transition tuple.
        """
        tree_index = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_index, priority)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_index: int, priority: float) -> None:
        """
        Updates the priority of a tree leaf and propagates the change upward.

        Args:
            tree_index (int): Leaf index in the tree array.
            priority (float): New priority value.
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value: float) -> Tuple[int, float, tuple]:
        """
        Retrieves a leaf based on a cumulative priority value.

        Args:
            value (float): Sample value in [0, total_priority].

        Returns:
            Tuple[int, float, tuple]:
                tree_index, priority, stored transition
        """
        parent = 0

        while True:
            left = 2 * parent + 1
            right = left + 1

            if left >= len(self.tree):
                leaf = parent
                break

            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right

        data_index = leaf - self.capacity + 1
        return leaf, float(self.tree[leaf]), self.data[data_index]

    def state_dict(self) -> dict:
        """
        Serializes the sum tree for checkpointing.

        Returns:
            dict: Sum tree state.
        """
        return {
            "capacity": self.capacity,
            "tree": self.tree.copy(),
            "data": list(self.data),
            "write": self.write,
            "size": self.size,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restores the sum tree from a checkpoint.

        Args:
            state (dict): Serialized sum tree state.
        """
        self.capacity = int(state["capacity"])
        self.tree = np.array(state["tree"], dtype=np.float32, copy=True)
        self.data = list(state["data"])
        self.write = int(state["write"])
        self.size = int(state["size"])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer backed by a Sum Tree.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1.0e-5) -> None:
        """
        Initializes the PER buffer.

        Args:
            capacity (int): Maximum number of transitions.
            alpha (float): Prioritization exponent.
            epsilon (float): Small constant to prevent zero priority.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def _priority_from_td_error(self, td_error: float) -> float:
        """
        Converts TD error into a priority value.

        Args:
            td_error (float): Temporal difference error.

        Returns:
            float: Priority value.
        """
        return float((abs(td_error) + self.epsilon) ** self.alpha)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """
        Adds a transition to the PER buffer with an initial priority.

        Args:
            state (np.ndarray): Current state observation.
            action (int): Index of selected action.
            reward (float): Received reward value.
            next_state (np.ndarray): Resulting state observation.
            done (bool): Terminal flag for the transition.
            priority (Optional[float]): Initial priority. Defaults to max_priority if None.

        Returns:
            None
        """
        if priority is None:
            priority = self.max_priority

        transition = (
            np.array(state, copy=True, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, copy=True, dtype=np.float32),
            float(done),
        )
        priority = float(priority)
        tree_index = self.tree.write + self.tree.capacity - 1
        overwritten_priority = float(self.tree.tree[tree_index])
        self.tree.add(priority, transition)
        if overwritten_priority >= self.max_priority and priority < overwritten_priority:
            self.max_priority = self.tree.max_priority
        else:
            self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int, beta: float):
        """
        Samples a mini batch proportional to priority with IS weights.

        Args:
            batch_size (int): Number of transitions to sample.
            beta (float): Importance sampling correction parameter.

        Returns:
            tuple:
                A tuple containing (states (np.ndarray), actions (np.ndarray), rewards (np.ndarray), 
                next_states (np.ndarray), dones (np.ndarray), indices (np.ndarray), weights (np.ndarray)).
        """
        indices: List[int] = []
        priorities: List[float] = []
        batch: List[tuple] = []

        total_priority = self.tree.total_priority
        segment = total_priority / batch_size

        for i in range(batch_size):
            start = segment * i
            end = segment * (i + 1)
            data = None
            while data is None:
                value = min(random.uniform(start, end), total_priority - 1e-6)
                index, priority, data = self.tree.get_leaf(value)
            indices.append(index)
            priorities.append(priority)
            batch.append(data)

        states, actions, rewards, next_states, dones = zip(*batch)

        probabilities = np.array(priorities, dtype=np.float32) / max(total_priority, 1.0e-8)
        weights = (self.tree.size * probabilities) ** (-beta)
        weights /= np.max(weights)

        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
            np.array(indices, dtype=np.int64),
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Updates priorities for sampled transitions.

        Args:
            indices (np.ndarray): Tree indices of sampled transitions.
            td_errors (np.ndarray): New TD errors for sampled transitions.
        """
        for index, td_error in zip(indices, td_errors):
            previous_priority = float(self.tree.tree[int(index)])
            priority = self._priority_from_td_error(float(td_error))
            self.tree.update(int(index), priority)
            if previous_priority >= self.max_priority and priority < previous_priority:
                self.max_priority = self.tree.max_priority
            else:
                self.max_priority = max(self.max_priority, priority)

    def get_max_priority(self) -> float:
        """
        Returns the current maximum priority.

        Returns:
            float: Maximum stored priority.
        """
        return self.max_priority

    def state_dict(self) -> dict:
        """
        Serializes the PER buffer for checkpointing.

        Returns:
            dict: PER buffer state.
        """
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "max_priority": self.max_priority,
            "tree": self.tree.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restores the PER buffer from a checkpoint.

        Args:
            state (dict): Serialized PER buffer state.
        """
        self.capacity = int(state["capacity"])
        self.alpha = float(state["alpha"])
        self.epsilon = float(state["epsilon"])
        self.max_priority = float(state["max_priority"])
        self.tree = SumTree(self.capacity)
        self.tree.load_state_dict(state["tree"])

    def __len__(self) -> int:
        """
        Returns the current number of stored transitions.

        Returns:
            int: Current buffer size.
        """
        return self.tree.size
