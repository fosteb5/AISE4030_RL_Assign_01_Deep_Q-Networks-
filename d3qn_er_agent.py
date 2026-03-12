"""
D3QN agent with uniform experience replay.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from d3qn_agent import D3QNAgent
from replay_buffer import ReplayBuffer


class D3QNERAgent(D3QNAgent):
    """
    Double Dueling DQN agent with uniform replay buffer.
    """

    def __init__(self, state_shape: Tuple[int, int, int], num_actions: int, config: Dict) -> None:
        """
        Initializes the replay based D3QN agent.

        Args:
            state_shape (Tuple[int, int, int]): Observation shape.
            num_actions (int): Number of actions.
            config (Dict): Full configuration dictionary.
        """
        super().__init__(state_shape, num_actions, config)

        replay_cfg = config["replay"]
        self.batch_size = int(replay_cfg["batch_size"])
        self.learning_starts = int(replay_cfg["learning_starts"])
        self.replay_buffer = ReplayBuffer(int(replay_cfg["capacity"]))

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Stores the transition and learns from a uniform random mini batch.

        Args:
            state (np.ndarray): Current state.
            action (int): Selected action.
            reward (float): Received reward.
            next_state (np.ndarray): Next state.
            done (bool): Terminal flag.

        Returns:
            Optional[float]: Loss value if learning occurred, else None.
        """
        self.global_step += 1
        self.replay_buffer.add(state, action, reward, next_state, done)

        loss = None
        if len(self.replay_buffer) >= self.learning_starts and len(self.replay_buffer) >= self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            loss, _ = self._learn_from_batch(*batch)

        if self.global_step % self.target_sync_steps == 0:
            self._sync_target_network()

        self._update_epsilon()
        return loss

    def get_checkpoint_state(self) -> Dict:
        """
        Returns the serializable training state for checkpointing.

        Returns:
            Dict: Agent checkpoint state.
        """
        return super().get_checkpoint_state()

    def load_checkpoint_state(self, checkpoint: Dict) -> None:
        """
        Restores the agent from a serialized checkpoint state.

        Args:
            checkpoint (Dict): Agent checkpoint state.
        """
        super().load_checkpoint_state(checkpoint)
