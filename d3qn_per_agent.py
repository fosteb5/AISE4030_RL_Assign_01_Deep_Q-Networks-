"""
D3QN agent with prioritized experience replay.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from d3qn_agent import D3QNAgent
from per_buffer import PrioritizedReplayBuffer


class D3QNPERAgent(D3QNAgent):
    """
    Double Dueling DQN agent with Prioritized Experience Replay.
    """

    def __init__(self, state_shape: Tuple[int, int, int], num_actions: int, config: Dict) -> None:
        """
        Initializes the PER based D3QN agent.

        Args:
            state_shape (Tuple[int, int, int]): Observation shape.
            num_actions (int): Number of actions.
            config (Dict): Full configuration dictionary.
        """
        super().__init__(state_shape, num_actions, config)

        replay_cfg = config["replay"]
        per_cfg = config["per"]
        training_cfg = config["training"]

        self.batch_size = int(replay_cfg["batch_size"])
        self.learning_starts = int(replay_cfg["learning_starts"])

        self.beta_start = float(per_cfg["beta_start"])
        self.beta_end = float(per_cfg["beta_end"])
        self.total_episodes = int(training_cfg["total_episodes"])
        self.max_steps_per_episode = int(training_cfg["max_steps_per_episode"])
        self.total_anneal_steps = max(1, self.total_episodes * self.max_steps_per_episode)

        self.per_buffer = PrioritizedReplayBuffer(
            capacity=int(replay_cfg["capacity"]),
            alpha=float(per_cfg["alpha"]),
            epsilon=float(per_cfg["epsilon"]),
        )

    def _get_beta(self) -> float:
        """
        Computes the current beta value by linear annealing.

        Returns:
            float: Current beta parameter.
        """
        fraction = min(1.0, self.global_step / self.total_anneal_steps)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Stores the transition, samples prioritized mini batches,
        applies importance weights, and updates priorities.

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

        initial_priority = self.per_buffer.get_max_priority()
        self.per_buffer.add(state, action, reward, next_state, done, priority=initial_priority)

        loss = None
        if len(self.per_buffer) >= self.learning_starts and len(self.per_buffer) >= self.batch_size:
            beta = self._get_beta()
            batch = self.per_buffer.sample(self.batch_size, beta=beta)
            states, actions, rewards, next_states, dones, indices, weights = batch

            loss, td_errors = self._learn_from_batch(
                states,
                actions,
                rewards,
                next_states,
                dones,
                weights=weights,
            )
            self.per_buffer.update_priorities(indices, td_errors)

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
