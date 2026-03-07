"""
Main training entry point for all Mario D3QN experiments.
"""

import os
from typing import Dict

import numpy as np

from d3qn_agent import D3QNAgent
from d3qn_er_agent import D3QNERAgent
from d3qn_per_agent import D3QNPERAgent
from environment import make_mario_env
from utils import ensure_dir, load_config, maybe_create_comparison_plots, plot_agent_history, save_history, set_seed


def build_agent(config: Dict, state_shape, num_actions: int):
    """
    Instantiates the correct agent based on config.

    Args:
        config (Dict): Full configuration dictionary.
        state_shape: Observation shape.
        num_actions (int): Number of actions.

    Returns:
        object: Initialized agent instance.
    """
    agent_type = config["agent_type"].lower()

    if agent_type == "d3qn":
        return D3QNAgent(state_shape, num_actions, config)

    if agent_type == "d3qn_er":
        return D3QNERAgent(state_shape, num_actions, config)

    if agent_type == "d3qn_per":
        return D3QNPERAgent(state_shape, num_actions, config)

    raise ValueError(f"Unsupported agent_type: {config['agent_type']}")


def get_results_dir(config: Dict) -> str:
    """
    Returns the results directory for the selected agent.

    Args:
        config (Dict): Full configuration dictionary.

    Returns:
        str: Results directory path.
    """
    agent_type = config["agent_type"].lower()
    paths_cfg = config["paths"]

    if agent_type == "d3qn":
        return paths_cfg["d3qn_results"]
    if agent_type == "d3qn_er":
        return paths_cfg["d3qn_er_results"]
    if agent_type == "d3qn_per":
        return paths_cfg["d3qn_per_results"]

    raise ValueError(f"Unsupported agent_type: {config['agent_type']}")


def train() -> None:
    """
    Runs the full training loop for the configured agent.
    """
    config = load_config("config.yaml")
    set_seed(int(config["seed"]))

    env = make_mario_env(
        env_id=config["env_id"],
        render_mode=config.get("render_mode", None),
        seed=int(config["seed"]),
    )

    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent = build_agent(config, state_shape, num_actions)
    results_dir = get_results_dir(config)
    ensure_dir(results_dir)

    training_cfg = config["training"]
    total_episodes = int(training_cfg["total_episodes"])
    max_steps_per_episode = int(training_cfg["max_steps_per_episode"])
    save_every = int(training_cfg["save_every"])
    log_every = int(training_cfg["log_every"])
    window = int(training_cfg["moving_average_window"])

    history = {
        "agent_type": config["agent_type"],
        "device": str(agent.device),
        "episode_rewards": [],
        "episode_losses": [],
        "episode_lengths": [],
        "epsilon_values": [],
    }

    print(f"Training agent: {config['agent_type']}")
    print(f"Device: {agent.device}")
    print(f"Observation shape: {state_shape}")
    print(f"Action space: {num_actions}")

    for episode in range(1, total_episodes + 1):
        state, info = env.reset(seed=int(config["seed"]) + episode)
        state = np.array(state, dtype=np.float32)

        episode_reward = 0.0
        episode_losses = []
        episode_steps = 0

        done = False
        while not done and episode_steps < max_steps_per_episode:
            action = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            done = bool(terminated or truncated)

            loss = agent.step(state, action, reward, next_state, done)

            state = next_state
            episode_reward += float(reward)
            episode_steps += 1

            if loss is not None:
                episode_losses.append(float(loss))

        mean_loss = float(np.mean(episode_losses)) if len(episode_losses) > 0 else 0.0

        history["episode_rewards"].append(float(episode_reward))
        history["episode_losses"].append(mean_loss)
        history["episode_lengths"].append(int(episode_steps))
        history["epsilon_values"].append(float(agent.epsilon))

        if episode % log_every == 0 or episode == 1:
            recent_rewards = history["episode_rewards"][-window:]
            moving_reward = float(np.mean(recent_rewards))
            print(
                f"Episode {episode}/{total_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg Reward: {moving_reward:.2f} | "
                f"Loss: {mean_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

        if episode % save_every == 0:
            checkpoint_path = os.path.join(results_dir, f"checkpoint_ep_{episode}.pth")
            agent.save(checkpoint_path)

    final_model_path = os.path.join(results_dir, "final_model.pth")
    agent.save(final_model_path)

    save_history(history, results_dir, filename="history.json")
    plot_agent_history(history, results_dir, window=window)
    maybe_create_comparison_plots(config)

    env.close()
    print("Training complete.")
    print(f"Saved outputs to: {results_dir}")


if __name__ == "__main__":
    train()
