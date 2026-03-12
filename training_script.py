"""
Main training entry point for all Mario D3QN experiments.
"""

import os
import re
from typing import Dict

import numpy as np
import torch
import yaml

from d3qn_agent import D3QNAgent
from d3qn_er_agent import D3QNERAgent
from d3qn_per_agent import D3QNPERAgent
from environment import make_mario_env
from utils import (
    ensure_dir,
    load_config,
    load_history,
    maybe_create_comparison_plots,
    plot_agent_history,
    resolve_paths,
    save_history,
    set_seed,
)


def get_resume_signature(config: Dict) -> Dict:
    """
    Builds the subset of config values that must match for resuming.

    Args:
        config (Dict): Full configuration dictionary.

    Returns:
        Dict: Resume signature.
    """
    signature = {
        "agent_type": config["agent_type"],
        "env_id": config["env_id"],
        "seed": int(config["seed"]),
        "frame_skip": int(config.get("frame_skip", 4)),
        "training": {
            key: config["training"][key]
            for key in (
                "total_episodes",
                "max_steps_per_episode",
                "learning_rate",
                "gamma",
                "target_sync_steps",
                "gradient_clip",
            )
        },
    }

    agent_type = config["agent_type"].lower()
    if agent_type in {"d3qn_er", "d3qn_per"}:
        signature["replay"] = dict(config["replay"])
    if agent_type == "d3qn_per":
        signature["per"] = dict(config["per"])

    return signature


def find_latest_checkpoint(results_dir: str) -> str | None:
    """
    Finds the latest numbered checkpoint in the results directory.

    Args:
        results_dir (str): Directory containing checkpoints.

    Returns:
        str | None: Latest checkpoint path, if any.
    """
    rolling_checkpoint = os.path.join(results_dir, "checkpoint_latest.pth")
    if os.path.exists(rolling_checkpoint):
        return rolling_checkpoint

    latest_path = None
    latest_episode = -1
    pattern = re.compile(r"checkpoint_ep_(\d+)\.pth$")

    for filename in os.listdir(results_dir):
        match = pattern.match(filename)
        if match is None:
            continue

        episode = int(match.group(1))
        if episode > latest_episode:
            latest_episode = episode
            latest_path = os.path.join(results_dir, filename)

    return latest_path


def save_training_checkpoint(
    agent,
    filepath: str,
    config: Dict,
    completed_episodes: int,
) -> None:
    """
    Saves a resumable training checkpoint.

    Args:
        agent: Active agent instance.
        filepath (str): Checkpoint path.
        config (Dict): Full configuration.
        completed_episodes (int): Number of finished episodes.
    """
    checkpoint = {
        "resume_signature": get_resume_signature(config),
        "completed_episodes": int(completed_episodes),
        "agent_state": agent.get_checkpoint_state(),
    }
    torch.save(checkpoint, filepath)


def try_resume_training(agent, config: Dict, results_dir: str):
    """
    Attempts to resume training from the latest matching checkpoint.

    Args:
        agent: Active agent instance.
        config (Dict): Full configuration.
        results_dir (str): Results directory for the selected agent.

    Returns:
        tuple[int, Dict | None]:
            starting episode index and restored history if resume succeeds.
    """
    checkpoint_path = find_latest_checkpoint(results_dir)
    if checkpoint_path is None:
        return 1, None, None

    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    if checkpoint.get("resume_signature") != get_resume_signature(config):
        return 1, None, None

    agent.load_checkpoint_state(checkpoint["agent_state"])
    completed_episodes = int(checkpoint.get("completed_episodes", 0))
    history = checkpoint.get("history", None)
    if history is None:
        history_path = os.path.join(results_dir, "history.json")
        if os.path.exists(history_path):
            history = load_history(history_path)
    return completed_episodes + 1, history, checkpoint_path


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
    paths_cfg = resolve_paths(config)

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

    env, state_shape, num_actions = make_mario_env(
        env_id=config["env_id"],
        render_mode=config.get("render_mode", None),
        seed=int(config["seed"]),
        frame_skip=int(config.get("frame_skip", 4)),
    )

    agent = build_agent(config, state_shape, num_actions)
    results_dir = get_results_dir(config)
    ensure_dir(results_dir)

    with open(os.path.join(results_dir, "config_used.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

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
        "flag_reached": [],
        "flag_reach_rate_percent": [],
        "end_reason": [],
        "death_rate_percent": [],
        "stagnation_rate_percent": [],
        "timeout_rate_percent": [],
    }
    start_episode, resumed_history, resumed_checkpoint = try_resume_training(agent, config, results_dir)
    if resumed_history is not None:
        history = resumed_history
        print(f"Resuming from episode {start_episode} using {os.path.basename(resumed_checkpoint)}")

    if torch.cuda.is_available():
        print(f"[GPU] Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[CPU] CUDA not available - training on CPU")
    print(f"Training agent: {config['agent_type']}")
    print(f"Device: {agent.device}")
    print(f"Observation shape: {state_shape}")
    print(f"Action space: {num_actions}")

    if start_episode > total_episodes:
        print("Training already matches or exceeds the configured total_episodes. Nothing to resume.")
        env.close()
        return

    for episode in range(start_episode, total_episodes + 1):
        state, info = env.reset(seed=int(config["seed"]) + episode)
        state = np.asarray(state, dtype=np.float32)

        episode_reward = 0.0
        episode_loss_total = 0.0
        episode_loss_count = 0
        episode_steps = 0
        reached_flag = False
        end_reason = "timeout"
        last_info = {}

        done = False
        while not done and episode_steps < max_steps_per_episode:
            action = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32)
            done = bool(terminated or truncated)
            reached_flag = reached_flag or bool(info.get("flag_get", False))
            last_info = info

            loss = agent.step(state, action, reward, next_state, done)

            state = next_state
            episode_reward += float(reward)
            episode_steps += 1

            if loss is not None:
                episode_loss_total += float(loss)
                episode_loss_count += 1

        if reached_flag:
            end_reason = "flag"
        elif bool(last_info.get("stagnation_terminated", False)):
            end_reason = "stagnation"
        elif done:
            end_reason = "death"

        mean_loss = episode_loss_total / episode_loss_count if episode_loss_count > 0 else 0.0

        history["episode_rewards"].append(float(episode_reward))
        history["episode_losses"].append(mean_loss)
        history["episode_lengths"].append(int(episode_steps))
        history["epsilon_values"].append(float(agent.epsilon))
        history["flag_reached"].append(int(reached_flag))
        history["end_reason"].append(end_reason)
        history["flag_reach_rate_percent"].append(
            100.0 * float(np.mean(history["flag_reached"]))
        )
        history["death_rate_percent"].append(
            100.0 * float(np.mean([reason == "death" for reason in history["end_reason"]]))
        )
        history["stagnation_rate_percent"].append(
            100.0 * float(np.mean([reason == "stagnation" for reason in history["end_reason"]]))
        )
        history["timeout_rate_percent"].append(
            100.0 * float(np.mean([reason == "timeout" for reason in history["end_reason"]]))
        )

        if episode % log_every == 0 or episode == 1:
            recent_rewards = history["episode_rewards"][-window:]
            moving_reward = float(np.mean(recent_rewards))
            recent_flag_rate = 100.0 * float(np.mean(history["flag_reached"][-window:]))
            recent_death_rate = 100.0 * float(
                np.mean([reason == "death" for reason in history["end_reason"][-window:]])
            )
            recent_stagnation_rate = 100.0 * float(
                np.mean([reason == "stagnation" for reason in history["end_reason"][-window:]])
            )
            recent_timeout_rate = 100.0 * float(
                np.mean([reason == "timeout" for reason in history["end_reason"][-window:]])
            )
            print(
                f"Episode {episode}/{total_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg Reward: {moving_reward:.2f} | "
                f"Loss: {mean_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Flag Rate: {recent_flag_rate:.1f}% | "
                f"Death: {recent_death_rate:.1f}% | "
                f"Stag: {recent_stagnation_rate:.1f}% | "
                f"Timeout: {recent_timeout_rate:.1f}%"
            )

        if episode % save_every == 0:
            checkpoint_path = os.path.join(results_dir, "checkpoint_latest.pth")
            save_training_checkpoint(agent, checkpoint_path, config, episode)
            save_history(history, results_dir, filename="history.json")

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
