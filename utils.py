"""
Utility helpers for config loading, plotting, seeding, and history saving.
"""

import json
import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def ensure_dir(path: str) -> None:
    """
    Creates a directory if it does not already exist.

    Args:
        path (str): Directory path.
    """
    os.makedirs(path, exist_ok=True)


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        Dict: Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_paths(config: Dict) -> Dict:
    """
    Resolves output paths, optionally nesting them under a run version.

    Args:
        config (Dict): Full configuration dictionary.

    Returns:
        Dict: Resolved path mapping.
    """
    paths_cfg = dict(config["paths"])
    run_version = config.get("run_version", None)
    if run_version in (None, ""):
        return paths_cfg

    return {
        key: os.path.join(path, str(run_version))
        for key, path in paths_cfg.items()
    }


def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def moving_average(values: List[float], window: int = 50) -> List[float]:
    """
    Computes a moving average over a list.

    Args:
        values (List[float]): Input values.
        window (int): Moving average window size.

    Returns:
        List[float]: Smoothed values.
    """
    if len(values) == 0:
        return []

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(float(np.mean(values[start:i + 1])))
    return smoothed


def save_history(history: Dict, save_dir: str, filename: str = "history.json") -> str:
    """
    Saves training history to a JSON file.

    Args:
        history (Dict): Training history dictionary.
        save_dir (str): Output directory.
        filename (str): Output file name.

    Returns:
        str: Saved file path.
    """
    ensure_dir(save_dir)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    return filepath


def load_history(filepath: str) -> Dict:
    """
    Loads a history JSON file.

    Args:
        filepath (str): Path to the history file.

    Returns:
        Dict: Parsed history dictionary.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def plot_metric(
    values: List[float],
    save_path: str,
    title: str,
    ylabel: str,
    window: int = 50,
) -> None:
    """
    Plots a metric and its moving average.

    Args:
        values (List[float]): Metric values.
        save_path (str): Output image path.
        title (str): Plot title.
        ylabel (str): Y-axis label.
        window (int): Moving average window.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(values, label="Raw")
    plt.plot(moving_average(values, window=window), label=f"Moving Avg ({window})")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_agent_history(history: Dict, save_dir: str, window: int = 50) -> None:
    """
    Plots reward and loss curves for a single agent.

    Args:
        history (Dict): Training history dictionary.
        save_dir (str): Output directory.
        window (int): Moving average window.
    """
    ensure_dir(save_dir)

    plot_metric(
        history["episode_rewards"],
        os.path.join(save_dir, "reward_curve.png"),
        "Episode Reward",
        "Reward",
        window=window,
    )

    plot_metric(
        history["episode_losses"],
        os.path.join(save_dir, "loss_curve.png"),
        "Episode Loss",
        "Loss",
        window=window,
    )

    if "flag_reach_rate_percent" in history:
        plot_metric(
            history["flag_reach_rate_percent"],
            os.path.join(save_dir, "flag_rate_curve.png"),
            "Flag Reach Rate",
            "Percent",
            window=window,
        )

    for metric_key, filename, title in (
        ("death_rate_percent", "death_rate_curve.png", "Death Rate"),
        ("stagnation_rate_percent", "stagnation_rate_curve.png", "Stagnation Rate"),
        ("timeout_rate_percent", "timeout_rate_curve.png", "Timeout Rate"),
    ):
        if metric_key in history:
            plot_metric(
                history[metric_key],
                os.path.join(save_dir, filename),
                title,
                "Percent",
                window=window,
            )


def plot_overlay(
    histories: Dict[str, Dict],
    save_path: str,
    metric_key: str,
    title: str,
    ylabel: str,
    window: int = 50,
) -> None:
    """
    Plots the same metric for multiple agents on a single figure.

    Args:
        histories (Dict[str, Dict]): Mapping of agent names to their respective history dictionaries.
        save_path (str): Full output file path for the image.
        metric_key (str): The specific history key to plot (e.g., 'episode_rewards').
        title (str): Title displayed on the plot.
        ylabel (str): Label for the y-axis.
        window (int): Moving average smoothing window size.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    for agent_name, history in histories.items():
        values = history.get(metric_key, [])
        plt.plot(moving_average(values, window=window), label=agent_name)

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def maybe_create_comparison_plots(config: Dict) -> None:
    """
    Creates overlaid comparison plots if at least two history files exist.

    Args:
        config (Dict): Full configuration dictionary.
    """
    paths_cfg = resolve_paths(config)
    comparison_dir = paths_cfg["comparison_results"]
    ensure_dir(comparison_dir)

    mapping = {
        "D3QN": os.path.join(paths_cfg["d3qn_results"], "history.json"),
        "D3QN_ER": os.path.join(paths_cfg["d3qn_er_results"], "history.json"),
        "D3QN_PER": os.path.join(paths_cfg["d3qn_per_results"], "history.json"),
    }

    histories = {}
    for name, filepath in mapping.items():
        if os.path.exists(filepath):
            histories[name] = load_history(filepath)

    if len(histories) < 2:
        return

    window = int(config["training"]["moving_average_window"])

    plot_overlay(
        histories,
        os.path.join(comparison_dir, "reward_overlay.png"),
        metric_key="episode_rewards",
        title="Reward Curve Comparison",
        ylabel="Reward",
        window=window,
    )

    plot_overlay(
        histories,
        os.path.join(comparison_dir, "loss_overlay.png"),
        metric_key="episode_losses",
        title="Loss Curve Comparison",
        ylabel="Loss",
        window=window,
    )

    if all("flag_reach_rate_percent" in history for history in histories.values()):
        plot_overlay(
            histories,
            os.path.join(comparison_dir, "flag_rate_overlay.png"),
            metric_key="flag_reach_rate_percent",
            title="Flag Reach Rate Comparison",
            ylabel="Percent",
            window=window,
        )

    for metric_key, filename, title in (
        ("death_rate_percent", "death_rate_overlay.png", "Death Rate Comparison"),
        ("stagnation_rate_percent", "stagnation_rate_overlay.png", "Stagnation Rate Comparison"),
        ("timeout_rate_percent", "timeout_rate_overlay.png", "Timeout Rate Comparison"),
    ):
        if all(metric_key in history for history in histories.values()):
            plot_overlay(
                histories,
                os.path.join(comparison_dir, filename),
                metric_key=metric_key,
                title=title,
                ylabel="Percent",
                window=window,
            )
