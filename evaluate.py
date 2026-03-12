"""
Render-only evaluation script. Loads a saved model and watches it play Mario.
"""

import argparse
import time

import numpy as np
import torch
import yaml

from d3qn_network import D3QNNetwork
from environment import make_mario_env


def load_model(model_path: str, state_shape: tuple, num_actions: int, device: torch.device) -> D3QNNetwork:
    net = D3QNNetwork(state_shape, num_actions).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    if "policy_state_dict" in state:
        net.load_state_dict(state["policy_state_dict"])
    else:
        net.load_state_dict(state)
    net.eval()
    return net


def select_action(net: D3QNNetwork, state: np.ndarray, device: torch.device) -> int:
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = net(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def run(model_path: str, config_path: str, episodes: int, delay: float) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env, state_shape, num_actions = make_mario_env(
        env_id=config["env_id"],
        render_mode="human",
        seed=config.get("seed"),
        frame_skip=int(config.get("frame_skip", 4)),
    )

    net = load_model(model_path, state_shape, num_actions, device)
    print(f"Loaded: {model_path}")
    print(f"Running {episodes} episode(s)...\n")

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=config.get("seed", 42) + ep)
        state = np.asarray(state, dtype=np.float32)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = select_action(net, state, device)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = np.asarray(next_state, dtype=np.float32)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            if delay > 0:
                time.sleep(delay)

        flag = bool(info.get("flag_get", False))
        print(f"Episode {ep:>3} | Steps: {steps:>5} | Reward: {total_reward:>8.2f} | Flag: {'YES' if flag else 'no'}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved Mario model with rendering.")
    parser.add_argument("model", help="Path to .pth model file (final_model.pth or checkpoint_latest.pth)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (default: 5)")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between steps, e.g. 0.02 to slow down (default: 0)")
    args = parser.parse_args()

    run(args.model, args.config, args.episodes, args.delay)
