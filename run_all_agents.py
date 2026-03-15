import argparse
import subprocess
import time
import os
import yaml

CONFIG_FILE = "config.yaml"
SUPPORTED_AGENTS = ["d3qn", "d3qn_er", "d3qn_per"]

def run_agent(agent_type: str) -> None:
    """
    Reads config.yaml, sets the requested agent, and runs the training script.

    Args:
        agent_type (str): The identifier for the agent (d3qn, d3qn_er, d3qn_per).

    Returns:
        None
    """
    print(f"Starting Training for Agent: {agent_type}")
    print("=" * 50)
    
    # Read the existing config
    with open(CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)
        
    print(f"\n[{time.strftime('%H:%M:%S')}] Preparing to train {agent_type}...")
    
    # Update config.yaml with the current agent
    config['agent_type'] = agent_type
    
    with open(CONFIG_FILE, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)
        
    print(f"[{time.strftime('%H:%M:%S')}] Updated {CONFIG_FILE} to run {agent_type}.")
    print(f"[{time.strftime('%H:%M:%S')}] Launching training script...")
    
    # Run the training script
    try:
        # We use subprocess.run so it waits until the training is completely finished before moving on
        process = subprocess.run(
            ["python", "training_script.py"],
            check=True
        )
        print(f"\n[{time.strftime('%H:%M:%S')}] Successfully finished training {agent_type}!\n")
        print("-" * 50)
        
    except subprocess.CalledProcessError as e:
        print(f"\n[{time.strftime('%H:%M:%S')}] ERROR: Training failed for {agent_type}.")
        print(f"Error code: {e.returncode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a specific RL agent for training.")
    parser.add_argument(
        "agent", 
        type=str, 
        choices=SUPPORTED_AGENTS, 
        help=f"The agent to train. Choose from: {SUPPORTED_AGENTS}"
    )
    
    args = parser.parse_args()
    run_agent(args.agent)
