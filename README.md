# AISE 4030 RL Assignment 01

## Group
**RL Group 10**

## Project Description
Developed and evaluated deep reinforcement learning agents in Python and PyTorch to study training stability, convergence, and replay strategies in **Super Mario Bros**.

This project implements and compares three **Double Dueling Deep Q Network** variants on **SuperMarioBros-1-1-v3**:

1. **D3QN without Experience Replay**
2. **D3QN with Uniform Experience Replay**
3. **D3QN with Prioritized Experience Replay using a Sum Tree**

The goal is to isolate how different experience management strategies affect learning stability, convergence speed, and overall performance.

## Assignment Objective
All three agents use the same **Double Dueling DQN** architecture. The only intended change between experiments is the way experience is handled:

- **Online learning** from single transitions
- **Uniform random sampling** from a replay buffer
- **Priority based sampling** with importance sampling correction

This keeps the comparison fair and makes it easier to evaluate the effect of replay and prioritization.

## Environment
The project uses:

- **Environment:** `SuperMarioBros-1-1-v3`
- **Action Space:** `Discrete(2)`
- **Final Observation Shape:** `(4, 84, 84)`

### Action Mapping
The simplified action space uses only two actions:

- `0` → move right
- `1` → move right + jump

### Observation Preprocessing
The Mario environment is wrapped in the following order:

1. `SkipFrame`
2. `GrayScaleObservation`
3. `ResizeObservation`
4. `FrameStackObservation`

The custom reward wrapper also applies:

- x position progress reward
- time penalty per step
- flag completion bonus
- death penalty
- stagnation penalty on early truncation
- reward clipping to `[-15, 15]`

## Repository Structure
```text
.
├── config.yaml
├── environment.py
├── d3qn_network.py
├── d3qn_agent.py
├── d3qn_er_agent.py
├── d3qn_per_agent.py
├── replay_buffer.py
├── per_buffer.py
├── training_script.py
├── utils.py
├── README.md
├── requirements.txt
├── d3qn_results/
├── d3qn_er_results/
└── d3qn_per_results/
```

## Core Files

### `environment.py`
Creates the Super Mario Bros environment and applies all required wrappers.

### `d3qn_network.py`
Defines the shared **Double Dueling DQN** network with:

- convolutional feature extractor
- value stream
- advantage stream
- dueling Q value recombination

### `d3qn_agent.py`
Implements the baseline **online D3QN** agent that learns directly from the current transition without a replay buffer.

### `replay_buffer.py`
Implements a standard **uniform replay buffer**.

### `d3qn_er_agent.py`
Implements **D3QN with uniform experience replay**.

### `per_buffer.py`
Implements **Prioritized Experience Replay** using a **Sum Tree**.

### `d3qn_per_agent.py`
Implements **D3QN with PER**, including:

- priority based sampling
- importance sampling weights
- priority updates from TD error
- beta annealing over training

### `training_script.py`
Main entry point for training all agent variants using the `agent_type` field in `config.yaml`.

### `utils.py`
Includes helper functions for:

- loading config files
- setting seeds
- saving training history
- plotting reward and loss curves
- generating overlay comparison plots

## Installation

### 1. Create the Conda Environment
```bash
conda create -n AISE4030 python=3.10 -y
conda activate AISE4030
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Required Dependencies
The project uses the following dependencies:

```txt
torch
torchvision
gymnasium==1.2.3
gym==0.26.2
gym-super-mario-bros==7.4.0
nes-py
numpy==1.26.4
matplotlib
pyyaml
opencv-python
```

## Verify the Environment
Run the following command to confirm that the environment is set up correctly:

```bash
python -c "
from training_script import make_mario_env
env, obs_shape, action_size = make_mario_env('SuperMarioBros-1-1-v3', render_mode=None, seed=42)
obs, info = env.reset()
import numpy as np
obs = np.array(obs)
print('Environment created successfully!')
print('Observation shape:', obs.shape)
print('Action space:', env.action_space)
env.close()
print('Setup is complete!')
"
```

Expected output:

```text
Environment created successfully!
Observation shape: (4, 84, 84)
Action space: Discrete(2)
Setup is complete!
```

## Configuration
All training settings are controlled through `config.yaml`.

If `run_version` is set in `config.yaml`, outputs are written into versioned subfolders such as `d3qn_results/<run_version>/`, `d3qn_er_results/<run_version>/`, `d3qn_per_results/<run_version>/`, and `comparison_results/<run_version>/`.

### Current Default Agent
```yaml
agent_type: d3qn
```

### Supported Agent Types
Set `agent_type` to one of the following:

```yaml
agent_type: d3qn
```

```yaml
agent_type: d3qn_er
```

```yaml
agent_type: d3qn_per
```

### Current Key Hyperparameters
```yaml
render_mode: none
device: auto
frame_skip: 4
run_version: v2

training:
  total_episodes: 5000
  max_steps_per_episode: 50000
  learning_rate: 0.00025
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_min: 0.05
  epsilon_decay: 0.999995
  target_sync_steps: 1000
  gradient_clip: 1.0
  save_every: 500
  log_every: 5
  moving_average_window: 50

replay:
  batch_size: 32
  capacity: 100000
  learning_starts: 5000

per:
  alpha: 0.6
  beta_start: 0.4
  beta_end: 1.0
  epsilon: 1.0e-5
```

## How to Run Training

### Train the Baseline D3QN Agent
Set this in `config.yaml`:

```yaml
agent_type: d3qn
```

Then run:

```bash
python training_script.py
```

### Train the Uniform Replay Agent
Set this in `config.yaml`:

```yaml
agent_type: d3qn_er
```

Then run:

```bash
python training_script.py
```

### Train the Prioritized Replay Agent
Set this in `config.yaml`:

```yaml
agent_type: d3qn_per
```

Then run:

```bash
python training_script.py
```

## Training Output
Each training run saves outputs to a dedicated folder:

- `d3qn_results/`
- `d3qn_er_results/`
- `d3qn_per_results/`

Each folder may contain:

- model checkpoints
- final model
- `history.json`
- `reward_curve.png`
- `loss_curve.png`
- `flag_rate_curve.png`
- `death_rate_curve.png`
- `stagnation_rate_curve.png`
- `timeout_rate_curve.png`

If at least two agent histories are available, the project also generates overlay plots in:

- `comparison_results/`

These include:

- `reward_overlay.png`
- `loss_overlay.png`
- `flag_rate_overlay.png`
- `death_rate_overlay.png`
- `stagnation_rate_overlay.png`
- `timeout_rate_overlay.png`

## Training Workflow
A typical workflow is:

1. Verify the environment setup
2. Train `d3qn`
3. Train `d3qn_er`
4. Train `d3qn_per`
5. Compare the generated reward and loss curves
6. Use the plots in the final report

## Method Summary

### D3QN
The model uses a shared convolutional encoder followed by:

- a **value stream**
- an **advantage stream**

The final Q values are computed using dueling recombination.

### Double DQN Target Computation
The implementation uses:

- the **policy network** to choose the next action
- the **target network** to evaluate that action

This reduces overestimation bias compared to standard DQN target computation.

### Uniform Experience Replay
The replay based agent stores transitions and samples random mini batches once the buffer reaches the minimum learning threshold.

### Prioritized Experience Replay
The PER agent uses a Sum Tree to sample more important transitions with higher probability. It also applies importance sampling weights to correct the bias introduced by prioritization.

## Reproducibility
The code sets a random seed from `config.yaml` for:

- Python random
- NumPy
- PyTorch

This helps improve experiment consistency across runs.

## Notes
- This implementation is designed to match the assignment file naming and structure.
- The same main training script is used for all three agents.
- The selected agent is controlled only by `config.yaml`.
- Reward and loss plots are automatically generated after training.
- Comparison plots are generated when enough histories exist.

## Report Support
This repository is structured to support the assignment report by producing:

- environment verification output
- per agent reward plots
- per agent loss plots
- overlaid reward comparison plots
- overlaid loss comparison plots
- saved training histories for later analysis

## AI Usage Log Reminder
The assignment requires an **AI Usage Log** as an appendix. Make sure to document:

- tool name and version
- purpose of use
- full prompt history
- what was changed before submission

## Author Notes
This project was organized to make it easy to:

- switch between agent variants quickly
- keep experiments consistent
- compare replay strategies fairly
- generate plots directly for the final report

## Training Efficiency Notes
The codebase includes a small set of implementation-level optimizations to reduce CPU overhead during training while preserving the assignment's fairness requirement that the only experimental difference between agents is the experience management strategy.

### Implementation Optimizations Applied
- action selection and target computation avoid unnecessary gradient tracking during evaluation passes
- gradient updates use `optimizer.zero_grad(set_to_none=True)` to reduce optimizer memory work
- repeated state conversions in the training loop use `np.asarray(...)` to avoid unnecessary copies when possible
- the PER buffer caches the current maximum priority instead of rescanning the full sum tree on every insertion

These changes do **not** alter the Dueling Network architecture, Double DQN target computation, replay behavior definitions, or the intended differences between `d3qn`, `d3qn_er`, and `d3qn_per`.

### Deviations from Assignment Defaults and Justifications

The following parameters deviate from the assignment spec. All deviations are applied identically across all three agents to preserve a fair comparison.

| Parameter | Spec Default | Used | Justification |
|---|---|---|---|
| `gamma` | 0.9 | **0.99** | Mnih et al. (2015) use 0.99 for Atari. `gamma=0.9` gives a ~10-step effective horizon, insufficient for Mario's obstacle sequences. `0.99` gives ~100 steps, consistent with the literature. |
| `target_sync_steps` | 10000 | **1000** | With ~236 steps/episode, 10000 steps = ~42 episodes between syncs — too infrequent for a 5000-episode budget. 1000 steps (~4 episodes) provides stable but responsive target updates. |
| `epsilon_min` | 0.1 | **0.05** | With only 2 actions, 10% terminal exploration wastes significant exploitation time. 5% is more appropriate for a minimal action space. |
| `epsilon_decay` | 0.99999975 | **0.999995** | Original decay reaches `epsilon_min` at ~39,000 episodes. Recalibrated to reach minimum at ~episode 2400, providing ~2600 exploitation episodes within the 5000-episode budget. |
| `learning_starts` | 1000 | **5000** | A larger warmup allows ER and PER buffers to accumulate diverse transitions before learning begins. Particularly important for PER, where early random transitions with high TD error can dominate priority sampling. Applied to both ER and PER for consistency. |

### Reward Shaping

Current reward structure applied identically to all three agents:

- Forward x-position progress: `+0.02 × pixels moved` (floored at 0 when moving forward)
- Backward or no movement: `0.02 × pixels` (negative)
- Time penalty: `-0.1` per step (encourages faster completion)
- Death: `-15` (clipped)
- Stagnation termination (150 steps without progress): `-10`
- Flag reached: `+50` (applied after clipping)

### PER Bug Fix

The sum tree `get_leaf` method can return `None` data due to floating-point sampling overshooting `total_priority`. Fixed in `per_buffer.py` by retrying the sample when `None` data is encountered.
