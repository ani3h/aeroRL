# aeroRL: AI-Powered F1 Aerodynamics Optimization

AeroRL is an AI-driven F1 aerodynamics optimization system that uses **Reinforcement Learning (PPO)** and an **analytical surrogate CFD model** to discover optimal car configurations that minimize drag and maximize downforce. The project includes an interactive **Visual Wind Tunnel Dashboard** built with Flask.

## Features

- **PPO Agent** — Proximal Policy Optimization with Gaussian policy, GAE advantages, and tuned hyperparameters
- **Surrogate CFD Model** — Analytical aerodynamic model capturing wing angles, ground effect, diffuser, and gurney flap trade-offs
- **Visual Wind Tunnel** — Interactive Flask dashboard with real-time parameter sliders, gauges, and charts
- **Configuration Comparison** — Side-by-side view of Baseline vs RL-Optimized vs Custom parameters
- **Training Visualization** — Loss curves, reward plots, and aerodynamic coefficient tracking

## Project Structure

```
aeroRL/
├── main.py                     # CLI entry point (train, evaluate, dashboard)
├── config/
│   └── config.yaml             # Hyperparameters and environment settings
├── environment/
│   └── f1_env.py               # Gymnasium environment with surrogate CFD
├── rl/
│   ├── model.py                # TensorFlow actor-critic networks
│   ├── agent.py                # PPO agent (action selection, updates)
│   └── train.py                # Training loop with logging & checkpoints
├── webapp/
│   ├── app.py                  # Flask dashboard application
│   ├── surrogate.py            # Standalone surrogate CFD model
│   └── templates/              # HTML templates (Jinja2)
│       ├── base.html           # Dark-themed base layout
│       ├── dashboard.html      # Home: training summary + metrics
│       ├── wind_tunnel.html    # Interactive parameter exploration
│       ├── compare.html        # Baseline vs RL vs custom
│       └── training.html       # Training curves viewer
├── utils/
│   ├── logger.py               # Training logger with plots & TensorBoard
│   └── replay_buffer.py        # On-policy rollout buffer with GAE
├── cfd/                        # CFD simulation scripts (OpenFOAM, optional)
├── results/                    # Training outputs (models, logs, plots)
└── requirements.txt
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ani3h/aeroRL.git
cd aeroRL
```

### 2. Set Up Environment

```bash
# Using conda (recommended)
conda create -n aeroRL python=3.9
conda activate aeroRL
pip install -r requirements.txt

# Or using venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the RL Model

```bash
# Quick test (~30 seconds)
python main.py train --total-timesteps 10000

# Medium run (~5 minutes)
python main.py train --total-timesteps 100000

# Full training (~50 minutes, best results)
python main.py train --total-timesteps 1000000
```

Training saves:
- Model checkpoints → `results/models/`
- Training plots → `results/logs/<run-id>/plots/`
- Metrics CSV → `results/logs/<run-id>/`

### 4. Evaluate the Trained Model

```bash
python main.py evaluate --model-path results/models/best --episodes 20
```

### 5. Launch the Wind Tunnel Dashboard

```bash
python main.py dashboard
```

Open **http://localhost:5000** in your browser. The dashboard includes:

| Page | Description |
|------|-------------|
| **Dashboard** | Training summary, baseline vs RL metrics, quick links |
| **Wind Tunnel** | 8 parameter sliders with real-time Cd/Downforce/Efficiency gauges |
| **Compare** | Side-by-side Baseline vs RL vs Custom with bar charts |
| **Training** | View saved training plots (loss, reward, aero coefficients) |

## Configuration

All hyperparameters are in `config/config.yaml`:

| Section | Key Parameters |
|---------|---------------|
| `rl.learning_rate` | 0.0001 (tuned for stability) |
| `rl.policy_network.hidden_layers` | [512, 256] |
| `rl.training.batch_size` | 128 |
| `rl.clip_ratio` | 0.15 |
| `rl.entropy_coef` | 0.03 |
| `environment.rewards` | Multi-component normalized reward function |

## Architecture

```
┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
│  PPO Agent   │────▶│  F1 Aero Env      │────▶│  Surrogate   │
│  (Actor +    │     │  (Gymnasium)      │     │  CFD Model   │
│   Critic)    │◀────│  8-dim obs/action │◀────│  (Analytical)│
└──────────────┘     └───────────────────┘     └──────────────┘
       │                                              │
       ▼                                              ▼
┌──────────────┐                             ┌──────────────┐
│  Checkpoints │                             │  Flask       │
│  & Logs      │                             │  Dashboard   │
└──────────────┘                             └──────────────┘
```


