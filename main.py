"""
AeroRL: AI-Powered F1 Aerodynamics Optimization — Main Entry Point

Usage:
    python main.py train [--total-timesteps N] [--config path]
    python main.py evaluate --model-path path [--episodes N]
    python main.py cfd-validate --model-path path
"""

import os
import sys
import yaml
import argparse
import logging

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Configure root logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("aerorl")


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    logger.warning(f"Config not found at {config_path}, using defaults.")
    return {}


# ── Commands ────────────────────────────────────────────────────────────────


def cmd_train(args):
    """Train the PPO agent on the F1 aero environment."""
    from rl.train import Trainer

    config = load_config(args.config)

    # Override timesteps if provided
    if args.total_timesteps is not None:
        config.setdefault("rl", {}).setdefault("training", {})[
            "total_timesteps"
        ] = args.total_timesteps

    trainer = Trainer(
        config=config,
        config_path=args.config,
        model_dir=args.model_dir,
        log_tensorboard=not args.no_tensorboard,
    )

    summary = trainer.train()

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)


def cmd_evaluate(args):
    """Evaluate a trained model."""
    from environment.f1_env import F1AeroEnv
    from rl.agent import PPOAgent

    config = load_config(args.config)

    env = F1AeroEnv(config_path=args.config, max_steps=200)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, config=config)
    agent.load(args.model_path)

    rewards = []
    efficiencies = []
    best_params = None
    best_efficiency = -float("inf")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        eff = info.get("efficiency", 0)
        rewards.append(total_reward)
        efficiencies.append(eff)

        if eff > best_efficiency:
            best_efficiency = eff
            best_params = info.get("parameters", {})

        print(f"Episode {ep + 1}/{args.episodes}: "
              f"reward={total_reward:.2f}, efficiency={eff:.4f}")

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    import numpy as np
    print(f"  Mean Reward:      {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"  Mean Efficiency:  {np.mean(efficiencies):.4f}")
    print(f"  Best Efficiency:  {best_efficiency:.4f}")
    if best_params:
        print("  Best Parameters:")
        for k, v in best_params.items():
            print(f"    {k}: {v:.2f}")
    print("=" * 60)


def cmd_cfd_validate(args):
    """Validate best RL parameters against CFD simulation."""
    from environment.f1_env import F1AeroEnv
    from rl.agent import PPOAgent
    from cfd.run_cfd import run_single_simulation, setup_directories

    config = load_config(args.config)

    # Load agent and run one greedy episode to get best params
    env = F1AeroEnv(config_path=args.config, max_steps=200)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, config=config)
    agent.load(args.model_path)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    best_params = info.get("parameters", {})
    print("\nBest parameters from RL agent:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.2f}")

    # Run CFD simulation with those parameters
    print("\nRunning CFD validation simulation...")
    dirs = setup_directories()
    sim_type = args.sim_type

    try:
        result = run_single_simulation(best_params, dirs, sim_type)
        if result.get("success"):
            aero = result.get("aero_metrics", {})
            print("\n" + "=" * 60)
            print("  CFD VALIDATION RESULTS")
            print("=" * 60)
            print(f"  Cd (CFD):         {aero.get('cd', 'N/A')}")
            print(f"  Cl (CFD):         {aero.get('cl', 'N/A')}")
            print(f"  Efficiency (CFD): {aero.get('efficiency', 'N/A')}")
            print("=" * 60)
        else:
            print(f"\nCFD simulation failed: {result.get('error', 'unknown')}")
    except Exception as e:
        logger.error(f"CFD validation failed: {e}")
        print(f"\nCFD validation could not be run: {e}")
        print("This is expected if OpenFOAM / SimScale is not installed locally.")


# ── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AeroRL: AI-Powered F1 Aerodynamics Optimization"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train the PPO agent")
    train_parser.add_argument(
        "--config", type=str, default="config/config.yaml"
    )
    train_parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Override total training timesteps"
    )
    train_parser.add_argument(
        "--model-dir", type=str, default="results/models"
    )
    train_parser.add_argument(
        "--no-tensorboard", action="store_true"
    )

    # --- evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument(
        "--config", type=str, default="config/config.yaml"
    )
    eval_parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to saved model directory"
    )
    eval_parser.add_argument(
        "--episodes", type=int, default=10
    )

    # --- cfd-validate ---
    cfd_parser = subparsers.add_parser(
        "cfd-validate", help="Validate RL policy with real CFD"
    )
    cfd_parser.add_argument("--config", type=str, default="config/config.yaml")
    cfd_parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to saved model directory"
    )
    cfd_parser.add_argument(
        "--sim-type", type=str, default="openfoam_local",
        choices=["openfoam_local", "openfoam_docker", "simscale_api"],
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Change working directory to project root so relative paths work
    os.chdir(PROJECT_ROOT)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "cfd-validate":
        cmd_cfd_validate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
