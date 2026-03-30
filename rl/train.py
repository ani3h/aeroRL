"""
Training Loop for AeroRL

Handles the full train → evaluate → log → checkpoint cycle for PPO
on the F1 aerodynamics environment.
"""

import os
import sys
import time
import yaml
import logging
import argparse
import numpy as np
from typing import Dict, Optional, Any

import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.f1_env import F1AeroEnv
from rl.agent import PPOAgent
from utils.logger import AeroRLLogger

logger = logging.getLogger(__name__)


class Trainer:
    """
    PPO trainer managing the full training pipeline.

    Workflow per iteration:
      1. Collect `rollout_steps` transitions in the environment
      2. Compute GAE advantages
      3. Run `n_epochs` of PPO updates over minibatches
      4. Log metrics, optionally evaluate, save checkpoints
    """

    def __init__(
        self,
        config: dict,
        config_path: str = "config/config.yaml",
        model_dir: str = "results/models",
        log_tensorboard: bool = True,
    ):
        self.config = config
        self.model_dir = model_dir

        rl_cfg = config.get("rl", {})
        training_cfg = rl_cfg.get("training", {})

        # Training hyperparameters
        self.total_timesteps = training_cfg.get("total_timesteps", 1_000_000)
        self.rollout_steps = 2048  # transitions per rollout
        self.save_freq = training_cfg.get("save_frequency", 50_000)
        self.eval_freq = training_cfg.get("eval_frequency", 10_000)
        self.seed = training_cfg.get("random_seed", 42)

        # Set seeds
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Create environment
        self.env = F1AeroEnv(config_path=config_path, max_steps=200)
        self.eval_env = F1AeroEnv(config_path=config_path, max_steps=200)

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        # Create agent
        self.agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, config=config)

        # Create logger
        try:
            self.aero_logger = AeroRLLogger(
                config_path=config_path,
                use_tensorboard=log_tensorboard,
                use_matplotlib=True,
            )
        except Exception as e:
            logger.warning(f"Could not create AeroRLLogger: {e}. Continuing without.")
            self.aero_logger = None

        # Tracking
        self.total_steps = 0
        self.episode_count = 0
        self.best_eval_efficiency = -float("inf")

        os.makedirs(model_dir, exist_ok=True)
        logger.info("Trainer initialized")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Run the full training loop. Returns summary metrics."""
        logger.info(f"Starting training for {self.total_timesteps} timesteps")
        start_time = time.time()

        iteration = 0

        while self.total_steps < self.total_timesteps:
            iteration += 1

            # 1. Collect rollout
            rollout_info = self._collect_rollout()

            # 2. PPO update
            update_metrics = self.agent.update()

            # 3. Logging
            self._log_iteration(iteration, rollout_info, update_metrics)

            # 4. Periodic evaluation
            if self.total_steps % self.eval_freq < self.rollout_steps:
                eval_metrics = self.evaluate(n_episodes=5)
                self._log_evaluation(eval_metrics)

                # Save best model
                mean_eff = eval_metrics["mean_efficiency"]
                if mean_eff > self.best_eval_efficiency:
                    self.best_eval_efficiency = mean_eff
                    self.agent.save(os.path.join(self.model_dir, "best"))
                    logger.info(f"New best model saved (efficiency={mean_eff:.4f})")

            # 5. Periodic checkpoint
            if self.total_steps % self.save_freq < self.rollout_steps:
                self.agent.save(
                    os.path.join(self.model_dir, f"checkpoint_{self.total_steps}")
                )

        elapsed = time.time() - start_time
        fps = self.total_steps / elapsed if elapsed > 0 else 0

        summary = {
            "total_timesteps": self.total_steps,
            "total_episodes": self.episode_count,
            "elapsed_time_s": elapsed,
            "fps": fps,
            "best_eval_efficiency": self.best_eval_efficiency,
        }

        logger.info(f"Training complete: {self.total_steps} steps in {elapsed:.1f}s "
                     f"({fps:.0f} fps)")
        logger.info(f"Best evaluation efficiency: {self.best_eval_efficiency:.4f}")

        # Final save
        self.agent.save(os.path.join(self.model_dir, "final"))

        # Close logger
        if self.aero_logger:
            self.aero_logger.close()

        return summary

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self) -> Dict[str, Any]:
        """
        Collect `rollout_steps` transitions using the current policy.
        """
        self.agent.create_buffer(self.rollout_steps)

        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_rewards = []
        episode_lengths = []
        episode_efficiencies = []

        for step in range(self.rollout_steps):
            action, value, log_prob = self.agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.agent.buffer.add(obs, action, reward, value, log_prob, done)

            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_efficiencies.append(info.get("efficiency", 0))
                self.episode_count += 1

                # Log episode
                if self.aero_logger:
                    self.aero_logger.log_episode(
                        episode_num=self.episode_count,
                        timestep=self.total_steps,
                        total_reward=episode_reward,
                        episode_length=episode_length,
                        drag_coef=info.get("cd", 0),
                        downforce_coef=info.get("downforce", 0),
                    )

                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

        # Compute last value for GAE bootstrap
        _, last_value, _ = self.agent.select_action(obs)
        self.agent.buffer.compute_returns_and_advantages(
            last_value=last_value, last_done=False
        )

        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "mean_efficiency": np.mean(episode_efficiencies) if episode_efficiencies else 0.0,
            "n_episodes": len(episode_rewards),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """Run evaluation episodes with deterministic policy."""
        rewards = []
        lengths = []
        efficiencies = []
        cds = []
        downforces = []

        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            total_reward = 0.0
            length = 0
            done = False

            while not done:
                action, _, _ = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
                length += 1

            rewards.append(total_reward)
            lengths.append(length)
            efficiencies.append(info.get("efficiency", 0))
            cds.append(info.get("cd", 0))
            downforces.append(info.get("downforce", 0))

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "mean_efficiency": np.mean(efficiencies),
            "mean_cd": np.mean(cds),
            "mean_downforce": np.mean(downforces),
            "best_efficiency": max(efficiencies),
        }

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_iteration(
        self, iteration: int, rollout_info: Dict, update_metrics: Dict
    ):
        """Log metrics for one training iteration."""
        logger.info(
            f"Iter {iteration} | Steps: {self.total_steps} | "
            f"Episodes: {rollout_info['n_episodes']} | "
            f"MeanReward: {rollout_info['mean_reward']:.2f} | "
            f"MeanEff: {rollout_info['mean_efficiency']:.4f} | "
            f"Loss: {update_metrics.get('loss', 0):.4f}"
        )

        if self.aero_logger:
            step = self.total_steps
            self.aero_logger.log_training_metrics(
                step=step,
                loss=update_metrics.get("loss", 0),
                value_loss=update_metrics.get("value_loss"),
                policy_loss=update_metrics.get("policy_loss"),
                entropy=update_metrics.get("entropy"),
                learning_rate=self.agent.learning_rate,
            )

    def _log_evaluation(self, eval_metrics: Dict):
        """Log evaluation results."""
        logger.info(
            f"EVAL @ step {self.total_steps}: "
            f"Reward={eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f} | "
            f"Efficiency={eval_metrics['mean_efficiency']:.4f} | "
            f"Cd={eval_metrics['mean_cd']:.4f} | "
            f"Downforce={eval_metrics['mean_downforce']:.4f}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train AeroRL PPO agent")

    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Override total training timesteps"
    )
    parser.add_argument(
        "--model-dir", type=str, default="results/models",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true",
        help="Disable TensorBoard logging"
    )
    parser.add_argument(
        "--eval-only", type=str, default=None,
        help="Path to a saved model to evaluate (skip training)"
    )

    args = parser.parse_args()

    # Load config
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        config = {}

    # Override total timesteps if specified
    if args.total_timesteps is not None:
        config.setdefault("rl", {}).setdefault("training", {})["total_timesteps"] = args.total_timesteps

    trainer = Trainer(
        config=config,
        config_path=config_path,
        model_dir=args.model_dir,
        log_tensorboard=not args.no_tensorboard,
    )

    if args.eval_only:
        # Evaluation-only mode
        trainer.agent.load(args.eval_only)
        eval_results = trainer.evaluate(n_episodes=20)
        print("\n=== Evaluation Results ===")
        for k, v in eval_results.items():
            print(f"  {k}: {v:.4f}")
    else:
        # Training mode
        summary = trainer.train()
        print("\n=== Training Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
