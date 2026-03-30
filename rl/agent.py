"""
PPO (Proximal Policy Optimisation) Agent

Implements the PPO-Clip algorithm with Generalised Advantage Estimation (GAE).
Wraps actor-critic networks and handles the full select → store → update cycle.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple, Any

from rl.model import ActorNetwork, CriticNetwork, build_actor_critic
from utils.replay_buffer import RolloutBuffer

logger = logging.getLogger(__name__)


class PPOAgent:
    """
    Proximal Policy Optimization (PPO-Clip) agent.

    Hyperparameters are loaded from the `rl` section of config.yaml.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: dict,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config

        # RL hyperparameters
        rl_cfg = config.get("rl", {})
        self.learning_rate = rl_cfg.get("learning_rate", 3e-4)
        self.gamma = rl_cfg.get("discount_factor", 0.99)
        self.gae_lambda = rl_cfg.get("gae_lambda", 0.95)
        self.clip_ratio = rl_cfg.get("clip_ratio", 0.2)
        self.entropy_coef = rl_cfg.get("entropy_coef", 0.01)
        self.value_coef = rl_cfg.get("value_coef", 0.5)
        self.max_grad_norm = rl_cfg.get("max_grad_norm", 0.5)

        training_cfg = rl_cfg.get("training", {})
        self.batch_size = training_cfg.get("batch_size", 64)
        self.n_epochs = training_cfg.get("epochs", 10)

        # Build actor-critic networks
        self.actor, self.critic = build_actor_critic(obs_dim, act_dim, config)

        # Optimisers
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        # Rollout buffer (size will be set externally before collecting)
        self.buffer: Optional[RolloutBuffer] = None

        logger.info(
            f"PPOAgent created: obs_dim={obs_dim}, act_dim={act_dim}, "
            f"lr={self.learning_rate}, gamma={self.gamma}, "
            f"clip={self.clip_ratio}, epochs={self.n_epochs}"
        )

    # ------------------------------------------------------------------
    # Buffer Management
    # ------------------------------------------------------------------

    def create_buffer(self, buffer_size: int):
        """Create a new rollout buffer for the next collection phase."""
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    # ------------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select an action for a single observation.

        Returns:
            (action, value, log_prob)
        """
        obs_t = tf.convert_to_tensor(obs.reshape(1, -1), dtype=tf.float32)

        # Value
        value = self.critic(obs_t).numpy().item()

        if deterministic:
            mean = self.actor(obs_t).numpy().flatten()
            action = np.clip(mean, -1.0, 1.0)
            # Log prob of the mean is 0 under Gaussian (approximately)
            log_prob = 0.0
        else:
            action, log_prob_arr = self.actor.sample_action(obs)
            log_prob = float(log_prob_arr.sum()) if not np.isscalar(log_prob_arr) else float(log_prob_arr)

        return action, value, log_prob

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """
        Run PPO update using data in self.buffer.

        Returns dict of loss metrics.
        """
        if self.buffer is None or self.buffer.size == 0:
            logger.warning("No data in buffer, skipping update")
            return {}

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size, shuffle=True):
                metrics = self._update_step(batch)
                total_policy_loss += metrics["policy_loss"]
                total_value_loss += metrics["value_loss"]
                total_entropy += metrics["entropy"]
                total_loss += metrics["loss"]
                n_updates += 1

        n_updates = max(n_updates, 1)
        result = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "loss": total_loss / n_updates,
        }

        logger.info(
            f"PPO update: loss={result['loss']:.4f}, "
            f"policy={result['policy_loss']:.4f}, "
            f"value={result['value_loss']:.4f}, "
            f"entropy={result['entropy']:.4f}"
        )

        return result

    def _update_step(self, batch) -> Dict[str, float]:
        """Single gradient step on a minibatch."""
        obs = tf.convert_to_tensor(batch.observations, dtype=tf.float32)
        actions = tf.convert_to_tensor(batch.actions, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(batch.old_log_probs, dtype=tf.float32)
        advantages = tf.convert_to_tensor(batch.advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(batch.returns, dtype=tf.float32)

        # ----- Actor (Policy) loss -----
        with tf.GradientTape() as actor_tape:
            new_log_probs, entropy = self.actor.evaluate_actions(obs, actions)

            # PPO clipped objective
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            )
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            # Entropy bonus (encourages exploration)
            entropy_loss = -tf.reduce_mean(entropy)

            actor_loss = policy_loss + self.entropy_coef * entropy_loss

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        if self.max_grad_norm > 0:
            actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables)
        )

        # ----- Critic (Value) loss -----
        with tf.GradientTape() as critic_tape:
            values = self.critic(obs)
            value_loss = tf.reduce_mean(tf.square(returns - values))
            critic_loss = self.value_coef * value_loss

        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        if self.max_grad_norm > 0:
            critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm)
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables)
        )

        return {
            "policy_loss": float(policy_loss.numpy()),
            "value_loss": float(value_loss.numpy()),
            "entropy": float(tf.reduce_mean(entropy).numpy()),
            "loss": float(actor_loss.numpy() + critic_loss.numpy()),
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save actor and critic weights."""
        os.makedirs(path, exist_ok=True)
        actor_path = os.path.join(path, "actor.weights.h5")
        critic_path = os.path.join(path, "critic.weights.h5")
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        logger.info(f"Agent saved to {path}")

    def load(self, path: str):
        """Load actor and critic weights."""
        actor_path = os.path.join(path, "actor.weights.h5")
        critic_path = os.path.join(path, "critic.weights.h5")
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        logger.info(f"Agent loaded from {path}")
