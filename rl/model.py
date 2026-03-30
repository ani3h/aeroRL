"""
Actor-Critic Neural Network Models for PPO

Implements TensorFlow-based policy (actor) and value (critic) networks
for Proximal Policy Optimisation. Architecture is configurable via config.yaml.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple, Optional

import logging

logger = logging.getLogger(__name__)


class ActorNetwork(keras.Model):
    """
    Gaussian policy network.

    Outputs the mean of a diagonal Gaussian distribution over continuous actions.
    Log standard deviations are learnable per-action parameters (state-independent).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int] = (256, 256),
        activation: str = "tanh",
        log_std_init: float = -0.5,
        name: str = "actor",
    ):
        super().__init__(name=name)

        act_fn = _get_activation(activation)

        # Build MLP
        self.net = keras.Sequential(name="actor_mlp")
        for i, h in enumerate(hidden_sizes):
            self.net.add(layers.Dense(h, activation=act_fn, name=f"fc{i}",
                                      kernel_initializer="orthogonal"))
        self.net.add(layers.Dense(act_dim, activation=None, name="mean",
                                   kernel_initializer=keras.initializers.Orthogonal(gain=0.01)))

        # Learnable log_std (state-independent)
        self.log_std = tf.Variable(
            tf.fill([act_dim], log_std_init), trainable=True, name="log_std"
        )

        # Build the model by calling it once
        dummy = tf.zeros((1, obs_dim))
        self(dummy)

        logger.info(f"ActorNetwork created: obs_dim={obs_dim}, act_dim={act_dim}, "
                     f"hidden={hidden_sizes}, activation={activation}")

    @tf.function
    def call(self, obs: tf.Tensor) -> tf.Tensor:
        """Return action means given observations."""
        return self.net(obs)

    def get_distribution(self, obs: tf.Tensor):
        """Return a TFP-like object (mean, std) for sampling and log_prob."""
        mean = self(obs)
        std = tf.exp(self.log_std)
        return mean, std

    def sample_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample an action and return (action, log_prob)."""
        obs_t = tf.convert_to_tensor(obs.reshape(1, -1), dtype=tf.float32)
        mean, std = self.get_distribution(obs_t)

        # Reparameterised sample
        noise = tf.random.normal(tf.shape(mean))
        action = mean + std * noise

        # Log probability under diagonal Gaussian
        log_prob = self._log_prob(action, mean, std)

        # Clip action to [-1, 1]
        action_clipped = tf.clip_by_value(action, -1.0, 1.0)

        return action_clipped.numpy().flatten(), log_prob.numpy().flatten()

    @staticmethod
    def _log_prob(action: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
        """Diagonal Gaussian log probability."""
        var = std ** 2
        log_std = tf.math.log(std)
        lp = -0.5 * (
            tf.reduce_sum(((action - mean) ** 2) / var, axis=-1)
            + tf.reduce_sum(2.0 * log_std, axis=-1)
            + tf.cast(tf.shape(action)[-1], tf.float32) * tf.math.log(2.0 * np.pi)
        )
        return lp

    def evaluate_actions(self, obs: tf.Tensor, actions: tf.Tensor):
        """Compute log_prob and entropy for given obs-action pairs."""
        mean, std = self.get_distribution(obs)
        log_prob = self._log_prob(actions, mean, std)

        # Entropy of diagonal Gaussian
        entropy = 0.5 * tf.reduce_sum(
            tf.math.log(2.0 * np.pi * np.e * std ** 2), axis=-1
        )
        return log_prob, entropy


class CriticNetwork(keras.Model):
    """
    State value function V(s).

    Outputs a single scalar value estimate for each observation.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: List[int] = (256, 256),
        activation: str = "tanh",
        name: str = "critic",
    ):
        super().__init__(name=name)

        act_fn = _get_activation(activation)

        self.net = keras.Sequential(name="critic_mlp")
        for i, h in enumerate(hidden_sizes):
            self.net.add(layers.Dense(h, activation=act_fn, name=f"fc{i}",
                                      kernel_initializer="orthogonal"))
        self.net.add(layers.Dense(1, activation=None, name="value",
                                   kernel_initializer=keras.initializers.Orthogonal(gain=1.0)))

        # Build
        dummy = tf.zeros((1, obs_dim))
        self(dummy)

        logger.info(f"CriticNetwork created: obs_dim={obs_dim}, "
                     f"hidden={hidden_sizes}, activation={activation}")

    @tf.function
    def call(self, obs: tf.Tensor) -> tf.Tensor:
        """Return value estimate (scalar per observation)."""
        return tf.squeeze(self.net(obs), axis=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_activation(name: str):
    """Map activation name string to a Keras activation."""
    activations = {
        "tanh": "tanh",
        "relu": "relu",
        "elu": "elu",
        "sigmoid": "sigmoid",
        "selu": "selu",
    }
    if name.lower() not in activations:
        logger.warning(f"Unknown activation '{name}', falling back to tanh.")
        return "tanh"
    return activations[name.lower()]


def build_actor_critic(
    obs_dim: int,
    act_dim: int,
    config: dict,
) -> Tuple[ActorNetwork, CriticNetwork]:
    """Factory to build actor-critic pair from a config dict."""
    rl_cfg = config.get("rl", {})
    policy_cfg = rl_cfg.get("policy_network", {})
    value_cfg = rl_cfg.get("value_network", {})

    actor = ActorNetwork(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=policy_cfg.get("hidden_layers", [256, 256]),
        activation=policy_cfg.get("activation", "tanh"),
    )

    critic = CriticNetwork(
        obs_dim=obs_dim,
        hidden_sizes=value_cfg.get("hidden_layers", [256, 256]),
        activation=value_cfg.get("activation", "tanh"),
    )

    return actor, critic
