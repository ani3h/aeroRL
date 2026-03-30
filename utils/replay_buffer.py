"""
Rollout Buffer for On-Policy RL (PPO)

Stores trajectory data collected during rollouts. Computes GAE advantages
and discounted returns for PPO updates.
"""

import numpy as np
from typing import Generator, NamedTuple, Optional
import logging

logger = logging.getLogger(__name__)


class RolloutBatch(NamedTuple):
    """A named tuple holding a minibatch of rollout data."""
    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    values: np.ndarray


class RolloutBuffer:
    """
    Fixed-size buffer that stores one full rollout for on-policy training.

    Usage:
        buffer = RolloutBuffer(buffer_size, obs_dim, act_dim)
        for step in range(buffer_size):
            buffer.add(obs, action, reward, value, log_prob, done)
        buffer.compute_returns_and_advantages(last_value, last_done)
        for batch in buffer.get_batches(batch_size):
            # training step
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Pre-allocate arrays
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

        logger.info(f"RolloutBuffer created: size={buffer_size}, "
                     f"obs_dim={obs_dim}, act_dim={act_dim}, "
                     f"gamma={gamma}, gae_lambda={gae_lambda}")

    def reset(self):
        """Reset the buffer for a new rollout."""
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add a single transition to the buffer."""
        if self.pos >= self.buffer_size:
            logger.warning("Buffer overflow — resetting position to 0")
            self.pos = 0

        self.observations[self.pos] = obs.flatten()
        self.actions[self.pos] = action.flatten()
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob if np.isscalar(log_prob) else log_prob.sum()
        self.dones[self.pos] = float(done)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self, last_value: float, last_done: bool
    ):
        """
        Compute GAE (Generalised Advantage Estimation) and discounted returns.

        Must be called after the buffer is full and before sampling batches.
        """
        n = self.pos if not self.full else self.buffer_size

        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

        # Normalise advantages
        adv = self.advantages[:n]
        adv_std = adv.std()
        if adv_std > 1e-8:
            self.advantages[:n] = (adv - adv.mean()) / adv_std

    def get_batches(
        self, batch_size: int, shuffle: bool = True
    ) -> Generator[RolloutBatch, None, None]:
        """
        Yield minibatches from the buffer.

        Args:
            batch_size: Number of transitions per batch.
            shuffle: Whether to shuffle indices before batching.

        Yields:
            RolloutBatch named tuples.
        """
        n = self.pos if not self.full else self.buffer_size
        indices = np.arange(n)

        if shuffle:
            np.random.shuffle(indices)

        # Yield full batches
        start = 0
        while start + batch_size <= n:
            batch_indices = indices[start:start + batch_size]
            yield RolloutBatch(
                observations=self.observations[batch_indices],
                actions=self.actions[batch_indices],
                old_log_probs=self.log_probs[batch_indices],
                advantages=self.advantages[batch_indices],
                returns=self.returns[batch_indices],
                values=self.values[batch_indices],
            )
            start += batch_size

        # Yield remaining samples (if any)
        if start < n:
            batch_indices = indices[start:n]
            yield RolloutBatch(
                observations=self.observations[batch_indices],
                actions=self.actions[batch_indices],
                old_log_probs=self.log_probs[batch_indices],
                advantages=self.advantages[batch_indices],
                returns=self.returns[batch_indices],
                values=self.values[batch_indices],
            )

    @property
    def size(self) -> int:
        """Number of samples currently in the buffer."""
        return self.buffer_size if self.full else self.pos
