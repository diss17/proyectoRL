"""Replay buffer circular con almacenamiento en uint8 (bajo consumo de RAM)."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        n_actions: int,
    ) -> None:
        self.capacity = int(capacity)
        self.observation_shape = tuple(observation_shape)

        self.obs = np.zeros((capacity, *self.observation_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *self.observation_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.position = 0
        self.size = 0
        self.n_actions = n_actions

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self.position
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = float(bool(done))

        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if self.size < batch_size:
            raise ValueError(
                f"Buffer con {self.size} muestras; se piden {batch_size}."
            )
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }

    def __len__(self) -> int:
        return self.size
