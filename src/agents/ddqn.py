"""Agente DDQN (Double DQN) sobre fotogramas apilados."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """Arquitectura tipo Nature DQN adaptada a 84x84 en escala de grises."""

    def __init__(self, in_channels: int, n_actions: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            flat = self.features(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flat, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class DDQNAgent:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        n_actions: int,
        cfg: dict[str, Any],
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.n_actions = int(n_actions)
        self.observation_shape = tuple(observation_shape)

        self.gamma = float(cfg.get("gamma", 0.997))
        self.batch_size = int(cfg.get("batch_size", 32))
        self.target_update_freq = int(cfg.get("target_update_freq", 1000))
        self.min_buffer = int(cfg.get("min_buffer_to_learn", 20000))
        self.grad_clip = float(cfg.get("grad_clip", 10.0))
        self.double_dqn = bool(cfg.get("double_dqn", True))

        self.net = QNetwork(self.observation_shape[0], self.n_actions).to(self.device)
        self.target = QNetwork(self.observation_shape[0], self.n_actions).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=float(cfg.get("lr", 1e-4))
        )
        self.buffer = ReplayBuffer(
            int(cfg.get("buffer_size", 20_000)),
            self.observation_shape,
            self.n_actions,
        )
        # Exploración y muestreo del buffer dependen de la semilla del experimento.
        self._rng = np.random.default_rng(int(seed))
        self.learn_steps = 0

    # ------------------------------------------------------------------ #

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        if epsilon > 0 and self._rng.random() < epsilon:
            return int(self._rng.integers(0, self.n_actions))

        with torch.no_grad():
            batch = torch.as_tensor(obs[None, ...], dtype=torch.uint8, device=self.device)
            q = self.net(batch.float() / 255.0)
        return int(q.argmax(dim=1).item())

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------ #

    def update(self) -> float | None:
        if len(self.buffer) < max(self.min_buffer, self.batch_size):
            return None

        batch = self.buffer.sample(self.batch_size, self._rng)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device) / 255.0
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device) / 255.0
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        q_values = self.net(obs).gather(1, actions).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target(next_obs).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.sync_target()

        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.net.state_dict())

    # ------------------------------------------------------------------ #

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "net": self.net.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learn_steps": self.learn_steps,
            "observation_shape": self.observation_shape,
            "n_actions": self.n_actions,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str | Path) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.net.load_state_dict(payload["net"])
        self.target.load_state_dict(payload.get("target", payload["net"]))
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.learn_steps = int(payload.get("learn_steps", 0))
        return payload
