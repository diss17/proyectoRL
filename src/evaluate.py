"""Evaluación greedy del nivel completo y de etapas concretas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .agents.ddqn import DDQNAgent
from .env.kirby_env import KirbyEnv
from .utils import load_config, resolve_device, set_seed

SPAWN = "spawn"
BOSS_GOAL = "boss"


def evaluate_policy(
    agent: DDQNAgent,
    env: KirbyEnv,
    *,
    stage: str = SPAWN,
    goal: str | None = BOSS_GOAL,
    episodes: int = 10,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Corre `episodes` episodios sin exploración y reporta victorias."""
    wins = 0
    returns: list[float] = []
    lengths: list[int] = []

    for _ in range(episodes):
        obs, _ = env.reset(options={"stage": stage, "goal": goal})
        done = False
        ep_return = 0.0
        steps = 0
        info: dict[str, Any] = {}

        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            steps += 1
            done = terminated or truncated
            if max_steps is not None and steps >= max_steps:
                done = True

        wins += int(bool(info.get("won", False)))
        returns.append(ep_return)
        lengths.append(steps)

    return {
        "stage": stage,
        "goal": goal or "",
        "episodes": episodes,
        "wins": wins,
        "win_rate": wins / episodes if episodes else float("nan"),
        "return_mean": float(np.mean(returns)) if returns else float("nan"),
        "return_std": float(np.std(returns)) if returns else float("nan"),
        "length_mean": float(np.mean(lengths)) if lengths else float("nan"),
    }


def evaluate_full_level(
    agent: DDQNAgent, env: KirbyEnv, episodes: int = 100
) -> dict[str, Any]:
    return evaluate_policy(
        agent, env, stage=SPAWN, goal=BOSS_GOAL, episodes=episodes
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa un checkpoint con epsilon = 0")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--stage", default=SPAWN, help="ID de etapa (spawn|mid|door|arena)")
    parser.add_argument("--goal", default=BOSS_GOAL, help="'boss' o un ID de etapa")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_config(args.config) if args.config else payload["config"]
    set_seed(int(cfg.get("seed", 0)))

    device = resolve_device(args.device)
    env = KirbyEnv(cfg["env"])
    agent = DDQNAgent(env.observation_space.shape, env.action_space.n, cfg["agent"], device=device)
    agent.load(args.checkpoint)

    results = evaluate_policy(
        agent,
        env,
        stage=args.stage,
        goal=args.goal,
        episodes=args.episodes,
    )
    env.close()

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
