"""Política aleatoria en la arena: cuántos golpes recibe Whispy Woods por episodio.

Sirve para saber si existe señal de progreso contra el jefe que el agente podría
aprovechar. Whispy empieza con boss_hp = 6; se gana al llegar a 0.

Uso:
    python tools/probe_boss_damage.py [--episodes 60] [--config configs/arena_only.yaml]

Resultado de referencia (2026-11-08, arena_only.yaml, 60 episodios, semilla 0):
    0 golpes: 44 · 1 golpe: 11 · 2 golpes: 5 → 16/60 episodios dañan al jefe,
    media 0.35 golpes; ningún episodio pasa de 2 de los 6 necesarios.
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.kirby_env import KirbyEnv  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/arena_only.yaml"))
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = KirbyEnv(cfg["env"])
    rng = random.Random(args.seed)
    hits: list[int] = []
    for ep in range(args.episodes):
        _, info = env.reset(seed=args.seed * 1000 + ep, options={"stage": "arena", "goal": None})
        start = lowest = info.get("boss_hp")
        done = False
        while not done:
            _, _, term, trunc, info = env.step(rng.randrange(env.action_space.n))
            done = term or trunc
            if info.get("boss_hp") is not None:
                lowest = min(lowest, info["boss_hp"])
        hits.append(start - lowest)
    env.close()

    counts = dict(sorted(collections.Counter(hits).items()))
    damaged = sum(1 for h in hits if h > 0)
    print(f"Golpes a Whispy por episodio ({args.episodes} ep., política aleatoria): {counts}")
    print(f"Episodios con >= 1 golpe: {damaged}/{args.episodes} · media {sum(hits) / len(hits):.2f}")


if __name__ == "__main__":
    main()
