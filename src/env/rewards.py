"""Función de recompensa.

La recompensa es deliberadamente ESPARSA: el currículo sustituye al reward
shaping invasivo de la línea base. `progress` existe solo para ablaciones y
debe permanecer en 0.0 en los tres grupos experimentales.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    win: float = 10.0
    death: float = -5.0
    damage: float = -0.5
    progress: float = 0.0
    timeout: float = 0.0

    @classmethod
    def from_config(cls, raw: dict | None) -> "RewardConfig":
        return cls(**(raw or {}))


def compute_reward(
    cfg: RewardConfig,
    *,
    won: bool = False,
    died: bool = False,
    damaged: bool = False,
    x_progress: float = 0.0,
    truncated: bool = False,
) -> float:
    """Recompensa escalar de un paso de agente."""
    reward = 0.0
    if won:
        reward += cfg.win
    if died:
        reward += cfg.death
    if damaged:
        reward += cfg.damage
    if cfg.progress and x_progress:
        reward += cfg.progress * float(x_progress)
    if truncated and not (won or died):
        reward += cfg.timeout
    return float(reward)


def describe(cfg: RewardConfig) -> str:
    return (
        f"win={cfg.win} death={cfg.death} damage={cfg.damage} "
        f"progress={cfg.progress} timeout={cfg.timeout}"
    )
