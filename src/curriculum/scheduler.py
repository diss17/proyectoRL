"""Curriculum Scheduler basado en secuencias.

Generaliza el currículo a cualquier orden de etapas:

- `sequence[k]` es la etapa de la posición k (frontera actual).
- El muestreo mixto reanuda en `sequence[j]` con j < k, es decir, en etapas ya
  superadas, sea cual sea la dirección del currículo.
- Solo cuentan para la ventana los episodios con epsilon <= `epsilon_gate`, para
  que el 85 % no se alcance con exploración alta.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Sequence


class CurriculumScheduler:
    def __init__(
        self,
        sequence: Sequence[str],
        *,
        window: int = 50,
        threshold: float | None = 0.85,
        replay_prob: float = 0.20,
        epsilon_gate: float = 0.10,
        min_episodes_per_stage: int | None = None,
        fixed_frames_per_stage: int | None = None,
    ) -> None:
        if len(sequence) < 1:
            raise ValueError("La secuencia debe tener al menos una etapa")
        if not 0.0 <= replay_prob <= 1.0:
            raise ValueError("replay_prob debe estar en [0, 1]")

        self.stage_ids: list[str] = list(sequence)
        self.window = int(window)
        self.threshold = threshold
        self.replay_prob = float(replay_prob)
        self.epsilon_gate = float(epsilon_gate)
        self.min_episodes = (
            self.window if min_episodes_per_stage is None else int(min_episodes_per_stage)
        )
        self.fixed_frames_per_stage = fixed_frames_per_stage
        self.adaptive = threshold is not None

        self.position = 0
        self._history: dict[str, deque[float]] = {
            s: deque(maxlen=self.window) for s in self.stage_ids
        }
        self.stage_episodes = {s: 0 for s in self.stage_ids}
        self.stage_wins = {s: 0 for s in self.stage_ids}
        # Frames de emulador acumulados por etapa (presupuesto real de experiencia).
        self.stage_frames = {s: 0 for s in self.stage_ids}
        self.stage_ignored = {s: 0 for s in self.stage_ids}
        self.transitions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Frontera y selección
    # ------------------------------------------------------------------ #

    @property
    def current_stage(self) -> str:
        return self.stage_ids[self.position]

    @property
    def is_last_stage(self) -> bool:
        return self.position >= len(self.stage_ids) - 1

    def select_stage(self, rng: random.Random | None = None) -> str:
        """Etapa donde iniciará el próximo episodio (distribución p0(s))."""
        r = rng.random() if rng is not None else random.random()
        if self.position > 0 and r < self.replay_prob:
            chooser = rng if rng is not None else random
            return self.stage_ids[chooser.randint(0, self.position - 1)]
        return self.current_stage

    # ------------------------------------------------------------------ #
    # Registro
    # ------------------------------------------------------------------ #

    def add_frames(self, stage: str, frames: int) -> None:
        """Suma frames de emulador (no pasos de agente) a la etapa."""
        if stage in self.stage_frames:
            self.stage_frames[stage] += int(frames)

    def record_episode(self, stage: str, won: bool, epsilon: float | None = None) -> bool:
        """Registra un episodio. Devuelve si entró en la ventana de decisión.

        Los episodios con epsilon > `epsilon_gate` se cuentan pero no se usan
        para decidir la transición.
        """
        if stage not in self.stage_episodes:
            raise KeyError(f"Etapa fuera de la secuencia: {stage!r}")

        self.stage_episodes[stage] += 1

        if epsilon is not None and epsilon > self.epsilon_gate:
            self.stage_ignored[stage] += 1
            return False

        self._history[stage].append(1.0 if won else 0.0)
        self.stage_wins[stage] += int(bool(won))
        return True

    def win_rate(self, stage: str) -> float:
        h = self._history.get(stage)
        if not h:
            return 0.0
        return sum(h) / len(h)

    def window_size(self, stage: str) -> int:
        return len(self._history.get(stage, ()))

    def is_window_full(self, stage: str) -> bool:
        return self.window_size(stage) >= self.min_episodes

    # ------------------------------------------------------------------ #
    # Transición
    # ------------------------------------------------------------------ #

    def should_advance(self) -> bool:
        if self.is_last_stage:
            return False

        stage = self.current_stage

        if self.fixed_frames_per_stage is not None:
            return self.stage_frames[stage] >= self.fixed_frames_per_stage

        if not self.adaptive:
            return False

        return self.is_window_full(stage) and self.win_rate(stage) >= float(self.threshold)

    def maybe_advance(self) -> str | None:
        """Avanza de posición si corresponde. Devuelve la nueva etapa o None."""
        if not self.should_advance():
            return None

        stage = self.current_stage
        if self.adaptive:
            criterion = f"win_rate={self.win_rate(stage):.2%}"
        else:
            criterion = f"frames={self.stage_frames[stage]}"

        self.position += 1
        new_stage = self.current_stage
        self.transitions.append(
            {
                "from": stage,
                "to": new_stage,
                "position": self.position,
                "episodes": self.stage_episodes[stage],
                "criterion": criterion,
            }
        )
        return new_stage

    # ------------------------------------------------------------------ #
    # Serialización
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        return {
            "stage_ids": list(self.stage_ids),
            "window": self.window,
            "threshold": self.threshold,
            "replay_prob": self.replay_prob,
            "epsilon_gate": self.epsilon_gate,
            "min_episodes": self.min_episodes,
            "fixed_frames_per_stage": self.fixed_frames_per_stage,
            "position": self.position,
            "history": {s: list(h) for s, h in self._history.items()},
            "stage_episodes": dict(self.stage_episodes),
            "stage_wins": dict(self.stage_wins),
            "stage_frames": dict(self.stage_frames),
            "stage_ignored": dict(self.stage_ignored),
            "transitions": list(self.transitions),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.stage_ids = list(state["stage_ids"])
        self.window = int(state["window"])
        self.threshold = state["threshold"]
        self.replay_prob = float(state["replay_prob"])
        self.epsilon_gate = float(state["epsilon_gate"])
        self.min_episodes = int(state["min_episodes"])
        self.fixed_frames_per_stage = state["fixed_frames_per_stage"]
        self.adaptive = self.threshold is not None
        self.position = int(state["position"])
        self._history = {
            s: deque(h, maxlen=self.window) for s, h in state["history"].items()
        }
        self.stage_episodes = dict(state["stage_episodes"])
        self.stage_wins = dict(state["stage_wins"])
        self.stage_frames = dict(state["stage_frames"])
        self.stage_ignored = dict(state.get("stage_ignored", self.stage_ignored))
        self.transitions = list(state["transitions"])

    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        mode = "adaptativo" if self.adaptive else "duración fija"
        rows = [f"posición {self.position}/{len(self.stage_ids) - 1} ({mode})"]
        for position, stage in enumerate(self.stage_ids):
            mark = "->" if position == self.position else "  "
            rows.append(
                f" {mark} [{position}] {stage:<6} ep={self.stage_episodes[stage]:<7} "
                f"win={self.win_rate(stage):6.2%} "
                f"ventana={self.window_size(stage)}/{self.window} "
                f"ignorados={self.stage_ignored[stage]} "
                f"frames={self.stage_frames[stage]}"
            )
        return "\n".join(rows)


def build_scheduler(cfg: dict[str, Any], sequence: Sequence[str]) -> CurriculumScheduler:
    return CurriculumScheduler(
        sequence,
        window=cfg.get("window", 50),
        threshold=cfg.get("threshold", 0.85),
        replay_prob=cfg.get("replay_prob", 0.20),
        epsilon_gate=cfg.get("epsilon_gate", 0.10),
        min_episodes_per_stage=cfg.get("min_episodes_per_stage"),
        fixed_frames_per_stage=cfg.get("fixed_frames_per_stage"),
    )
