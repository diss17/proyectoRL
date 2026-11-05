"""Wrapper de Gymnasium sobre PyBoy para Kirby's Dream Land.

El entorno no decide el currículo: recibe la etapa y la meta mediante
`reset(options={"stage": <id>, "goal": <goal>})`. La meta por defecto es la
definida por la etapa en `curriculum/stages.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..curriculum.stages import BOSS_GOAL, Goal, get_stage, stage_names
from .preprocessing import FrameStack, preprocess_frame
from .ram import RamAddresses, read_ram
from .rewards import RewardConfig, compute_reward


class KirbyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        self.rom_path = Path(cfg["rom_path"])
        self.saves_dir = Path(cfg.get("saves_dir", "saves"))
        self.frame_skip = int(cfg.get("frame_skip", 4))
        self.frame_stack_n = int(cfg.get("frame_stack", 4))
        self.screen_size = tuple(cfg.get("screen_size", (84, 84)))
        self.grayscale = bool(cfg.get("grayscale", True))
        self.max_steps = int(cfg.get("max_steps_per_episode", 4000))
        self.max_frames = int(cfg.get("max_frames_per_episode", 20000))
        self.strict_goal_check = bool(cfg.get("strict_goal_check", True))

        self.actions: list[list[str]] = _build_actions(cfg.get("action_set"))
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.frame_stack_n, *self.screen_size),
            dtype=np.uint8,
        )

        self.ram = RamAddresses.from_config(cfg.get("ram"))
        self.reward_cfg = RewardConfig.from_config(cfg.get("reward"))
        self.room_ids = _build_room_ids(cfg.get("room_ids"))
        self._stack = FrameStack(self.frame_stack_n, (1, *self.screen_size))

        self._pyboy = None
        self.stage_id: str | None = None
        self.goal: Goal | None = None
        self._target_room: int | None = None
        # La meta 'boss' solo puede cumplirse después de ver al jefe con vida.
        # Sin esto, un episodio full-level empezaría con boss_hp=0 y "ganaría".
        self._boss_engaged = False
        self.episode_steps = 0
        self.episode_frames = 0
        self.total_frames = 0
        self.last_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Etapas y metas
    # ------------------------------------------------------------------ #

    def _resolve_goal(self, goal: Any, stage_id: str) -> Goal:
        if goal is None:
            return get_stage(stage_id).goal
        if isinstance(goal, Goal):
            return goal
        if isinstance(goal, str):
            return BOSS_GOAL if goal == "boss" else Goal("reach", goal)
        raise TypeError(f"Meta no válida: {goal!r}")

    def _room_of(self, stage_id: str) -> int | None:
        if stage_id in self.room_ids:
            return self.room_ids[stage_id]
        return get_stage(stage_id).room_id

    def _configure_goal(self) -> None:
        goal = self.goal
        assert goal is not None
        if goal.kind == "reach":
            if goal.target == self.stage_id:
                raise ValueError("La meta 'reach' apunta a la misma etapa de inicio")
            target = self._room_of(goal.target or "")
            if target is None:
                raise ValueError(
                    f"Falta `room_id` para la etapa destino {goal.target!r}. "
                    "Configúralo en `env.room_ids` (ver Fase 1)."
                )
            self._target_room = int(target)
        else:
            self._target_room = None

    def load_stage(self, stage_id: str) -> None:
        self._ensure_emulator()
        stage = get_stage(stage_id)
        path = stage.path(self.saves_dir)
        if not path.is_file():
            raise FileNotFoundError(f"Falta el save state {path}")
        # PyBoy >= 2.0 exige un objeto con interfaz de archivo, no una ruta.
        with path.open("rb") as fh:
            self._pyboy.load_state(fh)
        self.stage_id = stage_id

    # ------------------------------------------------------------------ #
    # Reset / step
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        options = options or {}
        stage_id = str(options.get("stage", "spawn"))
        self.load_stage(stage_id)

        self.goal = self._resolve_goal(options.get("goal"), stage_id)
        self._configure_goal()

        self.episode_steps = 0
        self.episode_frames = 0
        self._boss_engaged = False
        info = self._read_info()

        if self.strict_goal_check:
            self._validate_start(info)

        self.last_info = info
        obs = self._stack.reset(self._frame())
        return obs, dict(info)

    def _validate_start(self, info: dict[str, Any]) -> None:
        """Falla rápido si las direcciones de RAM no corresponden al save state."""
        goal = self.goal
        assert goal is not None

        if goal.kind == "reach":
            room = info.get("room")
            if self._target_room is not None and room == self._target_room:
                raise ValueError(
                    f"El save state de {self.stage_id!r} ya está en la sala destino "
                    f"({room}). Revisa `room_ids` o el save state."
                )
            if room is None:
                raise ValueError(
                    "La meta 'reach' requiere leer el ID de sala, pero la dirección "
                    "de RAM `room` no está definida o no es legible."
                )
        else:
            boss_hp = info.get("boss_hp")
            if boss_hp is None:
                raise ValueError(
                    "La meta 'boss' requiere leer `boss_hp`, pero la dirección de RAM "
                    "no está definida o no es legible."
                )
            # Solo se exige jefe vivo si el episodio empieza en la arena. En un
            # episodio full-level (spawn -> jefe) es correcto que valga 0 hasta
            # que el agente llegue a la arena.
            starts_at_boss = get_stage(self.stage_id or "").goal.kind == "boss"
            if starts_at_boss and boss_hp == 0:
                raise ValueError(
                    "`boss_hp` vale 0 al inicio del episodio en la arena: la "
                    "dirección es probablemente incorrecta y el agente 'ganaría' "
                    "en el paso 1."
                )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._pyboy is None:
            raise RuntimeError("El entorno no fue inicializado: llama reset() primero.")

        action = int(action)
        if not 0 <= action < len(self.actions):
            raise ValueError(f"Acción fuera de rango: {action}")

        info_prev = self.last_info
        won = died = False

        for _ in range(self.frame_skip):
            self._tick_action(self.actions[action])
            self.episode_frames += 1
            self.total_frames += 1

            info = self._read_info()
            if info["won"] or info["died"]:
                won, died = bool(info["won"]), bool(info["died"])
                break

        info = self._read_info()
        hp_prev = info_prev.get("hp")
        hp_now = info.get("hp")
        damaged = hp_prev is not None and hp_now is not None and hp_now < hp_prev

        self.episode_steps += 1
        truncated = (
            self.episode_steps >= self.max_steps
            or self.episode_frames >= self.max_frames
        ) and not (won or died)

        x_progress = 0.0
        if info.get("pos_x") is not None and info_prev.get("pos_x") is not None:
            x_progress = info["pos_x"] - info_prev["pos_x"]

        reward = compute_reward(
            self.reward_cfg,
            won=won,
            died=died,
            damaged=damaged and not died,
            x_progress=x_progress,
            truncated=truncated,
        )

        info["damaged"] = damaged
        self.last_info = info

        obs = self._stack.push(self._frame())
        terminated = bool(won or died)
        return obs, reward, terminated, truncated, dict(info)

    # ------------------------------------------------------------------ #
    # Emulador
    # ------------------------------------------------------------------ #

    def _tick_action(self, buttons: list[str]) -> None:
        """Mantiene los botones de la acción durante un frame de emulador."""
        pressed = []
        try:
            for button in buttons:
                if button != "noop":
                    self._pyboy.button_press(button)
                    pressed.append(button)
            self._pyboy.tick()
        finally:
            for button in pressed:
                self._pyboy.button_release(button)

    def _frame(self) -> np.ndarray:
        screen = self._pyboy.screen.ndarray
        return preprocess_frame(screen, self.screen_size, self.grayscale)

    def _read_info(self) -> dict[str, Any]:
        raw = read_ram(self._pyboy, self.ram)

        hp = raw.get("hp")
        game_over = raw.get("game_over")
        died = bool(hp is not None and hp <= 0) or bool(game_over)

        won = self._goal_reached(raw)

        return {
            **raw,
            "died": died,
            "won": won,
            "frames": self.episode_frames,
            "stage": self.stage_id,
            "goal": self.goal.describe(stage_names()) if self.goal else "",
        }

    def _goal_reached(self, raw: dict[str, Any]) -> bool:
        goal = self.goal
        if goal is None:
            return False

        if goal.kind == "reach":
            room = raw.get("room")
            return bool(self._target_room is not None and room == self._target_room)

        # Meta 'boss'
        if raw.get("stage_clear"):
            return True
        boss_hp = raw.get("boss_hp")
        if boss_hp is not None and boss_hp > 0:
            # El jefe ya apareció: a partir de aquí, llegar a 0 es victoria.
            self._boss_engaged = True
        return bool(self._boss_engaged and boss_hp == 0)

    def _ensure_emulator(self) -> None:
        if self._pyboy is not None:
            return
        if not self.rom_path.is_file():
            raise FileNotFoundError(
                f"No se encontró la ROM en {self.rom_path}. "
                "Coloca el archivo en roms/kirby.gb (no se versiona)."
            )

        from pyboy import PyBoy

        try:
            self._pyboy = PyBoy(str(self.rom_path), window="null", sound=False)
        except TypeError:  # versiones sin el argumento `sound`
            self._pyboy = PyBoy(str(self.rom_path), window="null")

        self._pyboy.set_emulation_speed(0)

    # ------------------------------------------------------------------ #

    def close(self) -> None:
        if self._pyboy is not None:
            try:
                self._pyboy.stop(save=False)
            finally:
                self._pyboy = None

    @property
    def pyboy(self) -> Any:
        """Acceso al emulador (útil para depurar direcciones de RAM)."""
        return self._pyboy


def _build_actions(action_set: list[str] | None) -> list[list[str]]:
    """Convierte los nombres del YAML en listas de botones presionados."""
    if not action_set:
        action_set = ["noop", "left", "right", "up", "down", "a", "b"]

    mapping: list[list[str]] = []
    for name in action_set:
        name = str(name).lower().replace(" ", "")
        if name in ("noop", "none", "idle"):
            mapping.append([])
        elif "+" in name:
            mapping.append([part for part in name.split("+") if part])
        else:
            mapping.append([name])
    return mapping


def _build_room_ids(raw: dict[str, Any] | None) -> dict[str, int]:
    ids: dict[str, int] = {}
    for stage_id, value in (raw or {}).items():
        if value is None:
            continue
        ids[stage_id] = int(value, 0) if isinstance(value, str) else int(value)
    return ids
