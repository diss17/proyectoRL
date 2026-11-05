"""Tests del wrapper KirbyEnv con un emulador falso (sin ROM)."""

from __future__ import annotations

import numpy as np
import pytest

from src.env.kirby_env import KirbyEnv

SAVES = {
    "phase0_whispy.state",
    "phase1_door.state",
    "phase2_mid.state",
    "phase3_spawn.state",
}

# Direcciones reales (ver src/env/ram.py); el emulador es falso pero la RAM
# se direcciona igual que en el juego.
HP = 0xD086
BOSS_HP = 0xD093
POS_X = 0xD051
POS_Y = 0xD052
ROOM = 0xD03E


class FakeScreen:
    def __init__(self) -> None:
        self.ndarray = np.zeros((144, 160, 4), dtype=np.uint8)


class FakePyBoy:
    def __init__(self) -> None:
        self.memory = [0] * 0x10000
        self.screen = FakeScreen()
        self.frames = 0
        self.held: list[str] = []

    def set_emulation_speed(self, value: int) -> None:
        pass

    def load_state(self, path: str) -> None:
        pass

    def tick(self, *args, **kwargs) -> bool:
        self.frames += 1
        return True

    def button_press(self, button: str) -> None:
        self.held.append(button)

    def button_release(self, button: str) -> None:
        self.held.remove(button)

    def stop(self, save: bool = False) -> None:
        pass


def make_env(tmp_path, **overrides) -> KirbyEnv:
    for name in SAVES:
        (tmp_path / name).write_bytes(b"x")

    cfg = {
        "rom_path": "missing.gb",
        "saves_dir": str(tmp_path),
        "frame_skip": 4,
        "frame_stack": 4,
        "screen_size": [84, 84],
        "grayscale": True,
        "max_steps_per_episode": 5,
        "max_frames_per_episode": 100,
        "action_set": ["noop", "left", "right", "a", "right+a"],
        "strict_goal_check": True,
        "room_ids": {"spawn": 0, "mid": 2, "door": 3, "arena": 4},
        "reward": {"win": 10.0, "death": -5.0, "damage": -0.5},
    }
    cfg.update(overrides)

    env = KirbyEnv(cfg)
    env._pyboy = FakePyBoy()
    return env


def test_observation_and_action_spaces(tmp_path) -> None:
    env = make_env(tmp_path)
    assert env.observation_space.shape == (4, 84, 84)
    assert env.action_space.n == 5
    assert env.actions[4] == ["right", "a"]
    env.close()


def test_reach_goal_wins_on_room_change(tmp_path) -> None:
    env = make_env(tmp_path)
    env.pyboy.memory[HP] = 6
    env.pyboy.memory[ROOM] = 0

    obs, info = env.reset(options={"stage": "spawn"})
    assert info["stage"] == "spawn"
    assert info["goal"] == "llegar a Sección media"

    env.pyboy.memory[ROOM] = 2
    obs, reward, terminated, truncated, info = env.step(1)

    assert terminated and not truncated
    assert info["won"] is True
    assert reward == pytest.approx(10.0)
    env.close()


def test_boss_goal_wins_when_boss_hp_reaches_zero(tmp_path) -> None:
    env = make_env(tmp_path)
    env.pyboy.memory[HP] = 6
    env.pyboy.memory[BOSS_HP] = 10
    env.pyboy.memory[ROOM] = 4

    env.reset(options={"stage": "arena"})
    env.pyboy.memory[BOSS_HP] = 0
    obs, reward, terminated, truncated, info = env.step(0)

    assert terminated and info["won"] is True
    assert reward == pytest.approx(10.0)
    env.close()


def test_full_level_boss_goal_requires_seeing_the_boss(tmp_path) -> None:
    """En spawn->jefe, boss_hp=0 al inicio no debe contar como victoria."""
    env = make_env(tmp_path)
    env.pyboy.memory[HP] = 6
    env.pyboy.memory[ROOM] = 0
    env.pyboy.memory[BOSS_HP] = 0

    _, info = env.reset(options={"stage": "spawn", "goal": "boss"})
    assert info["won"] is False
    assert env._boss_engaged is False

    # Sigue en el nivel con boss_hp=0: no gana.
    _, _, terminated, _, info = env.step(0)
    assert not terminated
    assert info["won"] is False

    # Llega a la arena y aparece el jefe con vida: todavía no gana.
    env.pyboy.memory[ROOM] = 4
    env.pyboy.memory[BOSS_HP] = 6
    _, _, terminated, _, info = env.step(0)
    assert not terminated
    assert info["won"] is False
    assert env._boss_engaged is True

    # Derrota al jefe: ahora sí.
    env.pyboy.memory[BOSS_HP] = 0
    _, reward, terminated, _, info = env.step(0)
    assert terminated and info["won"] is True
    assert reward == pytest.approx(10.0)
    env.close()


def test_death_ends_episode(tmp_path) -> None:
    env = make_env(tmp_path)
    env.pyboy.memory[HP] = 6
    env.pyboy.memory[ROOM] = 0

    env.reset(options={"stage": "spawn"})
    env.pyboy.memory[HP] = 0
    obs, reward, terminated, truncated, info = env.step(0)

    assert terminated and info["died"] is True
    assert reward == pytest.approx(-5.0)
    env.close()


def test_damage_is_penalised(tmp_path) -> None:
    env = make_env(tmp_path)
    env.pyboy.memory[HP] = 6
    env.pyboy.memory[ROOM] = 0

    env.reset(options={"stage": "spawn"})
    env.pyboy.memory[HP] = 4
    obs, reward, terminated, truncated, info = env.step(0)

    assert not terminated
    assert reward == pytest.approx(-0.5)
    assert info["damaged"] is True
    assert info["hp"] == 4
    env.close()


def test_truncates_at_max_steps(tmp_path) -> None:
    env = make_env(tmp_path, max_steps_per_episode=2)
    env.pyboy.memory[HP] = 6
    env.pyboy.memory[ROOM] = 0
    env.reset(options={"stage": "spawn"})

    _, _, terminated, truncated, _ = env.step(0)
    assert not terminated and not truncated

    _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated
    env.close()


def test_buttons_are_released_after_each_tick(tmp_path) -> None:
    env = make_env(tmp_path)
    env.pyboy.memory[HP] = 6
    env.pyboy.memory[ROOM] = 0
    env.reset(options={"stage": "spawn"})

    env.step(4)  # right + a
    assert env.pyboy.held == []
    env.close()


def test_boss_goal_rejects_zero_boss_hp_at_start(tmp_path) -> None:
    env = make_env(tmp_path)
    env.pyboy.memory[HP] = 6
    env.pyboy.memory[BOSS_HP] = 0
    env.pyboy.memory[ROOM] = 4

    with pytest.raises(ValueError, match="boss_hp"):
        env.reset(options={"stage": "arena"})
    env.close()


def test_reach_goal_rejects_start_inside_target_room(tmp_path) -> None:
    env = make_env(tmp_path)
    env.pyboy.memory[HP] = 6
    env.pyboy.memory[ROOM] = 2  # ya está en la sala destino de spawn

    with pytest.raises(ValueError, match="sala destino"):
        env.reset(options={"stage": "spawn"})
    env.close()


def test_explicit_goal_string_is_accepted(tmp_path) -> None:
    env = make_env(tmp_path)
    env.pyboy.memory[HP] = 6
    env.pyboy.memory[ROOM] = 0

    _, info = env.reset(options={"stage": "spawn", "goal": "door"})
    assert info["goal"] == "llegar a Interior del árbol"

    env.pyboy.memory[ROOM] = 3
    _, reward, terminated, _, _ = env.step(0)
    assert terminated and reward == pytest.approx(10.0)
    env.close()
