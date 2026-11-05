"""Smoke test del loop de entrenamiento con entorno y agente falsos."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.train as train_module
from src.curriculum.stages import STAGES


class FakeEnv:
    """Entorno mínimo: episodios de 3 pasos, gana a partir del 2.º intento."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.action_space = SimpleNamespace(n=3)
        self.observation_space = SimpleNamespace(shape=(4, 8, 8))
        self.total_frames = 0
        self.episode_steps = 0
        self.stage = None
        self.goal = None
        self._attempts: dict[str, int] = {}

    def reset(self, *, seed=None, options=None):
        options = options or {}
        self.stage = options.get("stage")
        self.goal = options.get("goal")
        self.episode_steps = 0
        return np.zeros((4, 8, 8), dtype=np.uint8), {
            "stage": self.stage,
            "goal": str(self.goal),
            "won": False,
        }

    def step(self, action):
        self.episode_steps += 1
        self.total_frames += 1
        done = self.episode_steps >= 3
        attempt = self._attempts.get(self.stage, 0)
        won = done and attempt >= 1
        if done:
            self._attempts[self.stage] = attempt + 1
        return (
            np.zeros((4, 8, 8), dtype=np.uint8),
            1.0,
            done,
            False,
            {"stage": self.stage, "won": won},
        )

    def close(self):
        pass


class FakeAgent:
    def __init__(self, observation_shape, n_actions, cfg, device="cpu"):
        self.cfg = cfg
        self.saved: list[Path] = []

    def act(self, obs, epsilon=0.0):
        return 0

    def push(self, *args, **kwargs):
        return None

    def update(self):
        return 0.01

    def save(self, path, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake", encoding="utf-8")
        self.saved.append(path)

    def load(self, path):
        return {"total_frames": 0}

    def close(self):
        pass


@pytest.fixture()
def tiny_config(tmp_path):
    saves = tmp_path / "saves"
    saves.mkdir()
    for stage in STAGES.values():
        (saves / stage.save_file).write_bytes(b"x")

    return {
        "seed": 0,
        "env": {"rom_path": "missing.gb", "saves_dir": str(saves)},
        "agent": {
            "gamma": 0.997,
            "batch_size": 2,
            "buffer_size": 100,
            "min_buffer_to_learn": 1,
            "train_freq": 1,
            "target_update_freq": 10,
            "epsilon_start": 0.05,
            "epsilon_end": 0.05,
            "epsilon_decay_frames": 1,
            "epsilon_stage_start": 0.05,
            "epsilon_stage_decay_frames": 1,
        },
        "curriculum": {
            "enabled": True,
            "order": "reverse",
            "frames": 12,
            "consolidation_frames": 6,
            "window": 2,
            "threshold": 0.6,
            "min_episodes_per_stage": 2,
            "epsilon_gate": 1.0,
            "replay_prob": 0.0,
            "fixed_steps_per_stage": None,
        },
        "train": {
            "total_frames": 18,
            "eval_every_frames": 5,
            "eval_episodes": 2,
            "final_eval_episodes": 3,
            "checkpoint_every_frames": 6,
            "log_every_episodes": 100,
            "out_dir": str(tmp_path / "results"),
            "run_name": "smoke",
            "device": "cpu",
        },
    }


def test_training_loop_runs_end_to_end(tiny_config, monkeypatch):
    monkeypatch.setattr(train_module, "KirbyEnv", FakeEnv)
    monkeypatch.setattr(train_module, "DDQNAgent", FakeAgent)

    train_module.run(tiny_config)

    run_dir = Path(tiny_config["train"]["out_dir"]) / "smoke_s0"
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "final_eval.json").is_file()

    with (run_dir / "episodes.csv").open(encoding="utf-8") as fh:
        episodes = list(csv.DictReader(fh))
    assert episodes
    assert {row["phase"] for row in episodes} == {"curriculum", "consolidation"}
    assert {row["stage"] for row in episodes} >= {"arena", "spawn"}

    with (run_dir / "evals.csv").open(encoding="utf-8") as fh:
        evals = list(csv.DictReader(fh))
    assert evals
    assert evals[-1]["phase"] == "final"
    assert int(evals[-1]["episodes"]) == 3


def test_forward_config_produces_forward_sequence(tiny_config, monkeypatch):
    tiny_config["curriculum"]["order"] = "forward"
    monkeypatch.setattr(train_module, "KirbyEnv", FakeEnv)
    monkeypatch.setattr(train_module, "DDQNAgent", FakeAgent)

    phases = train_module.build_phases(tiny_config)
    assert phases[0].scheduler.stage_ids[0] == "spawn"
    assert phases[-1].goal == "boss"
    assert phases[-1].stage == "spawn"


def test_control_has_single_phase(tiny_config):
    tiny_config["curriculum"]["enabled"] = False
    phases = train_module.build_phases(tiny_config)
    assert len(phases) == 1
    assert phases[0].name == "control"
    assert phases[0].stage == "spawn"
    assert phases[0].goal == "boss"
    assert phases[0].scheduler is None
