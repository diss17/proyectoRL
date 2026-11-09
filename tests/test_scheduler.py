"""Tests del CurriculumScheduler (no requieren ROM ni emulador)."""

from __future__ import annotations

import random

import pytest

from src.curriculum.scheduler import CurriculumScheduler, build_scheduler
from src.curriculum.stages import build_sequence

REVERSE = ["arena", "door", "mid", "spawn"]
FORWARD = ["spawn", "mid", "door", "arena"]


def test_starts_at_first_position_of_reverse() -> None:
    sched = CurriculumScheduler(REVERSE, window=50, threshold=0.85)
    assert sched.current_stage == "arena"


def test_starts_at_first_position_of_forward() -> None:
    sched = CurriculumScheduler(FORWARD, window=50, threshold=0.85)
    assert sched.current_stage == "spawn"


def test_does_not_advance_before_window_is_full() -> None:
    sched = CurriculumScheduler(REVERSE, window=50, threshold=0.85)
    for _ in range(49):
        sched.record_episode("arena", True)
    assert not sched.maybe_advance()
    assert sched.current_stage == "arena"


def test_advances_when_window_full_and_threshold_met() -> None:
    sched = CurriculumScheduler(REVERSE, window=50, threshold=0.85)
    for _ in range(43):
        sched.record_episode("arena", True)
    for _ in range(7):
        sched.record_episode("arena", False)

    assert sched.win_rate("arena") == pytest.approx(0.86)
    assert sched.maybe_advance() == "door"
    assert sched.current_stage == "door"
    assert sched.transitions[0]["from"] == "arena"


def test_does_not_advance_below_threshold() -> None:
    sched = CurriculumScheduler(REVERSE, window=50, threshold=0.85)
    for _ in range(42):
        sched.record_episode("arena", True)
    for _ in range(8):
        sched.record_episode("arena", False)

    assert sched.win_rate("arena") == pytest.approx(0.84)
    assert sched.maybe_advance() is None
    assert sched.current_stage == "arena"


def test_epsilon_gate_ignores_high_exploration_episodes() -> None:
    sched = CurriculumScheduler(REVERSE, window=50, threshold=0.85, epsilon_gate=0.10)

    for _ in range(100):
        counted = sched.record_episode("arena", True, epsilon=0.9)
        assert counted is False

    assert sched.stage_episodes["arena"] == 100
    assert sched.window_size("arena") == 0
    assert sched.stage_ignored["arena"] == 100
    assert not sched.maybe_advance()

    for _ in range(50):
        assert sched.record_episode("arena", True, epsilon=0.05) is True
    assert sched.maybe_advance() == "door"


def test_replay_uses_previous_positions_only() -> None:
    sched = CurriculumScheduler(REVERSE, threshold=0.85, replay_prob=1.0)
    sched.position = 2  # arena, door ya superadas; actual = mid

    rng = random.Random(0)
    picks = {sched.select_stage(rng) for _ in range(200)}
    assert picks == {"arena", "door"}


def test_replay_forward_also_uses_previous_positions() -> None:
    sched = CurriculumScheduler(FORWARD, threshold=0.85, replay_prob=1.0)
    sched.position = 2  # spawn, mid superadas; actual = door

    rng = random.Random(0)
    picks = {sched.select_stage(rng) for _ in range(200)}
    assert picks == {"spawn", "mid"}


def test_no_replay_when_probability_is_zero() -> None:
    sched = CurriculumScheduler(REVERSE, threshold=0.85, replay_prob=0.0)
    sched.position = 3
    rng = random.Random(1)
    assert all(sched.select_stage(rng) == "spawn" for _ in range(100))


def test_final_position_never_advances() -> None:
    sched = CurriculumScheduler(REVERSE, window=10, threshold=0.5)
    sched.position = 3
    for _ in range(20):
        sched.record_episode("spawn", True)
    assert sched.maybe_advance() is None
    assert sched.current_stage == "spawn"


def test_fixed_frames_mode_ignores_win_rate() -> None:
    sched = CurriculumScheduler(REVERSE, threshold=None, fixed_frames_per_stage=1000)
    sched.add_frames("arena", 999)
    assert sched.maybe_advance() is None

    sched.add_frames("arena", 1)
    assert sched.maybe_advance() == "door"


def test_state_dict_roundtrip_with_sequence() -> None:
    sched = CurriculumScheduler(FORWARD, window=20, threshold=0.8, replay_prob=0.3)
    for won in [True, False, True, True]:
        sched.record_episode("spawn", won, epsilon=0.05)
    sched.position = 1
    sched.add_frames("spawn", 123)

    clone = CurriculumScheduler(["spawn"])
    clone.load_state_dict(sched.state_dict())

    assert clone.stage_ids == FORWARD
    assert clone.current_stage == "mid"
    assert clone.stage_episodes["spawn"] == 4
    assert clone.stage_frames["spawn"] == 123
    assert clone.win_rate("spawn") == pytest.approx(sched.win_rate("spawn"))
    assert clone.state_dict() == sched.state_dict()


def test_build_scheduler_from_config() -> None:
    sched = build_scheduler(
        {
            "window": 50,
            "threshold": 0.85,
            "replay_prob": 0.2,
            "epsilon_gate": 0.1,
            "min_episodes_per_stage": 50,
            "fixed_frames_per_stage": None,
        },
        REVERSE,
    )
    assert sched.adaptive
    assert sched.epsilon_gate == pytest.approx(0.1)
    assert sched.current_stage == "arena"


def test_invalid_replay_probability_raises() -> None:
    with pytest.raises(ValueError):
        CurriculumScheduler(REVERSE, replay_prob=1.5)


def test_sequences_come_from_the_same_stage_registry() -> None:
    assert build_sequence(order="reverse") == REVERSE
    assert build_sequence(order="forward") == FORWARD
    assert set(build_sequence("reverse")) == set(build_sequence("forward"))
