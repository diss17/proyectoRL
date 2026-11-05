"""Tests del registro de etapas y sus metas."""

from __future__ import annotations

import pytest

from src.curriculum.stages import (
    BOSS_GOAL,
    LEVEL_ORDER,
    STAGES,
    build_sequence,
    describe_sequence,
    get_stage,
    validate_saves,
)


def test_level_order_is_spawn_to_boss() -> None:
    assert LEVEL_ORDER == ("spawn", "mid", "door", "arena")


def test_reverse_is_the_mirror_of_forward() -> None:
    forward = build_sequence("forward")
    reverse = build_sequence("reverse")
    assert reverse == list(reversed(forward))
    assert reverse[0] == "arena"
    assert reverse[-1] == "spawn"


def test_goal_chain_advances_toward_the_boss() -> None:
    assert get_stage("spawn").goal.kind == "reach"
    assert get_stage("spawn").goal.target == "mid"
    assert get_stage("mid").goal.target == "door"
    assert get_stage("door").goal.target == "arena"
    assert get_stage("arena").goal == BOSS_GOAL


def test_each_stage_has_a_save_file() -> None:
    for stage in STAGES.values():
        assert stage.save_file.endswith(".state")
    assert get_stage("arena").save_file == "phase0_whispy.state"
    assert get_stage("spawn").save_file == "phase3_spawn.state"


def test_unknown_stage_raises() -> None:
    with pytest.raises(KeyError):
        get_stage("castle")


def test_duplicate_sequence_raises() -> None:
    with pytest.raises(ValueError):
        build_sequence(sequence=["spawn", "spawn"])


def test_unknown_order_raises() -> None:
    with pytest.raises(ValueError):
        build_sequence(order="backwards")


def test_validate_saves_reports_missing_files(tmp_path) -> None:
    missing = validate_saves(["spawn", "arena"], tmp_path)
    assert len(missing) == 2

    (tmp_path / "phase3_spawn.state").write_bytes(b"x")
    missing = validate_saves(["spawn", "arena"], tmp_path)
    assert len(missing) == 1
    assert missing[0].endswith("phase0_whispy.state")


def test_describe_sequence_mentions_goals() -> None:
    text = describe_sequence(build_sequence("reverse"))
    assert "Arena de Whispy" in text
    assert "derrotar al jefe" in text
