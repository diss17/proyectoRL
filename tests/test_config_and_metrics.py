"""Tests de configuración, secuencias y métricas P1/P2."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.metrics import (
    censored_summary,
    convergence_frame,
    final_success,
    frames_to_consecutive_wins,
    frames_to_threshold,
    group_curve,
    longest_win_streak,
    retention_score,
    success_rate,
)
from src.utils import EpsilonSchedule, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _rows(rates, frames=None, last_episodes=100):
    frames = frames or [(i + 1) * 250_000 for i in range(len(rates))]
    rows = [
        {"frame": f, "win_rate": r, "episodes": 10}
        for f, r in zip(frames, rates)
    ]
    if rows:
        rows[-1]["episodes"] = last_episodes
    return rows


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

def test_base_config_has_expected_sections_and_values() -> None:
    cfg = load_config(CONFIG_DIR / "base.yaml")
    for section in ("env", "agent", "curriculum", "train"):
        assert section in cfg

    assert cfg["agent"]["gamma"] == pytest.approx(0.997)
    assert cfg["agent"]["buffer_size"] == 20000
    assert len(cfg["env"]["action_set"]) == 11
    assert cfg["train"]["eval_every_frames"] == 250_000
    assert cfg["train"]["eval_episodes"] == 10
    assert cfg["train"]["final_eval_episodes"] == 100


def test_control_disables_curriculum() -> None:
    cfg = load_config(CONFIG_DIR / "control.yaml")
    assert cfg["curriculum"]["enabled"] is False


def test_forward_and_reverse_only_differ_in_order() -> None:
    forward = load_config(CONFIG_DIR / "forward.yaml")
    reverse = load_config(CONFIG_DIR / "reverse.yaml")

    assert forward["curriculum"]["order"] == "forward"
    assert reverse["curriculum"]["order"] == "reverse"

    forward_norm = copy.deepcopy(forward)
    reverse_norm = copy.deepcopy(reverse)
    forward_norm["curriculum"]["order"] = None
    reverse_norm["curriculum"]["order"] = None
    forward_norm["train"]["run_name"] = None
    reverse_norm["train"]["run_name"] = None

    assert forward_norm == reverse_norm


def test_experiment_groups_share_agent_and_reward() -> None:
    base = load_config(CONFIG_DIR / "base.yaml")
    for name in ("control.yaml", "forward.yaml", "reverse.yaml"):
        cfg = load_config(CONFIG_DIR / name)
        assert cfg["agent"] == base["agent"], f"{name} altera hiperparámetros del agente"
        assert cfg["env"]["reward"] == base["env"]["reward"], f"{name} altera la recompensa"


def test_curriculum_budget_matches_total_frames() -> None:
    for name in ("forward.yaml", "reverse.yaml"):
        cfg = load_config(CONFIG_DIR / name)
        cur = cfg["curriculum"]
        assert cur["frames"] + cur["consolidation_frames"] == cfg["train"]["total_frames"]


def test_ablations_load_through_parent_default() -> None:
    cfg = load_config(CONFIG_DIR / "ablations" / "fixed_steps.yaml")
    assert cfg["curriculum"]["fixed_frames_per_stage"] == 375_000
    # 4 etapas x presupuesto fijo debe caber en el currículo (en frames).
    assert 4 * cfg["curriculum"]["fixed_frames_per_stage"] <= cfg["curriculum"]["frames"]
    assert cfg["agent"]["gamma"] == pytest.approx(0.997)


# --------------------------------------------------------------------------- #
# Epsilon
# --------------------------------------------------------------------------- #

def test_epsilon_schedule_is_linear_and_bounded() -> None:
    eps = EpsilonSchedule(start=1.0, end=0.05, decay_frames=1000)
    assert eps.value(0) == pytest.approx(1.0)
    assert eps.value(500) == pytest.approx(0.525)
    assert eps.value(1000) == pytest.approx(0.05)
    assert eps.value(5000) == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# P1 / P2
# --------------------------------------------------------------------------- #

def test_p1_requires_confirmation_when_available() -> None:
    rows = _rows([0.2, 0.5, 0.8, 0.8])
    assert frames_to_threshold(rows, threshold=0.8) == 750_000

    # El primer 0.8 no se confirma en la evaluación siguiente.
    rows = _rows([0.2, 0.8, 0.7, 0.9])
    assert frames_to_threshold(rows, threshold=0.8) == 1_000_000


def test_p1_returns_none_when_never_reached() -> None:
    assert frames_to_threshold(_rows([0.1, 0.2, 0.3]), threshold=0.8) is None


def test_p2_converges_when_curve_stabilises() -> None:
    rows = _rows([0.0, 0.0, 0.5, 0.8, 0.9, 0.9, 0.9])
    assert convergence_frame(rows) == 1_500_000


def test_p2_is_censored_below_final_gate() -> None:
    rows = _rows([0.0, 0.1, 0.3, 0.4, 0.5, 0.5])
    assert final_success(rows) == pytest.approx(0.5)
    assert convergence_frame(rows) is None


def test_p2_returns_none_with_too_few_points() -> None:
    assert convergence_frame(_rows([0.9, 0.9])) is None


def test_censored_summary() -> None:
    summary = censored_summary([100, None, 300], budget=1000)
    assert summary["n"] == 3
    assert summary["converged"] == 2
    assert summary["censored"] == 1
    assert summary["median"] == 200
    assert summary["values"] == [100, None, 300]


def test_group_curve_averages_seeds() -> None:
    runs = [_rows([0.0, 0.5, 1.0]), _rows([0.0, 0.5, 0.0])]
    curve = group_curve(runs)
    assert [c["win_rate"] for c in curve] == [0.0, 0.5, 0.5]
    assert curve[0]["n"] == 2


# --------------------------------------------------------------------------- #
# Métricas auxiliares
# --------------------------------------------------------------------------- #

def test_longest_win_streak() -> None:
    assert longest_win_streak([0, 1, 1, 0, 1, 1, 1]) == 3
    assert longest_win_streak([0, 0, 0]) == 0


def test_frames_to_consecutive_wins() -> None:
    frames = [10, 20, 30, 40, 50]
    wons = [False, True, True, False, True]
    assert frames_to_consecutive_wins(frames, wons, target=2) == 30
    assert frames_to_consecutive_wins(frames, wons, target=3) is None


def test_retention_and_success_rate() -> None:
    assert retention_score(0.8, 0.8) == pytest.approx(1.0)
    assert retention_score(0.8, 0.4) == pytest.approx(0.5)
    assert retention_score(0.8, 1.0) == pytest.approx(1.0)
    assert success_rate([True, False, True, True]) == pytest.approx(0.75)


def test_final_eval_row_is_excluded_from_curve_metrics() -> None:
    rows = [
        {"frame": 250_000, "phase": "curriculum", "win_rate": 0.5, "episodes": 10},
        {"frame": 500_000, "phase": "curriculum", "win_rate": 0.8, "episodes": 10},
        {"frame": 500_000, "phase": "final", "win_rate": 0.9, "episodes": 100},
    ]
    # S_final sigue saliendo de la evaluación final de 100 episodios.
    assert final_success(rows) == pytest.approx(0.9)
    curve = group_curve([rows])
    assert [c["frame"] for c in curve] == [250_000, 500_000]
    assert curve[-1]["win_rate"] == pytest.approx(0.8)
    # P1 sin confirmación disponible usa la última periódica, no la final.
    assert frames_to_threshold(rows, threshold=0.8) == 500_000
