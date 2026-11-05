"""Tests del script de comparación entre grupos."""

from __future__ import annotations

import csv

from src.compare import discover_runs, render_markdown, summarize_group
from src.metrics import read_evals


def write_run(root, name: str, rates: list[float], final: float) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    frames = [(i + 1) * 250_000 for i in range(len(rates))]

    with (run_dir / "evals.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "phase", "episodes", "wins", "win_rate", "return_mean", "return_std", "length_mean"])
        for frame, rate in zip(frames, rates):
            writer.writerow([frame, "eval", 10, int(rate * 10), rate, 0.0, 0.0, 100.0])
        writer.writerow([frames[-1], "final", 100, int(final * 100), final, 0.0, 0.0, 100.0])


def test_discover_runs_and_report(tmp_path) -> None:
    write_run(tmp_path, "forward_s0", [0.1, 0.4, 0.8, 0.85], 0.85)
    write_run(tmp_path, "forward_s1", [0.0, 0.3, 0.7, 0.8], 0.8)
    write_run(tmp_path, "reverse_s0", [0.6, 0.8, 0.9, 0.9], 0.9)
    write_run(tmp_path, "control_s0", [0.0, 0.0, 0.1, 0.2], 0.2)

    runs = discover_runs(tmp_path)
    assert set(runs) == {"forward", "reverse", "control"}
    assert set(runs["forward"]) == {0, 1}

    summary = {group: summarize_group(seed_rows) for group, seed_rows in runs.items()}
    report = render_markdown(summary)

    assert "| forward |" in report
    assert "| reverse |" in report
    assert "| control |" in report
    assert "P1" in report and "P2" in report
    # Control nunca alcanza el 80 %: P1 censurada.
    assert summary["control"]["p1"]["converged"] == 0


def test_read_evals_parses_numbers(tmp_path) -> None:
    write_run(tmp_path, "reverse_s0", [0.5, 0.9], 0.9)
    rows = read_evals(tmp_path / "reverse_s0" / "evals.csv")

    assert rows[0]["frame"] == 250_000
    assert isinstance(rows[0]["win_rate"], float)
    assert rows[0]["episodes"] == 10
