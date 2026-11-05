"""Comparación de los grupos experimentales a partir de `results/`.

Uso:
    python -m src.compare --results results
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from .metrics import (
    censored_summary,
    convergence_frame,
    final_success,
    frames_to_threshold,
    group_curve,
    read_evals,
)

GROUPS = ("control", "forward", "reverse")
RUN_PATTERN = re.compile(r"^(?P<group>[a-z_]+)_s(?P<seed>\d+)$")


def discover_runs(results_dir: Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    runs: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for path in sorted(results_dir.iterdir()):
        if not path.is_dir():
            continue
        match = RUN_PATTERN.match(path.name)
        if not match:
            continue
        evals = path / "evals.csv"
        if not evals.is_file():
            continue
        group = match.group("group")
        seed = int(match.group("seed"))
        runs.setdefault(group, {})[seed] = read_evals(evals)
    return runs


def summarize_group(seed_rows: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    seeds = sorted(seed_rows)
    p1 = [frames_to_threshold(seed_rows[s]) for s in seeds]
    p2 = [convergence_frame(seed_rows[s]) for s in seeds]
    finals = [final_success(seed_rows[s]) for s in seeds]
    budget = max(
        (max((r["frame"] for r in seed_rows[s]), default=0) for s in seeds),
        default=0,
    )

    return {
        "seeds": seeds,
        "p1": censored_summary(p1, budget),
        "p2": censored_summary(p2, budget),
        "final_mean": sum(finals) / len(finals) if finals else float("nan"),
        "final_values": finals,
        "curve": group_curve([seed_rows[s] for s in seeds]),
    }


def _frames(value: int | None) -> str:
    return "—" if value is None else f"{value / 1000:.0f}k"


def render_markdown(summary: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Comparación de grupos",
        "",
        "| Grupo | Semillas | P1: frames a 80 % | P2: frames a converger | Censuradas P2 | Éxito final |",
        "|---|---|---|---|---|---|",
    ]
    for group in GROUPS + tuple(g for g in summary if g not in GROUPS):
        data = summary.get(group)
        if data is None:
            continue
        seeds = ", ".join(str(s) for s in data["seeds"])
        p1, p2 = data["p1"], data["p2"]
        lines.append(
            f"| {group} | {seeds} | "
            f"{_frames(p1['median'])} ({p1['converged']}/{p1['n']}) | "
            f"{_frames(p2['median'])} ({p2['converged']}/{p2['n']}) | "
            f"{p2['censored']} | "
            f"{data['final_mean']:.1%} |"
        )

    lines += ["", "## Detalle por semilla", ""]
    for group, data in summary.items():
        lines.append(f"### {group}")
        for seed, p1, p2, final in zip(
            data["seeds"], data["p1"]["values"], data["p2"]["values"], data["final_values"]
        ):
            lines.append(
                f"- semilla {seed}: P1={_frames(p1)}, P2={_frames(p2)}, "
                f"éxito final={final:.1%}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara control / forward / reverse")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None, help="Archivo markdown de salida")
    args = parser.parse_args()

    runs = discover_runs(args.results)
    if not runs:
        raise SystemExit(f"No se encontraron corridas con evals.csv en {args.results}")

    summary = {group: summarize_group(seed_rows) for group, seed_rows in runs.items()}
    report = render_markdown(summary)
    print(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
