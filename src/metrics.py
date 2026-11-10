"""Métricas: sample efficiency (P1), convergencia (P2) y retención."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


# --------------------------------------------------------------------------- #
# Métricas básicas de episodios
# --------------------------------------------------------------------------- #

def longest_win_streak(wons: Iterable[bool]) -> int:
    best = current = 0
    for won in wons:
        if won:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def success_rate(wons: Sequence[bool]) -> float:
    if not wons:
        return float("nan")
    return sum(bool(w) for w in wons) / len(wons)


def frames_to_consecutive_wins(
    frames: Sequence[int],
    wons: Sequence[bool],
    target: int = 10,
) -> int | None:
    if len(frames) != len(wons):
        raise ValueError("frames y wons deben tener el mismo largo")

    streak = 0
    for frame, won in zip(frames, wons):
        streak = streak + 1 if won else 0
        if streak >= target:
            return int(frame)
    return None


def retention_score(before: float, after: float) -> float:
    """Persistencia del desempeño previo. 1.0 = intacto, 0.0 = perdido."""
    if before <= 0:
        return float("nan")
    return max(0.0, min(1.0, after / before))


# --------------------------------------------------------------------------- #
# Evaluaciones full-level
# --------------------------------------------------------------------------- #

NUMERIC_EVAL_FIELDS = (
    "frame",
    "frame_actual",
    "episodes",
    "wins",
    "win_rate",
    "return_mean",
    "return_std",
    "length_mean",
)


def read_evals(csv_path: str | Path) -> list[dict[str, Any]]:
    """Lee `evals.csv` y devuelve filas ordenadas por frame, con números."""
    rows: list[dict[str, Any]] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, Any] = dict(raw)
            for field in NUMERIC_EVAL_FIELDS:
                if field in row and row[field] not in ("", None):
                    row[field] = float(row[field])
            if "episodes" in row:
                row["episodes"] = int(row["episodes"])
            if "frame" in row:
                row["frame"] = int(row["frame"])
            if isinstance(row.get("frame_actual"), float):
                row["frame_actual"] = int(row["frame_actual"])
            rows.append(row)

    rows.sort(key=lambda r: r["frame"])
    return rows


def curve_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluaciones periódicas de la curva (sin la evaluación final de 100 ep.).

    La final comparte frame con la última periódica; incluirla duplicaría ese
    punto en P1, en el suavizado de P2 y en la curva de grupo.
    """
    return sorted(
        (r for r in rows if r.get("phase") != "final"), key=lambda r: r["frame"]
    )


def final_success(rows: Sequence[dict[str, Any]]) -> float:
    """S_final: win rate de la evaluación con más episodios (la final de 100)."""
    if not rows:
        return float("nan")
    best = max(rows, key=lambda r: (int(r.get("episodes", 0)), r["frame"]))
    return float(best["win_rate"])


def frames_to_threshold(
    rows: Sequence[dict[str, Any]],
    threshold: float = 0.80,
    confirm: bool = True,
) -> int | None:
    """P1: primer frame con win_rate >= threshold.

    Con `confirm=True`, la evaluación siguiente también debe superar el umbral
    (si existe). Devuelve `None` si nunca ocurre.
    """
    rows = curve_rows(rows)
    for i, row in enumerate(rows):
        if float(row["win_rate"]) < threshold:
            continue
        if confirm and i + 1 < len(rows) and float(rows[i + 1]["win_rate"]) < threshold:
            continue
        return int(row["frame"])
    return None


def convergence_frame(
    rows: Sequence[dict[str, Any]],
    *,
    final: float | None = None,
    band: float = 0.90,
    gate: float = 0.80,
    window: int = 3,
) -> int | None:
    """P2: primer frame desde el que la curva suavizada se mantiene estable.

    Definición:
      1. Solo se calcula si ``S_final >= gate`` (si no, la corrida es censurada).
      2. Se suaviza con media móvil de `window` evaluaciones.
      3. Converge en la evaluación i si desde i en adelante la curva suavizada
         nunca baja de ``band * S_final``.
    """
    s_final = final_success(rows) if final is None else float(final)
    rows = curve_rows(rows)
    if len(rows) < window:
        return None

    if s_final < gate:
        return None

    rates = [float(r["win_rate"]) for r in rows]
    smoothed: list[float] = []
    for i in range(len(rates)):
        chunk = rates[max(0, i - window + 1) : i + 1]
        smoothed.append(sum(chunk) / len(chunk))

    target = band * s_final
    for i in range(window - 1, len(smoothed)):
        if all(value >= target for value in smoothed[i:]):
            return int(rows[i]["frame"])
    return None


def censored_summary(
    values: Sequence[int | None],
    budget: int,
) -> dict[str, Any]:
    """Resumen robusto a censura: mediana solo sobre las corridas que convergen."""
    converged = [int(v) for v in values if v is not None]
    return {
        "n": len(values),
        "converged": len(converged),
        "censored": len(values) - len(converged),
        "median": statistics.median(converged) if converged else None,
        "min": min(converged) if converged else None,
        "max": max(converged) if converged else None,
        "budget": int(budget),
        "values": [int(v) if v is not None else None for v in values],
    }


def group_curve(
    runs: Sequence[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Curva promedio por frame a través de semillas (para graficar)."""
    by_frame: dict[int, list[float]] = {}
    for run in runs:
        for row in curve_rows(run):
            by_frame.setdefault(int(row["frame"]), []).append(float(row["win_rate"]))

    curve = []
    for frame in sorted(by_frame):
        values = by_frame[frame]
        curve.append(
            {
                "frame": frame,
                "win_rate": sum(values) / len(values),
                "n": len(values),
                "win_rate_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        )
    return curve


# --------------------------------------------------------------------------- #
# Resumen de episodios de entrenamiento
# --------------------------------------------------------------------------- #

def read_episodes(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def summarize_episodes(csv_path: str | Path) -> dict[str, Any]:
    rows = read_episodes(csv_path)
    summary: dict[str, Any] = {"total_episodes": len(rows), "per_stage": {}}

    stages = sorted({r["stage"] for r in rows if r.get("stage")})
    for stage in stages:
        subset = [r for r in rows if r["stage"] == stage]
        wons = [r["won"] in ("1", "True", "true") for r in subset]
        returns = [
            float(r["ep_return"]) for r in subset if r.get("ep_return") not in ("", None)
        ]
        summary["per_stage"][stage] = {
            "episodes": len(subset),
            "success_rate": success_rate(wons),
            "longest_streak": longest_win_streak(wons),
            "mean_return": sum(returns) / len(returns) if returns else float("nan"),
        }
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = [f"Episodios totales: {summary['total_episodes']}"]
    for stage, data in summary["per_stage"].items():
        lines.append(
            f"  {stage:<8} {data['episodes']:>6} episodios, "
            f"éxito {data['success_rate']:6.2%}, "
            f"racha máx {data['longest_streak']}, "
            f"retorno medio {data['mean_return']:.2f}"
        )
    return "\n".join(lines)
