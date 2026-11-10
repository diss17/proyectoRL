"""Gráficos de una corrida: curvas de entrenamiento (episodes.csv) y evaluación (evals.csv).

Uso:
    python tools/plot_run.py results/arena_only_s0
    python tools/plot_run.py results/arena_only_s0 --window 20 --out curvas.png

Genera una figura de 2x2:
  - retorno por episodio de entrenamiento (media móvil)
  - duración por episodio de entrenamiento (media móvil) + epsilon
  - retorno medio de las evaluaciones greedy
  - win rate y duración media de las evaluaciones greedy
Imprime además un resumen en texto por tramos de frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BLUE = "#2F6BA8"
ORANGE = "#D9822B"
GREY = "#8A8F98"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=20, help="ventana de media móvil")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    eps = pd.read_csv(args.run_dir / "episodes.csv")
    evals_path = args.run_dir / "evals.csv"
    evals = pd.read_csv(evals_path) if evals_path.stat().st_size > 0 else pd.DataFrame()
    curve = evals[evals["phase"] != "final"] if not evals.empty else evals

    w = args.window
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle(f"{args.run_dir.name}: entrenamiento y evaluación greedy")

    ax = axes[0, 0]
    ax.plot(eps["frame"], eps["ep_return"], color=GREY, alpha=0.3, lw=0.8, label="episodio")
    ax.plot(eps["frame"], eps["ep_return"].rolling(w, min_periods=1).mean(),
            color=BLUE, lw=2, label=f"media móvil ({w})")
    ax.set_title("Retorno por episodio (entrenamiento)")
    ax.set_xlabel("frames")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(eps["frame"], eps["length"].rolling(w, min_periods=1).mean(), color=BLUE, lw=2,
            label=f"duración, media móvil ({w})")
    ax.set_title("Duración del episodio (pasos) y ε")
    ax.set_xlabel("frames")
    ax2 = ax.twinx()
    ax2.plot(eps["frame"], eps["epsilon"], color=ORANGE, lw=1.5, ls="--", label="ε")
    ax2.set_ylim(0, 1.05)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper left")

    ax = axes[1, 0]
    if not curve.empty:
        ax.errorbar(curve["frame"], curve["return_mean"], yerr=curve["return_std"],
                    color=BLUE, marker="o", capsize=3)
    ax.set_title("Retorno medio en evaluación greedy (± d.e.)")
    ax.set_xlabel("frames")

    ax = axes[1, 1]
    if not curve.empty:
        ax.plot(curve["frame"], curve["win_rate"] * 100, color=BLUE, marker="o", label="win rate (%)")
        ax.set_ylim(-5, 105)
        ax3 = ax.twinx()
        ax3.plot(curve["frame"], curve["length_mean"], color=ORANGE, marker="s", ls="--",
                 label="duración media (pasos)")
        lines = ax.get_lines() + ax3.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper left")
    ax.set_title("Evaluación greedy: win rate y duración")
    ax.set_xlabel("frames")

    for a in axes.flat:
        a.spines[["top"]].set_visible(False)
        a.grid(alpha=0.2)

    out = args.out or (args.run_dir / "curvas.png")
    fig.savefig(out, dpi=150)
    print(f"Figura: {out}")

    # Resumen por tramos (cuartos del presupuesto)
    eps["tramo"] = pd.cut(eps["frame"], bins=4)
    summary = eps.groupby("tramo", observed=True).agg(
        episodios=("episode", "count"),
        retorno=("ep_return", "mean"),
        duracion=("length", "mean"),
        victorias=("won", "sum"),
        epsilon=("epsilon", "mean"),
    )
    print("\nEntrenamiento por tramos de frames:")
    print(summary.round(2).to_string())
    if not evals.empty:
        print("\nEvaluaciones:")
        print(evals[["frame", "phase", "episodes", "wins", "win_rate", "return_mean",
                     "return_std", "length_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
