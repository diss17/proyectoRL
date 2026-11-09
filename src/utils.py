"""Utilidades compartidas: configuración, semillas, logging, schedule de epsilon."""

from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusiona recursivamente `override` sobre `base` sin mutar los originales."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Carga un YAML resolviendo la herencia declarada en `defaults`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    parent = cfg.pop("defaults", None)
    if parent:
        base = load_config(path.parent / parent)
        cfg = _deep_merge(base, cfg)

    return cfg


def config_to_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(cfg)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def run_directory(cfg: dict[str, Any], seed: int | None = None) -> Path:
    name = cfg["train"].get("run_name") or "run"
    seed = cfg.get("seed", 0) if seed is None else seed
    out = Path(cfg["train"]["out_dir"]) / f"{name}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Exploración
# --------------------------------------------------------------------------- #

@dataclass
class EpsilonSchedule:
    """Decaimiento lineal de epsilon dentro de una fase o etapa."""

    start: float = 1.0
    end: float = 0.05
    decay_frames: int = 500_000

    def value(self, frames: int) -> float:
        if self.decay_frames <= 0:
            return self.end
        frac = min(1.0, max(0, frames) / self.decay_frames)
        return self.start + frac * (self.end - self.start)


# --------------------------------------------------------------------------- #
# Registro de episodios
# --------------------------------------------------------------------------- #

EPISODE_FIELDS = [
    "frame",
    "episode",
    "phase",
    "stage",
    "ep_return",
    "length",
    "won",
    "epsilon",
    "loss",
    "curriculum_stage",
    "win_rate_stage",
]

EVAL_FIELDS = [
    "frame",          # punto nominal de la grilla (múltiplo de eval_every)
    "frame_actual",   # contador real de frames al evaluar
    "phase",
    "episodes",
    "wins",
    "win_rate",
    "return_mean",
    "return_std",
    "length_mean",
]


class CSVLogger:
    """CSV incremental con cabecera opcional."""

    def __init__(self, path: str | Path, fields: list[str], append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        append = append and self.path.is_file()
        self._fh = self.path.open("a" if append else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fields)
        if not append:
            self._writer.writeheader()

    def log(self, **row: Any) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self._writer.fieldnames})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class EpisodeLogger:
    """CSV de episodios + TensorBoard opcional."""

    def __init__(self, out_dir: str | Path, append: bool = False, use_tensorboard: bool = True) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.csv = CSVLogger(out_dir / "episodes.csv", EPISODE_FIELDS, append=append)

        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(out_dir / "tb"))
            except Exception:
                self._tb = None

    def log(self, **row: Any) -> None:
        self.csv.log(**row)

        if self._tb is not None:
            step = int(row.get("frame", 0))
            if row.get("ep_return") not in ("", None):
                self._tb.add_scalar("episode/return", row["ep_return"], step)
            if row.get("won") not in ("", None):
                self._tb.add_scalar("episode/won", row["won"], step)
            if row.get("loss") not in ("", None):
                self._tb.add_scalar("train/loss", row["loss"], step)

    def close(self) -> None:
        self.csv.close()
        if self._tb is not None:
            self._tb.close()

    def __enter__(self) -> "EpisodeLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class EvalLogger:
    """CSV de evaluaciones full-level (una fila por evaluación)."""

    def __init__(self, out_dir: str | Path, append: bool = False) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.csv = CSVLogger(out_dir / "evals.csv", EVAL_FIELDS, append=append)

    def log(self, **row: Any) -> None:
        self.csv.log(**row)

    def close(self) -> None:
        self.csv.close()

    def __enter__(self) -> "EvalLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
