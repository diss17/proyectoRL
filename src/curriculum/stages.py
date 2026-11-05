"""Registro de etapas, metas y secuencias de entrenamiento.

Modelo canónico
---------------
Cada etapa se identifica por un **ID estable** (`spawn`, `mid`, `door`, `arena`),
no por un índice. La posición en la secuencia es lo que define el orden del
currículo, de modo que el mismo conjunto de save states sirve para el currículo
directo (forward) y el inverso (reverse).

    spawn --(meta)--> mid --(meta)--> door --(meta)--> arena --(meta)--> jefe

`forward`  = [spawn, mid, door, arena]
`reverse`  = [arena, door, mid, spawn]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Goal:
    """Meta de un episodio.

    - ``reach``: llegar a la sala de la etapa `target`.
    - ``boss``:  derrotar al jefe final.
    """

    kind: str
    target: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("reach", "boss"):
            raise ValueError(f"Tipo de meta desconocido: {self.kind}")
        if self.kind == "reach" and not self.target:
            raise ValueError("La meta 'reach' requiere una etapa destino")

    def describe(self, stage_names: dict[str, str] | None = None) -> str:
        if self.kind == "boss":
            return "derrotar al jefe"
        name = (stage_names or {}).get(self.target or "", self.target)
        return f"llegar a {name}"


BOSS_GOAL = Goal("boss")


@dataclass(frozen=True)
class Stage:
    id: str
    name: str
    save_file: str
    description: str
    room_id: int | None
    goal: Goal

    def path(self, saves_dir: str | Path) -> Path:
        return Path(saves_dir) / self.save_file


# `room_id` es el valor de `wCurStageScreen` (0xD03E) verificado con los save
# states. Puede sobreescribirse desde `env.room_ids` en el YAML.
#
# Green Greens: 0 = tramo 1, 1 = Poppy Bros. Sr., 2 = tramo 2,
#               3 = interior del árbol, 4 = arena de Whispy.
STAGES: dict[str, Stage] = {
    "spawn": Stage(
        id="spawn",
        name="Spawn inicial",
        save_file="phase3_spawn.state",
        description="Tramo exterior 1 desde el inicio del nivel.",
        room_id=0,
        goal=Goal("reach", "mid"),
    ),
    "mid": Stage(
        id="mid",
        name="Sección media",
        save_file="phase2_mid.state",
        description="Tramo exterior 2, tras derrotar a Poppy Bros. Sr.",
        room_id=2,
        goal=Goal("reach", "door"),
    ),
    "door": Stage(
        id="door",
        name="Interior del árbol",
        save_file="phase1_door.state",
        description="Sala vertical previa a la arena de Whispy Woods.",
        room_id=3,
        goal=Goal("reach", "arena"),
    ),
    "arena": Stage(
        id="arena",
        name="Arena de Whispy",
        save_file="phase0_whispy.state",
        description="Combate, absorción y proyectiles contra Whispy Woods.",
        room_id=4,
        goal=BOSS_GOAL,
    ),
}

# Orden natural del nivel, de inicio a meta.
LEVEL_ORDER: tuple[str, ...] = ("spawn", "mid", "door", "arena")

SEQUENCES: dict[str, tuple[str, ...]] = {
    "forward": LEVEL_ORDER,
    "reverse": tuple(reversed(LEVEL_ORDER)),
}


def get_stage(stage_id: str) -> Stage:
    try:
        return STAGES[stage_id]
    except KeyError as exc:
        raise KeyError(
            f"Etapa desconocida: {stage_id!r}. Válidas: {sorted(STAGES)}"
        ) from exc


def build_sequence(
    order: str | None = None,
    sequence: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Devuelve la lista de IDs en orden de entrenamiento."""
    if sequence is not None:
        seq = list(sequence)
        unknown = [s for s in seq if s not in STAGES]
        if unknown:
            raise ValueError(f"Etapas desconocidas en la secuencia: {unknown}")
        if len(set(seq)) != len(seq):
            raise ValueError("La secuencia no puede repetir etapas")
        return seq

    key = (order or "reverse").lower()
    if key not in SEQUENCES:
        raise ValueError(
            f"Orden desconocido: {order!r}. Válidos: {sorted(SEQUENCES)}"
        )
    return list(SEQUENCES[key])


def validate_saves(sequence: list[str], saves_dir: str | Path) -> list[str]:
    """Archivos de save state faltantes para una secuencia."""
    return [str(get_stage(s).path(saves_dir)) for s in sequence if not get_stage(s).path(saves_dir).is_file()]


def describe_sequence(sequence: list[str]) -> str:
    lines = ["Secuencia de entrenamiento:"]
    for position, stage_id in enumerate(sequence, start=1):
        stage = get_stage(stage_id)
        lines.append(
            f"  {position}. {stage.name:<20} <- {stage.save_file:<22} "
            f"meta: {stage.goal.describe()}"
        )
    return "\n".join(lines)


def stage_names() -> dict[str, str]:
    return {sid: stage.name for sid, stage in STAGES.items()}
