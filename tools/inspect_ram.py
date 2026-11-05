"""Inspecciona la RAM de los save states en las direcciones del desensamblado.

Direcciones confirmadas por el proyecto `huderlem/kirbydreamland`:
    0xD03B wCurStage             nivel (Green Greens = 0)
    0xD03E wCurStageScreen       pantalla/sala dentro del nivel
    0xD042 wStageLengthInMetatiles
    0xD051 wStageScrollTileX     scroll real (se reinicia por sala)
    0xD052 wStageScrollTileY
    0xD05C wPlayerScreenXCoord
    0xD05D wPlayerScreenYCoord
    0xD086 salud de Kirby
    0xD089 vidas restantes
    0xD093 salud del jefe
"""

from __future__ import annotations

import pathlib

from pyboy import PyBoy

SAVES = [
    ("spawn", "saves/phase3_spawn.state"),
    ("mid", "saves/phase2_mid.state"),
    ("door", "saves/phase1_door.state"),
    ("arena", "saves/phase0_whispy.state"),
]

FIELDS = [
    ("wCurStage", 0xD03B),
    ("wCurStageScreen", 0xD03E),
    ("wStageLength", 0xD042),
    ("scrollTileX", 0xD051),
    ("scrollTileY", 0xD052),
    ("playerScreenX", 0xD05C),
    ("playerScreenY", 0xD05D),
    ("kirbyHP", 0xD086),
    ("lives", 0xD089),
    ("bossHP", 0xD093),
]


def main() -> int:
    pyboy = PyBoy("roms/kirby.gb", window="null", sound=False)
    pyboy.set_emulation_speed(0)

    header = f"{'etapa':<9}" + "".join(f"{name:>17}" for name, _ in FIELDS)
    print(header)
    print("-" * len(header))

    for stage, path in SAVES:
        with pathlib.Path(path).open("rb") as fh:
            pyboy.load_state(fh)
        pyboy.tick()
        values = [pyboy.memory[addr] for _, addr in FIELDS]
        print(f"{stage:<9}" + "".join(f"{value:>17}" for value in values))

    pyboy.stop(save=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
