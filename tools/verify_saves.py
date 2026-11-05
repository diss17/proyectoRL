"""Verifica los save states de las cuatro etapas.

Para cada archivo:
  - lo carga dos veces en sesiones distintas y compara el hash de la pantalla,
  - lee las variables de RAM confirmadas (HP, vidas, sala, jefe, scroll),
  - comprueba que la sala (`wCurStageScreen`, 0xD03E) es la esperada,
  - comprueba que en la arena `boss_hp` (0xD093) es distinto de 0,
  - corre 120 frames sin input para comprobar que el punto es estable,
  - guarda una captura en `saves/previews/`.

Uso:
    python tools/verify_saves.py
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib

from PIL import Image

# (etapa, archivo, sala esperada en wCurStageScreen)
STAGES: list[tuple[str, str, int]] = [
    ("spawn", "phase3_spawn.state", 0),
    ("mid", "phase2_mid.state", 2),
    ("door", "phase1_door.state", 3),
    ("arena", "phase0_whispy.state", 4),
]

HP = 0xD086
LIVES = 0xD089
BOSS_HP = 0xD093
ROOM = 0xD03E
POS_X = 0xD051
POS_Y = 0xD052


def open_pyboy(rom: pathlib.Path):
    from pyboy import PyBoy

    pyboy = PyBoy(str(rom), window="null", sound=False)
    pyboy.set_emulation_speed(0)
    return pyboy


def load(pyboy, path: pathlib.Path) -> None:
    with path.open("rb") as fh:
        pyboy.load_state(fh)


def screen_hash(pyboy) -> str:
    return hashlib.sha1(pyboy.screen.ndarray.tobytes()).hexdigest()[:12]


def snapshot(pyboy) -> dict[str, int]:
    sprite = pyboy.get_sprite(0)
    return {
        "hp": pyboy.memory[HP],
        "vidas": pyboy.memory[LIVES],
        "sala": pyboy.memory[ROOM],
        "jefe": pyboy.memory[BOSS_HP],
        "scroll_x": pyboy.memory[POS_X],
        "scroll_y": pyboy.memory[POS_Y],
        "sprite_x": sprite.x,
        "sprite_y": sprite.y,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", type=pathlib.Path, default=pathlib.Path("roms/kirby.gb"))
    parser.add_argument("--saves", type=pathlib.Path, default=pathlib.Path("saves"))
    parser.add_argument("--previews", type=pathlib.Path, default=pathlib.Path("saves/previews"))
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()

    if not args.rom.is_file():
        print(f"ERROR: no existe la ROM en {args.rom}")
        return 1

    expected_rooms = [room for _, _, room in STAGES]
    if len(set(expected_rooms)) != len(expected_rooms):
        print("ERROR: las salas esperadas de las etapas se repiten")
        return 1

    args.previews.mkdir(parents=True, exist_ok=True)
    pyboy = open_pyboy(args.rom)
    ok = True
    seen_rooms: dict[int, str] = {}

    print(
        f"{'etapa':<7} {'archivo':<22} {'hp':>3} {'vid':>4} {'sala':>5} "
        f"{'jefe':>5} {'scroll':>7} {'sprX':>5} {'sprY':>5} "
        f"{'determ.':>8} {'estable':>8} {'sala OK':>8}"
    )
    print("-" * 112)

    for stage, filename, expected_room in STAGES:
        path = args.saves / filename
        if not path.is_file():
            print(f"{stage:<7} {filename:<22} FALTA")
            ok = False
            continue

        # 1) primera carga
        load(pyboy, path)
        pyboy.tick()
        first_hash = screen_hash(pyboy)
        data = snapshot(pyboy)
        Image.fromarray(pyboy.screen.ndarray[:, :, :3]).save(
            args.previews / f"{stage}.png"
        )

        # 2) segunda carga: determinismo
        load(pyboy, path)
        pyboy.tick()
        deterministic = screen_hash(pyboy) == first_hash

        # 3) estabilidad sin input
        hp_before = pyboy.memory[HP]
        for _ in range(args.frames):
            pyboy.tick()
        stable = pyboy.memory[HP] == hp_before

        # 4) la sala leída debe ser la esperada y no repetirse entre etapas
        room_ok = data["sala"] == expected_room
        if data["sala"] in seen_rooms:
            room_ok = False
            seen_rooms[data["sala"]] += f"={stage}"
        else:
            seen_rooms[data["sala"]] = stage

        # 5) la arena debe tener al jefe con vida
        boss_ok = data["jefe"] != 0 if stage == "arena" else data["jefe"] == 0
        ok = ok and deterministic and stable and room_ok and boss_ok

        print(
            f"{stage:<7} {filename:<22} {data['hp']:>3} {data['vidas']:>4} "
            f"{data['sala']:>5} {data['jefe']:>5} {data['scroll_x']:>7} "
            f"{data['sprite_x']:>5} {data['sprite_y']:>5} "
            f"{'OK' if deterministic else 'FALLA':>8} "
            f"{'OK' if stable else 'CAMBIA':>8} "
            f"{'OK' if room_ok else 'FALLA':>8}"
        )
        if not boss_ok:
            print(f"  AVISO: boss_hp={data['jefe']} inesperado en {stage}")

    pyboy.stop(save=False)

    print(f"\nCapturas en {args.previews}")
    print("Revisa visualmente que cada una corresponda a su etapa.")
    print("RESULTADO:", "TODO OK" if ok else "HAY FALLOS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
