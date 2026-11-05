"""Captura los save states de las cuatro etapas jugando manualmente.

Abre PyBoy con ventana a velocidad real. Mientras juegas, pulsa:

    F1 -> saves/phase3_spawn.state   (spawn, tramo exterior 1)
    F2 -> saves/phase2_mid.state     (tramo exterior 2)
    F3 -> saves/phase1_door.state    (interior del árbol, pre-arena)
    F4 -> saves/phase0_whispy.state  (arena de Whispy)
    F5 -> salir guardando

Teclas del juego (PyBoy 2.9): flechas para mover, A = saltar, S = inhalar,
Enter = Start, Espacio = acelerar. Cuidado: Z y X guardan/cargan el estado
interno del emulador; no son botones del juego.

Cada guardado imprime el estado de RAM leído (HP, vidas, sala y salud del jefe)
para que puedas comprobar que estás en la sala correcta.

Sala (0xD03E): 0 = tramo 1, 1 = Poppy Bros. Sr., 2 = tramo 2,
               3 = interior del árbol, 4 = arena de Whispy.

Uso:
    python tools/capture_saves.py
    python tools/capture_saves.py --selftest   # solo comprueba que abre la ventana
"""

from __future__ import annotations

import argparse
import ctypes
import pathlib
import sys

# Etapas en el orden del nivel (F1..F4).
STAGES: dict[int, tuple[str, str]] = {
    0x70: ("spawn", "phase3_spawn.state"),      # F1
    0x71: ("mid", "phase2_mid.state"),          # F2
    0x72: ("door", "phase1_door.state"),        # F3
    0x73: ("arena", "phase0_whispy.state"),     # F4
}
QUIT_KEY = 0x74  # F5

HP = 0xD086
LIVES = 0xD089
ROOM = 0xD03E
BOSS_HP = 0xD093
POS_X = 0xD051


def key_down(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def describe(pyboy) -> str:
    return (
        f"hp={pyboy.memory[HP]} vidas={pyboy.memory[LIVES]} "
        f"sala={pyboy.memory[ROOM]} jefe={pyboy.memory[BOSS_HP]} "
        f"x={pyboy.memory[POS_X]}"
    )


def save_state(pyboy, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".state.tmp")
    with tmp.open("wb") as fh:
        pyboy.save_state(fh)
    if path.is_file():
        path.replace(path.with_suffix(".state.bak"))
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", type=pathlib.Path, default=pathlib.Path("roms/kirby.gb"))
    parser.add_argument("--saves", type=pathlib.Path, default=pathlib.Path("saves"))
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="abre la ventana, corre 60 frames y sale (no guarda nada)",
    )
    args = parser.parse_args()

    if not args.rom.is_file():
        print(f"ERROR: no existe la ROM en {args.rom}", file=sys.stderr)
        return 1

    from pyboy import PyBoy

    print(f"Cargando {args.rom} ...")
    pyboy = PyBoy(str(args.rom), window="SDL2", scale=args.scale)
    pyboy.set_emulation_speed(1)

    # Deja el juego en el primer frame jugable si aún está en los menús.
    try:
        pyboy.game_wrapper.start_game()
        print("Partida iniciada automáticamente (nivel 1).")
    except Exception as exc:  # noqa: BLE001
        print(f"Aviso: no se pudo iniciar la partida automáticamente ({exc}).")
        print("Empieza tú desde el menú.")

    print("\n=== Teclas para jugar (PyBoy 2.9) ===")
    print("  Flechas    mover")
    print("  A          boton A -> saltar")
    print("  S          boton B -> inhalar / escupir (mantener y soltar)")
    print("  Enter      Start")
    print("  Backspace  Select")
    print("  Espacio    acelerar (mantener)")
    print("  P          pausa")
    print("  Esc        salir")
    print("  C          cambiar paleta")
    print("  F11        pantalla completa")
    print("\n  OJO: Z y X son guardar/cargar estado del emulador. No los uses")
    print("       como botones del juego: Z escribe roms/kirby.gb.state.")
    print("\n=== Teclas de captura ===")
    for vk, (stage, name) in STAGES.items():
        print(f"  F{vk - 0x6F}: guardar {stage:<6} -> {args.saves / name}")
    print("  F5: salir guardando\n")

    if args.selftest:
        for _ in range(60):
            if not pyboy.tick():
                break
        print(f"selftest OK ({describe(pyboy)})")
        pyboy.stop(save=False)
        return 0

    prev: set[int] = set()
    saved: dict[str, int] = {}

    try:
        running = True
        while running:
            running = pyboy.tick()

            now = {vk for vk in (*STAGES, QUIT_KEY) if key_down(vk)}
            for vk in now - prev:
                if vk == QUIT_KEY:
                    running = False
                    break
                stage, name = STAGES[vk]
                path = args.saves / name
                save_state(pyboy, path)
                saved[stage] = saved.get(stage, 0) + 1
                print(f"[F{vk - 0x6F}] {stage:<6} -> {path}  ({describe(pyboy)})")
            prev = now
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        pyboy.stop(save=False)

    print("\nResumen:")
    for stage, _ in STAGES.values():
        times = saved.get(stage, 0)
        print(f"  {stage:<6} {'guardado' if times else 'NO guardado'}"
              f"{f' ({times} veces)' if times else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
