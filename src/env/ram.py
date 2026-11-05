"""Lectura de variables del juego desde la RAM del emulador.

Las direcciones de este módulo están **confirmadas** sobre la ROM USA/Europe
(MD5 ``a66e4918edcd042ec171a57fe3ce36c3``) y pueden sobreescribirse desde el
YAML en ``env.ram``.

Origen de las direcciones: desensamblado ``huderlem/kirbydreamland``
(``wram.asm``) y la RAM map de Data Crystal. Correspondencia con los save
states de ``saves/``:

===============  ========  =========================================
Variable         Address   Valores observados (spawn/mid/door/arena)
===============  ========  =========================================
``hp``           0xD086    6 / 2 / 5 / 5
``boss_hp``      0xD093    0 / 0 / 0 / 6
``room``         0xD03E    0 / 2 / 3 / 4
``pos_x``        0xD051    scroll real en tiles (se reinicia por sala)
``pos_y``        0xD052    scroll real en tiles
===============  ========  =========================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RamAddresses:
    """Direcciones (en hexadecimal) de las variables instrumentadas.

    Valores confirmados sobre la ROM USA/Europe con los save states de
    `saves/`. La fuente es el desensamblado `huderlem/kirbydreamland`.
    """

    hp: int | None = 0xD086          # salud de Kirby (6 al empezar)
    boss_hp: int | None = 0xD093     # salud del jefe (Whispy = 6)
    pos_x: int | None = 0xD051       # wStageScrollTileX (se reinicia por sala)
    pos_y: int | None = 0xD052       # wStageScrollTileY
    room: int | None = 0xD03E        # wCurStageScreen (sala dentro del nivel)
    game_over: int | None = None     # se deriva de hp <= 0 / vidas
    stage_clear: int | None = None   # se deriva de boss_hp == 0

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None) -> "RamAddresses":
        """Construye el mapa desde `env.ram` del YAML, ignorando claves nulas."""
        defaults = asdict(cls())
        merged = {**defaults, **(raw or {})}
        return cls(**{k: _to_int(v) for k, v in merged.items() if k in defaults})


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        return int(value, 0)  # admite "0xC000" y "49152"
    return int(value)


def read_ram(pyboy: Any, addresses: RamAddresses) -> dict[str, int | None]:
    """Devuelve las variables leídas; `None` cuando la dirección no está definida."""
    values: dict[str, int | None] = {}
    for name, address in asdict(addresses).items():
        if address is None:
            values[name] = None
            continue
        try:
            values[name] = int(pyboy.memory[address])
        except Exception:
            # Una dirección inválida no debe tumbar el entrenamiento.
            values[name] = None
    return values


def describe(addresses: RamAddresses) -> str:
    lines = ["Mapa de RAM:"]
    for name, address in asdict(addresses).items():
        if address is None:
            lines.append(f"  {name:<14} no definida")
        else:
            lines.append(f"  {name:<14} 0x{address:04X}")
    return "\n".join(lines)
