# Save states

Cada etapa del currículo se reanuda desde una instantánea exacta del emulador.
Los archivos y sus IDs:

| Archivo | ID de etapa | Sala (`0xD03E`) | Punto del juego | Meta local |
|---|---|---|---|---|
| `phase0_whispy.state` | `arena` | 4 | Inicio de la arena de Whispy Woods | Derrotar al jefe |
| `phase1_door.state` | `door` | 3 | Interior del árbol, en la entrada (abajo) | Entrar a la arena |
| `phase2_mid.state` | `mid` | 2 | Tramo exterior 2, tras derrotar a Poppy Bros. Sr. | Llegar al interior del árbol |
| `phase3_spawn.state` | `spawn` | 0 | Spawn inicial, X = 0 | Llegar a la sala 2 y derrotar a Poppy Bros. Sr. |

El mismo conjunto sirve para el currículo **forward** (`spawn → mid → door →
arena`) y el **reverse** (`arena → door → mid → spawn`).

## Estado

| Archivo | Estado | Sala | Notas |
|---|---|---|---|
| `phase3_spawn.state` | ✅ verificado | 0 | Inicio del nivel |
| `phase2_mid.state` | ✅ verificado | 2 | Tramo exterior 2 |
| `phase1_door.state` | ✅ verificado | 3 | Interior del árbol |
| `phase0_whispy.state` | ✅ verificado | 4 | Arena; `boss_hp = 6` |

Los cuatro cargan, son deterministas y estables 120 frames sin input. Ejecuta
`python tools/verify_saves.py` para repetir la comprobación (sala esperada,
`boss_hp`, determinismo y estabilidad).

> Se eliminó `phase1_door.state.bak`, una captura antigua que en realidad
> correspondía a la sala 2 (`mid`). Si vuelves a capturar una etapa, el archivo
> anterior se guarda como `.bak`: comprueba su sala antes de usarlo.

En esta versión de PyBoy (2.9.0) `save_state` y `load_state` exigen un **objeto
con interfaz de archivo**, no una ruta. El wrapper ya lo hace así.

## RAM verificada

| Variable | Dirección | spawn | mid | door | arena |
|---|---|---|---|---|---|
| HP de Kirby | `0xD086` | 6 | 2 | 5 | 5 |
| Vidas | `0xD089` | 5 | 4 | 2 | 2 |
| Sala | `0xD03E` | 0 | 2 | 3 | 4 |
| Salud del jefe | `0xD093` | 0 | 0 | 0 | 6 |
| Scroll real X | `0xD051` | 1 | 10 | 5 | 1 |
| Scroll real Y | `0xD052` | 1 | 1 | 1 | 17 |

Green Greens usa las salas `0` (tramo 1), `2` (tramo 2, que empieza con
Poppy Bros. Sr.), `3` (interior del árbol) y `4` (arena). La sala `1` no se
visita en el camino principal (verificado con la ROM). Poppy Bros. Sr. usa la
misma dirección de vida del jefe (`0xD093` = 3 al aparecer). Las direcciones provienen del
desensamblado [huderlem/kirbydreamland](https://github.com/huderlem/kirbydreamland).

## Cómo generarlos

### Script de captura (recomendado)

```powershell
python tools/capture_saves.py
```

Abre PyBoy con ventana a velocidad real y captura con teclas:

| Tecla | Etapa | Archivo |
|---|---|---|
| `F1` | `spawn` | `phase3_spawn.state` |
| `F2` | `mid` | `phase2_mid.state` |
| `F3` | `door` | `phase1_door.state` |
| `F4` | `arena` | `phase0_whispy.state` |
| `F5` | — | salir |

Teclas para jugar (PyBoy 2.9, layout QWERTY):

| Tecla | Acción |
|---|---|
| Flechas | D-pad |
| `A` | botón A — saltar |
| `S` | botón B — inhalar / escupir |
| `Enter` | Start |
| `Backspace` | Select |
| `Espacio` | acelerar (mantener) |
| `P` | pausa |
| `Esc` | salir |
| `C` | cambiar paleta |
| `F11` | pantalla completa |
| `Z` / `X` | guardar / cargar el estado interno del emulador (**no** son botones del juego) |

Cada guardado imprime `hp`, vidas, **sala** y **salud del jefe** leídos de la
RAM, para comprobar que estás en el punto correcto. Si el archivo ya existe se
conserva como `.bak`.

## Dónde capturar cada etapa (Green Greens)

Green Greens se recorre en este orden (sala de RAM entre paréntesis):

1. **Tramo exterior** (0) — caminas a la derecha entre Waddle Dees, Bronto
   Burts y Cappies. Termina en una **puerta** (caminas dentro y pulsas `Arriba`).
2. **Segundo tramo exterior** (2) — empieza con el miniboss **Poppy Bros. Sr.**;
   tras derrotarlo sigue el tramo con más enemigos y otra puerta al final.
3. **Dentro del árbol** (3) — sala **vertical**: hay que **volar hacia arriba**
   (salta con `A` y mantén `A` para flotar) esquivando enemigos.
4. **Whispy Woods** (4) — aparece el jefe.

Capturas recomendadas:

| Tecla | Cuándo | Qué es |
|---|---|---|
| `F1` | Al empezar, sin moverte | `spawn` (sala 0, jefe 0) |
| `F2` | Justo después de derrotar a Poppy Bros. Sr. | `mid` (sala 2, jefe 0) |
| `F3` | Nada más entrar al árbol, abajo, antes de subir | `door` (sala 3, jefe 0) |
| `F4` | Cuando ya ves a Whispy Woods | `arena` (sala 4, jefe 6) |

**Uniformidad entre etapas.** Graba las cuatro en **una sola partida sin
morir** y con la **vida llena (hp = 6)** en cada captura, para que todas
empiecen con la misma vida y el mismo contador de vidas en el HUD (el HUD es
parte de la observación). El script avisa si una captura difiere. No corrijas
vida o vidas escribiendo en la RAM: el HUD no se redibuja y la pantalla queda
incoherente. Truco: pulsa `Z` antes de un tramo difícil y `X` para volver ahí
si te golpean (es el estado interno del emulador, no afecta a las capturas).

La tecla `F3` es la más importante: el agente aprenderá a salir del árbol y
entrar a la arena desde ahí. Si te equivocas, vuelve a pulsar la tecla en el
punto correcto: se sobrescribe y el anterior queda como `.bak`.

> Si te atascas en el tramo 4, recuerda que se sube volando: salta (`A`), y
> mientras estás en el aire vuelve a pulsar `A` para inflarte y flotar.

### Manual

```python
from pyboy import PyBoy

pyboy = PyBoy("roms/kirby.gb", window="SDL2")
pyboy.set_emulation_speed(1)

# Jugar manualmente hasta el punto deseado y luego:
with open("saves/phase0_whispy.state", "wb") as fh:
    pyboy.save_state(fh)

pyboy.stop()
```

Repetir para cada etapa. Verificar después:

```python
with open("saves/phase0_whispy.state", "rb") as fh:
    pyboy.load_state(fh)
```

## Validación

1. Cargar el state dos veces y comprobar que pantalla y RAM son idénticas.
2. Confirmar que el punto es estable (el agente no muere sin recibir input).
3. Anotar el **ID de sala** en `env.room_ids` de `configs/base.yaml`: el entorno
   lo necesita para detectar cuándo se cumple la meta de la etapa.
4. Confirmar que `boss_hp` es distinto de 0 al inicio de la arena (si vale 0, la
   dirección de RAM es incorrecta y el episodio "ganaría" en el primer paso).
5. Registrar junto a los states la versión de PyBoy usada.

Todo lo anterior está automatizado:

```powershell
python tools/verify_saves.py
```

Salida esperada: `RESULTADO: TODO OK` y las cuatro capturas en
`saves/previews/`.

> Fijar la versión exacta de PyBoy en `requirements.txt` una vez validados los
> states: un cambio de versión puede alterar el formato interno y dejarlos
> ilegibles.
