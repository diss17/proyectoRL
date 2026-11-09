# Forward vs. Reverse Curriculum — Kirby's Dream Land

Comparación experimental entre **curriculum learning directo (forward)** y
**reverse curriculum generation** sobre DDQN en Kirby's Dream Land (Game Boy,
1992), emulado con **PyBoy**. La tarea objetivo es superar Green Greens completo
y derrotar a Whispy Woods.

> Contexto, hipótesis, diseño experimental y planificación: [`PROYECTO_RL.md`](PROYECTO_RL.md)

---

## Estado del repositorio

| Componente | Estado |
|---|---|
| Diseño experimental (3 grupos × 3 semillas) | ✅ |
| Etapas por ID + secuencias forward/reverse | ✅ |
| Scheduler por secuencia (85 %, ventana 50, ε ≤ 0.1) | ✅ |
| Muestreo mixto anti-olvido (20 %) | ✅ |
| Fases de currículo + consolidación | ✅ |
| Métricas P1 (sample efficiency) y P2 (convergencia) | ✅ |
| Tabla comparativa (`src/compare.py`) | ✅ |
| Wrapper Gymnasium con metas por etapa | 🟡 requiere ROM |
| Mapa de RAM (`room_ids`, HP, coords) | ✅ verificado |
| Save states de las 4 etapas | ✅ verificados |

### Experimentos preliminares

| Experimento | Resultado | Documento |
|---|---|---|
| Solo arena, 250k frames, semilla 0 | 0 % de victorias; la política aprende a esconderse. ~160 fps en CPU | [`docs/experimento_arena_only.md`](docs/experimento_arena_only.md) |

---

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Además:

1. ROM legal de *Kirby's Dream Land* en `roms/kirby.gb`.
2. Los cuatro save states en `saves/` (ver [`saves/README.md`](saves/README.md)).
   Generación asistida: `python tools/capture_saves.py`.

---

## Los tres grupos

| Grupo | Secuencia | Presupuesto |
|---|---|---|
| `control` | nivel completo desde `spawn` | 2M frames |
| `forward` | spawn → mid → door → arena + consolidación | 1.5M + 0.5M |
| `reverse` | arena → door → mid → spawn + consolidación | 1.5M + 0.5M |

Las tres configs comparten agente, recompensa y presupuesto; un test verifica que
`forward.yaml` y `reverse.yaml` solo difieren en el orden.

---

## Inicio rápido

```powershell
# Verificar todo sin ROM ni emulador (63 tests)
pytest tests/ -v

# Entrenar
python -m src.train --config configs/control.yaml --seed 0
python -m src.train --config configs/forward.yaml --seed 0
python -m src.train --config configs/reverse.yaml --seed 0

# Humo con presupuesto reducido
python -m src.train --config configs/reverse.yaml --seed 0 --total-frames 200000

# Evaluar un checkpoint (100 episodios, ε = 0)
python -m src.evaluate --checkpoint results/reverse_s0/checkpoints/last.pt --episodes 100

# Comparar los grupos
python -m src.compare --results results --out results/comparison.md
```

---

## Métricas

- **P1 — Sample efficiency:** primer frame con ≥ 80 % de éxito full-level,
  confirmado en la evaluación siguiente.
- **P2 — Convergencia:** primer frame desde el que la curva suavizada (3 evals) se
  mantiene ≥ 90 % de `S_final`; solo si `S_final ≥ 80 %`.
- **Secundarias:** éxito final (100 eps), AUC de la curva, retención simétrica.

Las corridas que no alcanzan el criterio se reportan como censuradas (`> 2M`),
no se excluyen.

---

## Estructura

```
proyectoRL/
├── PROYECTO_RL.md
├── configs/
│   ├── base.yaml
│   ├── control.yaml
│   ├── forward.yaml
│   ├── reverse.yaml
│   └── ablations/
├── saves/                  # 4 save states
├── roms/                   # ROM (no versionada)
├── src/
│   ├── curriculum/         # stages + scheduler
│   ├── env/                # wrapper, RAM, recompensa, preprocesado
│   ├── agents/             # DDQN + replay buffer
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── compare.py
├── tests/
└── results/
```

---

## Notas de implementación

- Las direcciones de RAM en `src/env/ram.py` y los `room_ids` en `configs/base.yaml`
  están **verificadas** sobre la ROM USA/Europe y los cuatro save states
  (`python tools/verify_saves.py`).
- El entorno **falla rápido** si la meta no es verificable, para que el agente no
  "gane" por una dirección incorrecta.
- En episodios full-level (`spawn -> jefe`), la meta `boss` solo se cumple
  después de haber visto al jefe con vida: `boss_hp = 0` al inicio del nivel no
  cuenta como victoria.
- Recompensa esparsa: el currículo sustituye al reward shaping invasivo.
- `γ = 0.997` y buffer de 20k: decisiones tomadas para que la recompensa terminal
  sea visible y el buffer quepa en RAM.
- Los checkpoints guardan agente **y** estado del scheduler.

---

## Referencias

- Florensa, C. et al. (2017). *Reverse Curriculum Generation for Reinforcement Learning*. CoRL.
- Narvekar, S. et al. (2020). *Curriculum Learning for Reinforcement Learning Domains*. JMLR.
- ModusMorris. *Kirbys_Dreamland_RL*.
- PyBoy: https://github.com/Baekalfen/PyBoy
