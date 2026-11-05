# Comparación de Currículos Directo e Inverso para Kirby's Dream Land

> Documento de contexto y guía de desarrollo.
> **Origen:** `Presentación proyecto semestral RL.pdf` (8 diapositivas).
> **Alcance actualizado:** el objetivo original era aplicar *Reverse Curriculum
> Generation*; el foco pasó a ser una **comparación experimental entre currículo
> directo (forward) y currículo inverso (reverse)**.
> Las secciones marcadas como **[Propuesta]** son extensiones técnicas del equipo,
> no contenido literal de la presentación.

---

## 1. Ficha del proyecto

| Campo | Detalle |
|---|---|
| **Título** | Comparación de Curriculum Learning directo e inverso en Kirby's Dream Land |
| **Objeto de estudio** | Aprendizaje por Refuerzo Profundo (Deep RL) sobre PyBoy |
| **Juego** | Kirby's Dream Land (Game Boy, 1992) |
| **Tarea objetivo** | Superar Green Greens completo y derrotar a Whispy Woods |
| **Comparación central** | Forward curriculum vs. Reverse curriculum (misma descomposición, mismo presupuesto) |
| **Métricas primarias** | Sample efficiency (P1) y tiempo de convergencia (P2) |
| **Integrantes** | Tomás Cárdenas · Daniel Soto · Franco Ponce |
| **Profesor / Ayudante** | Julio Godoy / Mirko Peñailillo |
| **Duración** | 8 semanas (4 fases de 2 semanas) |

---

## 2. Objetivo e hipótesis

**Objetivo:** determinar si el orden en que se presentan las subtareas de un nivel
de plataformas afecta la eficiencia de muestra y la estabilidad del aprendizaje,
manteniendo constantes el algoritmo, la recompensa, la arquitectura y el
presupuesto de interacciones.

**Hipótesis:**

- **H1 (eficiencia):** el currículo inverso alcanza el umbral de competencia
  (80 % de éxito en el nivel completo) con menos frames que el directo.
- **H2 (olvido):** el olvido catastrófico es simétrico y opuesto: el currículo
  inverso degrada la habilidad de combate al aprender navegación, y el directo
  degrada la navegación al aprender el combate.
- **H3 (estabilidad):** el currículo inverso converge a su asíntota en menos
  frames, porque nunca entrena con recompensa inalcanzable.

**Resultado nulo aceptable:** si tras la fase de consolidación las diferencias
desaparecen, se reporta que el currículo acelera la fase inicial pero no el
desempeño asintótico. Ese también es un hallazgo válido.

---

## 3. Problema y motivación

Entrenar desde cero en un plataformas con recompensa esparsa es costoso: el
agente debe descubrir por exploración una secuencia larga de acciones antes de
recibir señal alguna. En Kirby's Dream Land el nivel combina scroll horizontal,
plataformas móviles, pozos letales, enemigos y una arena de jefe con patrón propio.

La línea base (`ModusMorris/Kirbys_Dreamland_RL`) usa DDQN sobre fotogramas
crudos y compensa la esparsidad con *reward shaping* invasivo. Este proyecto
ataca el mismo problema **reordenando las subtareas** en lugar de modificar la
función de retorno.

---

## 4. Currículo directo vs. inverso

Ambos currículos usan **la misma descomposición** del nivel y **la misma meta
local por etapa**. Solo cambia el orden.

### Descomposición (etapas por ID)

| ID | Save state | Contenido | Meta local |
|---|---|---|---|
| `spawn` | `phase3_spawn.state` | Sala 0: tramo exterior 1 | llegar a `mid` |
| `mid` | `phase2_mid.state` | Sala 2: tramo exterior 2 | llegar a `door` |
| `door` | `phase1_door.state` | Sala 3: interior del árbol | entrar a `arena` |
| `arena` | `phase0_whispy.state` | Sala 4: combate contra Whispy | derrotar al jefe |

### Secuencias

```
forward  = [spawn, mid, door, arena]     # del inicio hacia la meta
reverse  = [arena, door, mid, spawn]     # de la meta hacia el inicio
```

### Fundamento

- **Reverse** (Florensa et al., 2017): construir el currículo desde el estado
  objetivo hacia atrás. La distribución de estados iniciales `p₀(s)` se expande
  en dirección contraria a la meta. Cada etapa es resoluble con la política
  actual, así que la señal es densa *por construcción*, sin tocar la recompensa.
- **Forward** (Narvekar et al., 2020): el orden clásico, del inicio del nivel
  hacia la meta. El agente aprende primero las habilidades tempranas y enfrenta
  el combate al final.

> **Nota metodológica:** esta comparación es de **orden de subtareas**, no la
> formulación original de Florensa (que mantiene la meta final fija y solo mueve
> `p₀(s)`). Se eligió así para aislar la variable "dirección" con los mismos
> cuatro save states. Debe declararse en el informe.

### Consolidación

Ningún currículo entrena el nivel completo de principio a fin: reverse termina en
`spawn → mid` y forward termina en `arena → jefe`. Por eso ambos incluyen una
**fase de consolidación idéntica**: `spawn → jefe`, 500k frames. Sin ella, la
comparación final mediría la última etapa de cada currículo y no la tarea objetivo.

---

## 5. Metodología

### 5.1 Transición adaptativa

- Ventana deslizante de **N = 50 episodios** por etapa.
- Se avanza solo si `WinRate ≥ 85 %`.
- **Solo cuentan los episodios con `ε ≤ 0.10`**: evita que el umbral se alcance
  por suerte durante la exploración alta.

### 5.2 Exploración por etapa

- Primera etapa: `ε` de 1.0 a 0.05 en 500k frames.
- Cada transición reinicia `ε` a **0.3** y decae a 0.05 en 150k frames.
- Motivo: llegar a una sección nueva con `ε ≈ 0.05` impide descubrirla.

### 5.3 Muestreo mixto anti-olvido

- En cada `reset()`, con probabilidad **20 %** se reanuda en una etapa ya superada
  (posición anterior en la secuencia).
- Aplica igual a forward y reverse, para no confundir dirección con anti-olvido.
- Se registra cuántos frames aporta cada etapa: el 20 % de *episodios* no es el
  20 % de *experiencia*, porque la arena dura mucho menos que la navegación.

### 5.4 Recompensa

Esparsa y sin shaping: `+10` por cumplir la meta, `-5` por muerte, `-0.5` por
daño. Idéntica en los tres grupos (hay un test que lo verifica).

---

## 6. Diseño experimental

| Grupo | Secuencia | Presupuesto | Consolidación |
|---|---|---|---|
| **Control** | — (nivel completo desde `spawn`) | 2M frames | — |
| **Forward** | spawn → mid → door → arena | 1.5M frames | 500k frames (`spawn → jefe`) |
| **Reverse** | arena → door → mid → spawn | 1.5M frames | 500k frames (`spawn → jefe`) |

- **3 semillas** (0, 1, 2), las mismas en los tres grupos → comparación pareada.
- Misma arquitectura, hiperparámetros y recompensa (verificado por test).
- Evaluaciones **full-level desde `spawn`** cada 250k frames, 10 episodios, en los
  tres grupos: la curva es comparable porque no depende de la etapa actual.
- Evaluación final de 100 episodios para estimar `S_final`.

---

## 7. Métricas

### 7.1 Primarias

| Métrica | Definición operacional |
|---|---|
| **P1 — Sample efficiency** | Primer frame en que la evaluación full-level alcanza **≥ 80 %** de éxito, confirmado en la evaluación siguiente |
| **P2 — Tiempo de convergencia** | Primer frame desde el que la curva suavizada (media móvil de 3 evaluaciones) se mantiene **≥ 90 % de `S_final`**, y solo si `S_final ≥ 80 %` |

**Por qué dos:** P1 mide velocidad para volverse competente; P2 mide velocidad
para dejar de mejorar *una vez competente*. Un umbral absoluto en P2 lo haría
redundante con P1. La puerta `S_final ≥ 80 %` evita que un agente que converge a
un desempeño bajo gane la métrica por ser malo.

**Censura:** las corridas que no alcanzan el criterio se reportan como `> 2M`, no
se excluyen. Con 3 semillas se reporta mediana y rango, nunca media de las que
" sí convergieron".

### 7.2 Secundarias

- Éxito final (100 episodios, ε = 0).
- AUC de la curva de éxito full-level.
- Frame de primera victoria contra el jefe (favorece trivialmente a reverse;
  declararlo).
- **Retención simétrica**: win rate en `arena` antes/después de las etapas de
  navegación (reverse) y en `spawn` antes/después de la arena (forward).

### 7.3 Reporte

- Curva media por grupo con banda de desviación entre semillas.
- Tabla P1/P2 por semilla + mediana + censuradas (`python -m src.compare`).
- Plano (P1, P2) con un punto por corrida y el éxito final como color.

---

## 8. Arquitectura técnica

### 8.1 Estructura

```
proyectoRL/
├── PROYECTO_RL.md              # este documento
├── README.md
├── configs/
│   ├── base.yaml               # hiperparámetros compartidos
│   ├── control.yaml            # sin currículo, nivel completo
│   ├── forward.yaml            # currículo directo + consolidación
│   ├── reverse.yaml            # currículo inverso + consolidación
│   └── ablations/
│       ├── fixed_steps.yaml    # duración fija por etapa
│       └── no_replay.yaml      # sin muestreo mixto
├── saves/                      # 4 save states de PyBoy
├── roms/                       # ROM (no versionada)
├── src/
│   ├── curriculum/
│   │   ├── stages.py           # etapas, metas y secuencias
│   │   └── scheduler.py        # frontera, transición, muestreo mixto
│   ├── env/
│   │   ├── kirby_env.py        # wrapper Gymnasium
│   │   ├── ram.py              # variables de RAM
│   │   ├── rewards.py          # recompensa esparsa
│   │   └── preprocessing.py    # escala de grises + área + stacking
│   ├── agents/
│   │   ├── ddqn.py
│   │   └── replay_buffer.py
│   ├── train.py                # fases, consolidación, evals
│   ├── evaluate.py             # evaluación greedy
│   ├── metrics.py              # P1, P2, censura, retención
│   └── compare.py              # tabla comparativa por grupo
├── tests/                      # 60+ tests sin ROM
└── results/                    # logs, checkpoints, tablas
```

### 8.2 Flujo

```
┌──────────────┐  stage id   ┌───────────────┐
│  Scheduler   │ ──────────► │  KirbyEnv     │
│  seq[k], 20% │             │ load_state(id)│
└──────┬───────┘             └──────┬────────┘
       │ record(eps, won)           │ obs, r, done
       ▼                            ▼
┌──────────────┐             ┌───────────────┐
│  Transición  │             │  DDQN + buffer│
│  85% / eps   │             └───────┬───────┘
└──────────────┘                     │
                              ┌──────▼───────┐
                              │ eval full    │
                              │ 250k frames  │
                              └──────────────┘
```

### 8.3 Puntos de implementación

- **Metas por etapa:** `reach` (ID de sala) o `boss` (HP del jefe). El entorno
  falla rápido si la meta no es verificable, para no "ganar" por una dirección
  de RAM incorrecta.
- **Acciones:** 11, incluyendo combos (`right+a`, `right+b`, etc.). Sin ellos no
  se puede saltar ni absorber mientras se camina.
- **Observaciones:** 84×84 en escala de grises con **promedio de área**, no
  vecino más cercano, para no borrar sprites pequeños.
- **Descuento:** `γ = 0.997` (horizonte ~330 pasos). Con `γ = 0.99` y episodios
  de 4000 pasos la recompensa terminal es invisible desde el inicio.
- **Buffer:** 20.000 observaciones (~1.1 GB). Con 100k no cabe en RAM.
- **Checkpoint:** guarda agente **y** estado del scheduler.

---

## 9. Planificación (8 semanas)

| Fase | Semanas | Hitos | Responsable |
|---|---|---|---|
| **1. Setup y captura** | 1–2 | Reproducir ModusMorris; identificar RAM (HP, X/Y, sala, HP del jefe); generar y validar 4 save states | Equipo |
| **2. Implementación** | 3–4 | Wrapper Gymnasium, metas por etapa, scheduler por secuencia, fases y consolidación | Daniel Soto |
| **3. Experimentos** | 5–6 | 3 grupos × 3 semillas; ajuste de γ y ε; monitoreo de curvas | Franco Ponce |
| **4. Análisis y entrega** | 7–8 | P1/P2, curvas, retención, informe y demo | Tomás Cárdenas |

**Dependencias:** los save states y el ID de sala bloquean todo lo demás; la
evaluación full-level debe existir antes de lanzar las corridas.

**Cómputo estimado:** 9 corridas × 2M frames ≈ 17–25 h a ~300 fps. Lanzar una
semilla de humo antes de las 9.

---

## 10. Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| No se identifica el ID de sala | Alto | Fallback a umbral de X; segundo fallback a firma de RAM contra el save state destino |
| `boss_hp` lee 0 fuera de la arena | Alto | Validación al iniciar el episodio (implementada) |
| Save states no reproducibles | Alto | Fijar versión de PyBoy; versionar los `.state`; test de determinismo |
| La consolidación borra las diferencias | Medio | Reportar también el estado al terminar el currículo, no solo el final |
| Forward es más difícil al inicio | Medio | Es la hipótesis, no un bug; declararlo |
| 20 % de episodios ≠ 20 % de frames | Medio | Registrar y reportar frames por etapa |
| Presupuesto insuficiente para 9 corridas | Alto | Humo con 1 semilla y `--total-frames` reducido |

---

## 11. Checklist

**Fase 1**
- [ ] Repo base clonado y ejecutando
- [ ] Direcciones de RAM documentadas en `src/env/ram.py`
- [x] `room_ids` reales en `configs/base.yaml` (0/2/3/4 vía `wCurStageScreen`)
- [ ] 4 save states validados y deterministas
- [ ] Carga de cada state verifica la meta al iniciar

**Fase 2**
- [ ] `KirbyEnv` con metas `reach`/`boss`
- [ ] `CurriculumScheduler` con secuencia y filtro ε
- [ ] Fases y consolidación integradas
- [ ] Checkpoint con agente + scheduler

**Fase 3**
- [ ] 3 configs verificadas (solo cambia el orden)
- [ ] 3 semillas por grupo ejecutadas
- [ ] γ y ε ajustados y documentados

**Fase 4**
- [ ] `python -m src.compare` genera la tabla P1/P2
- [ ] Curvas por grupo con banda de desviación
- [ ] Retención simétrica calculada
- [ ] Informe y demo

---

## 12. Referencias

1. Florensa, C., Held, D., Wulfmeier, M., Zhang, M., & Abbeel, P. (2017). *Reverse Curriculum Generation for Reinforcement Learning*. CoRL.
2. Narvekar, S., Peng, B., Leonetti, M., Sinapov, J., Taylor, M. E., & Stone, P. (2020). *Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey*. JMLR.
3. ModusMorris. *Kirbys_Dreamland_RL* — línea base DDQN sobre PyBoy.
4. PyBoy: https://github.com/Baekalfen/PyBoy
5. Kirby's Dream Land (1992), HAL Laboratory / Nintendo.

---

## 13. Glosario

- **Forward curriculum:** ordenar las subtareas del inicio del nivel hacia la meta.
- **Reverse curriculum:** ordenarlas desde la meta hacia el inicio.
- **Save state:** instantánea completa del emulador en un punto exacto.
- **Consolidación:** fase final común que entrena el nivel completo tras el currículo.
- **P1:** frames hasta alcanzar el 80 % de éxito full-level.
- **P2:** frames hasta estabilizarse en el 90 % del desempeño final propio.
- **Censura:** corrida que no alcanza el criterio dentro del presupuesto.
- **Muestreo mixto:** reanudar ocasionalmente en etapas ya superadas para evitar olvido.
- **Retención:** desempeño que sobrevive en una etapa antigua tras avanzar.
