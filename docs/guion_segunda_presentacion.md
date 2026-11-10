# Guion — Segunda presentación de avances (Kirby RL)

Límite: 10 minutos · Duración estimada: 9:10 (≈50 s de margen) · Ritmo calculado: ~140 palabras/min

## Reparto

| # | Diapositiva | Quién | Duración | Minuto |
|---|---|---|---|---|
| 1 | Portada | Tomás | 0:20 | 0:00–0:20 |
| 2 | Recordatorio e hipótesis | Tomás | 1:00 | 0:20–1:20 |
| 3 | Métodos implementados | Daniel | 1:00 | 1:20–2:20 |
| 4 | Currículo forward/reverse | Daniel | 1:00 | 2:20–3:20 |
| 5 | Validación del entorno | Daniel | 0:50 | 3:20–4:10 |
| 6 | Aprendizajes | Daniel | 0:50 | 4:10–5:00 |
| 7 | Corrida de humo | Franco | 0:40 | 5:00–5:40 |
| 8 | Experimento en arena | Franco | 1:20 | 5:40–7:00 |
| 9 | Correcciones | Franco | 1:00 | 7:00–8:00 |
| 10 | Planificación | Tomás | 0:50 | 8:00–8:50 |
| 11 | Cierre y preguntas | Tomás | 0:20 | 8:50–9:10 |

Total por persona: Daniel 3:40 · Franco 3:00 · Tomás 2:30

---

## 1. Portada — TOMÁS · 0:20 · minuto 0:00–0:20

Buenas, somos Tomás, Daniel y Franco. Les vamos a mostrar el segundo avance de nuestro proyecto: comparar un currículo directo contra uno inverso para entrenar un agente DDQN en Kirby’s Dream Land, en el nivel Green Greens completo, hasta Whispy Woods. Partimos con un recordatorio rápido.

## 2. Recordatorio e hipótesis — TOMÁS · 1:00 · minuto 0:20–1:20

La pregunta es si importa el orden en que se enseñan las subtareas. Dividimos el nivel en cuatro etapas y entrenamos el mismo agente, con el mismo algoritmo, recompensa, arquitectura y presupuesto de frames. Lo único que cambia es el orden: forward va del inicio al jefe; reverse parte desde el jefe y retrocede.

Tenemos tres hipótesis. H1, eficiencia: reverse alcanza 80 % de éxito en el nivel completo con menos frames. H2, olvido: el olvido es simétrico, cada currículo degrada la habilidad que aprendió primero. H3, estabilidad: reverse converge antes, porque nunca entrena con una recompensa inalcanzable.

También aceptamos un resultado nulo: que el currículo solo acelere el inicio y no el desempeño final. Eso también responde la pregunta.

→ Pase: «Daniel les cuenta qué tenemos implementado.»

## 3. Métodos implementados — DANIEL · 1:00 · minuto 1:20–2:20

Gracias. Hoy el pipeline corre completo, de punta a punta.

Primero, un entorno Gymnasium: la interfaz estándar de RL, con reset y step, montada sobre el emulador PyBoy. El agente elige entre 11 acciones y ve cuatro imágenes de 84 por 84 en escala de grises.

Segundo, leemos la RAM del juego: HP, vidas, sala y vida del jefe, con direcciones verificadas contra la ROM. Tenemos cuatro save states deterministas, uno por etapa, y con eso definimos la meta de cada etapa: llegar a una sala o dejar al jefe sin vida.

El agente es un Double DQN con gamma 0.997 y recompensa esparsa: +10 por cumplir la meta, −5 por morir y −0.5 por recibir daño. Encima van el scheduler de currículo y las métricas. Todo está cubierto por 73 tests automáticos, y uno de ellos verifica que la recompensa sea idéntica entre grupos.

## 4. Currículo forward/reverse — DANIEL · 1:00 · minuto 2:20–3:20

Las cuatro etapas son: spawn, llegar a la sala 2 y vencer al mini-jefe Poppy Bros. Sr.; mid, llegar al interior del árbol; door, subir el árbol y entrar a la arena; y arena, derrotar a Whispy Woods.

Forward las recorre en ese orden y reverse al revés. Una etapa se da por superada con 85 % de victorias en 50 episodios, contando solo episodios con poca exploración. Cada etapa nueva reinicia epsilon a 0.3 y, para evitar el olvido, el 20 % de los reinicios vuelve a una etapa ya superada.

Cada currículo usa 1.5 millones de frames y después 500 mil de consolidación en el nivel completo, idéntica para ambos. El control entrena 2 millones de frames sin currículo.

Una aclaración: comparamos el orden de las subtareas, no la formulación original de Florensa, que fija la meta y solo mueve el punto de partida. Eso lo declaramos en el informe.

## 5. Validación del entorno — DANIEL · 0:50 · minuto 3:20–4:10

Antes de entrenar verificamos que cada etapa sea lo que dice ser. Un script revisa en cada save state la sala esperada, la vida del jefe, que sea determinista y que se mantenga estable 120 frames sin input. Los cuatro pasan. (Si preguntan: la sala 1 no aparece porque nunca se visita en el camino principal.)

El dato más importante es este: en la primera versión, la etapa door partía justo frente a la puerta y una política aleatoria ganaba el 84 % de las veces. La etapa era trivial. La recapturamos en la entrada de la sala, y ahora el azar gana 0 %.

Además, el entorno corre a unos 1200 frames por segundo sin entrenar.

## 6. Aprendizajes — DANIEL · 0:50 · minuto 4:10–5:00

Ese no fue el único problema. Aquí resumimos lo que encontramos al validar con la ROM. Dos son graves, porque habrían invalidado la comparación sin dar ningún error: la etapa door trivial, y que el mini-jefe Poppy Bros. Sr. no se entrenaba en ninguna etapa. Ahora la meta de spawn exige vencerlo.

Otros ajustes: como PyBoy es determinista, agregamos entre 0 y 30 frames sin input en cada reinicio; subimos gamma a 0.997 para que la recompensa final se alcance a ver; y los timeouts ya no cortan el bootstrap.

La lección: cada dirección de RAM y cada save state hay que verificarlos empíricamente.

→ Pase: «Franco les muestra los primeros resultados de entrenamiento.»

## 7. Corrida de humo — FRANCO · 0:40 · minuto 5:00–5:40

Primero corrimos una prueba de humo: reverse, semilla 0, solo 80 mil frames. No es un experimento; sirve para comprobar que el ciclo completo funciona: currículo, consolidación, evaluación final, checkpoints y logs. Y funciona sin errores.

Lo que no demuestra es aprendizaje: 0 % de victorias y un retorno medio de −7 en la arena. Es lo esperable, porque el agente casi no alcanzó a actualizarse y epsilon seguía en 0.89. Todavía no hay corridas del experimento principal.

## 8. Experimento en arena — FRANCO · 1:20 · minuto 5:40–7:00

Para saber si el agente aprende algo, entrenamos solo en la arena, contra Whispy Woods, durante 250 mil frames.

El resultado: sobrevive más, pero nunca gana. Los episodios pasaron de 238 a 352 pasos en promedio, pero ganó 0 de 235 episodios, y la evaluación final da 0 %.

El gráfico muestra el retorno de la evaluación greedy cada 25 mil frames. Oscila entre dos comportamientos: arriba, se esconde hasta que se acaba el tiempo, con retorno cero; abajo, muere en unos 250 pasos, con retorno −7.5. La política final se esconde.

Con nuestra recompensa eso es racional: esconderse vale 0 y morir vale −5. Y el agente nunca vio la recompensa de victoria: jugando al azar, Whispy recibe a lo más 2 de los 6 golpes necesarios.

Esto importa porque el currículo inverso empieza justamente por la arena. Hoy, reverse partiría atascado.

## 9. Correcciones — FRANCO · 1:00 · minuto 7:00–8:00

Por eso proponemos estas correcciones. La principal es A: empezar más cerca de la meta, con save states donde Whispy tiene 1 o 2 de vida, luego 4 y luego 6. Es la idea original de Florensa y mantiene la recompensa esparsa. La combinamos con C: penalizar el timeout igual que morir, para eliminar el óptimo local de esconderse.

Dos temas más. El filtro de epsilon impide superar la primera etapa antes de unos 450 mil frames, y eso deja poco presupuesto para las otras tres. Y el cómputo: a unos 160 fps en CPU, cada corrida toma unas 3 horas y media, así que hay que acotar las evaluaciones.

La alternativa B, dar +1 por cada golpe al jefe, funciona pero es shaping; si la usamos, sería idéntica en los tres grupos.

→ Pase: «Tomás cierra con la planificación.»

## 10. Planificación — TOMÁS · 0:50 · minuto 8:00–8:50

Mantenemos el reparto de la primera presentación: Daniel lidera implementación, Franco experimentos y yo análisis e informe.

En la semana 5, Daniel aplica las correcciones A y C en la arena hasta ver victorias, Franco lanza los tres grupos con semilla 0 y yo armo la estructura del informe y el marco teórico. En la 6 corremos las semillas 1 y 2 y calculamos P1 y P2 sobre resultados parciales. En la 7 vienen las curvas comparativas, la retención y las ablaciones, si alcanza el cómputo. En la 8, la sección de resultados, la demo en video y el informe final.

La dependencia crítica: no lanzamos las corridas definitivas hasta que la arena sea resoluble.

(Pendiente: ajustar las semanas al calendario real del curso.)

## 11. Cierre y preguntas — TOMÁS · 0:20 · minuto 8:50–9:10

Entonces, el próximo hito es concreto: la primera victoria contra Whispy y, después, las 9 corridas, tres grupos por tres semillas, para comparar forward contra reverse. Muchas gracias. ¿Preguntas?

(Quedan unos 50 segundos de margen sobre los 10 minutos.)
