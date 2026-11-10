"""Loop de entrenamiento: currículo (forward/reverse), control y consolidación."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agents.ddqn import DDQNAgent
from .curriculum.scheduler import CurriculumScheduler, build_scheduler
from .curriculum.stages import build_sequence, describe_sequence, validate_saves
from .env.kirby_env import KirbyEnv
from .evaluate import EVAL_SEED, evaluate_policy
from .utils import (
    EpisodeLogger,
    EvalLogger,
    EpsilonSchedule,
    config_to_dict,
    load_config,
    resolve_device,
    run_directory,
    save_config,
    save_json,
    set_seed,
)

SPAWN = "spawn"
BOSS_GOAL = "boss"


@dataclass
class Phase:
    name: str
    frames: int
    stage: str | None
    goal: str | None
    scheduler: CurriculumScheduler | None
    epsilon_start: float
    epsilon_decay: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena DDQN con currículo forward/reverse")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-frames", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Fases
# --------------------------------------------------------------------------- #

def build_phases(cfg: dict[str, Any]) -> list[Phase]:
    agent_cfg = cfg["agent"]
    cur = cfg["curriculum"]
    total = int(cfg["train"]["total_frames"])

    eps_start = float(agent_cfg.get("epsilon_start", 1.0))
    eps_decay = int(agent_cfg.get("epsilon_decay_frames", 500_000))
    stage_start = float(agent_cfg.get("epsilon_stage_start", 0.3))
    stage_decay = int(agent_cfg.get("epsilon_stage_decay_frames", 150_000))

    if not cur.get("enabled", False):
        return [Phase("control", total, SPAWN, BOSS_GOAL, None, eps_start, eps_decay)]

    cur_frames = int(cur.get("frames", int(total * 0.75)))
    cons_frames = int(cur.get("consolidation_frames", total - cur_frames))
    if cur_frames + cons_frames > total:
        raise ValueError(
            f"curriculum.frames + consolidation_frames ({cur_frames + cons_frames}) "
            f"supera train.total_frames ({total})"
        )

    sequence = build_sequence(order=cur.get("order"), sequence=cur.get("sequence"))
    scheduler = build_scheduler(cur, sequence)

    phases = [
        Phase("curriculum", cur_frames, None, None, scheduler, eps_start, eps_decay)
    ]
    if cons_frames > 0:
        phases.append(
            Phase("consolidation", cons_frames, SPAWN, BOSS_GOAL, None, stage_start, stage_decay)
        )
    return phases


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.total_frames is not None:
        _override_budget(cfg, args.total_frames)

    try:
        run(cfg, resume=args.resume)
    except FileNotFoundError as exc:
        raise SystemExit(f"\nERROR: {exc}") from None
    except ValueError as exc:
        raise SystemExit(f"\nERROR de configuración: {exc}") from None


def _override_budget(cfg: dict[str, Any], total_frames: int) -> None:
    total = int(total_frames)
    cfg["train"]["total_frames"] = total
    if cfg["curriculum"].get("enabled"):
        cur_frames = int(total * 0.75)
        cfg["curriculum"]["frames"] = cur_frames
        cfg["curriculum"]["consolidation_frames"] = total - cur_frames


def run(cfg: dict[str, Any], resume: Path | None = None) -> None:
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(cfg["train"].get("device", "auto"))
    run_dir = run_directory(cfg)

    phases = build_phases(cfg)
    cfg["phases"] = [
        {"name": p.name, "frames": p.frames, "stage": p.stage, "goal": p.goal}
        for p in phases
    ]
    save_config(cfg, run_dir / "config.yaml")

    if cfg["curriculum"].get("enabled"):
        sequence = phases[0].scheduler.stage_ids  # type: ignore[union-attr]
        print(describe_sequence(sequence))
        missing = validate_saves(sequence, cfg["env"]["saves_dir"])
        if missing:
            print("\nADVERTENCIA: faltan save states:")
            for path in missing:
                print(f"  - {path}")
        print()

    seed = int(cfg.get("seed", 0))
    env = KirbyEnv(cfg["env"])
    obs_shape = env.observation_space.shape
    agent = DDQNAgent(obs_shape, env.action_space.n, cfg["agent"], device=device, seed=seed)

    start_frames = 0
    episode = 0
    saved_loop: dict[str, Any] = {}
    if resume is not None:
        payload = agent.load(resume)
        start_frames = int(payload.get("total_frames", 0))
        episode = int(payload.get("episode", 0))
        saved_loop = dict(payload.get("loop") or {})
        if "scheduler" in payload and phases[0].scheduler is not None:
            phases[0].scheduler.load_state_dict(payload["scheduler"])
        print(
            f"Reanudando desde {resume} (frames={start_frames}, episodio={episode}). "
            "Nota: el replay buffer no se guarda y se rellena desde cero."
        )
        env.total_frames = start_frames

    train_cfg = cfg["train"]
    eval_every = int(train_cfg.get("eval_every_frames", 250_000))
    eval_episodes = int(train_cfg.get("eval_episodes", 10))
    final_eval_episodes = int(train_cfg.get("final_eval_episodes", 100))
    ckpt_every = int(train_cfg.get("checkpoint_every_frames", 250_000))
    log_every = int(train_cfg.get("log_every_episodes", 10))
    train_freq = int(cfg["agent"].get("train_freq", 4))

    eval_env: KirbyEnv | None = None
    # Evaluaciones y checkpoints en una grilla fija (múltiplos de eval_every),
    # también al reanudar, para que las curvas de distintas semillas se alineen.
    next_eval = int(saved_loop.get("next_eval", (start_frames // eval_every + 1) * eval_every))
    next_ckpt = int(saved_loop.get("next_ckpt", (start_frames // ckpt_every + 1) * ckpt_every))
    # Estado del bucle que se guarda en cada checkpoint (para reanudar epsilon).
    loop: dict[str, Any] = {"next_eval": next_eval, "next_ckpt": next_ckpt}
    first_reset = True

    print(f"Salida: {run_dir}")
    print(f"Presupuesto: {train_cfg['total_frames']} frames | device={device}\n")

    episodes_log = EpisodeLogger(run_dir, append=resume is not None)
    evals_log = EvalLogger(run_dir, append=resume is not None)

    completed = False
    try:
        cursor = 0
        for phase in phases:
            phase_start, phase_end = cursor, cursor + phase.frames
            cursor = phase_end

            if env.total_frames >= phase_end:
                continue  # fase ya completada (resume)

            if saved_loop.get("phase") == phase.name:
                # Reanudar con el mismo schedule de epsilon (incluye reinicios por etapa).
                eps_start = float(saved_loop["eps_start"])
                eps_decay = int(saved_loop["eps_decay"])
                eps_origin = int(saved_loop["eps_origin"])
                saved_loop = {}
            else:
                eps_start, eps_decay = phase.epsilon_start, phase.epsilon_decay
                eps_origin = max(env.total_frames, phase_start)
            eps = EpsilonSchedule(eps_start, _eps_end(cfg), eps_decay)
            loop.update(
                phase=phase.name, eps_start=eps_start, eps_decay=eps_decay, eps_origin=eps_origin
            )
            print(f"== Fase {phase.name} ({phase.frames} frames) ==")

            while env.total_frames < phase_end:
                if phase.scheduler is not None:
                    stage = phase.scheduler.select_stage()
                    goal = None
                else:
                    stage = phase.stage or SPAWN
                    goal = phase.goal

                # Se siembra solo el primer reset del proceso; luego el RNG del
                # entorno avanza solo (no-ops distintos en cada episodio).
                reset_seed = seed * 1_000_003 + episode if first_reset else None
                first_reset = False
                obs, info = env.reset(seed=reset_seed, options={"stage": stage, "goal": goal})

                ep_return = 0.0
                last_loss: float | None = None
                epsilon = eps.value(env.total_frames - eps_origin)
                done = False

                while not done:
                    epsilon = eps.value(env.total_frames - eps_origin)
                    action = agent.act(obs, epsilon)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # Solo `terminated` corta el bootstrap: un timeout no es un
                    # estado terminal del MDP.
                    agent.push(obs, action, reward, next_obs, terminated)
                    if env.episode_steps % train_freq == 0:
                        loss = agent.update()
                        if loss is not None:
                            last_loss = loss

                    obs = next_obs
                    ep_return += reward

                episode += 1
                won = bool(info.get("won", False))

                if phase.scheduler is not None:
                    phase.scheduler.add_frames(stage, env.episode_frames)
                    phase.scheduler.record_episode(stage, won, epsilon)
                    new_stage = phase.scheduler.maybe_advance()
                    if new_stage is not None:
                        eps_start, eps_decay = _stage_eps_start(cfg), _stage_eps_decay(cfg)
                        eps = EpsilonSchedule(eps_start, _eps_end(cfg), eps_decay)
                        eps_origin = env.total_frames
                        loop.update(eps_start=eps_start, eps_decay=eps_decay, eps_origin=eps_origin)
                        print(
                            f"[frame {env.total_frames}] avance -> {new_stage} "
                            f"(epsilon reiniciado a {_stage_eps_start(cfg)})"
                        )

                episodes_log.log(
                    frame=env.total_frames,
                    episode=episode,
                    phase=phase.name,
                    stage=stage,
                    ep_return=round(ep_return, 4),
                    length=env.episode_steps,
                    won=int(won),
                    epsilon=round(epsilon, 4),
                    loss=round(last_loss, 6) if last_loss is not None else "",
                    curriculum_stage=(
                        phase.scheduler.current_stage if phase.scheduler else (phase.stage or "")
                    ),
                    win_rate_stage=(
                        round(phase.scheduler.win_rate(stage), 4)
                        if phase.scheduler
                        else ""
                    ),
                )

                if episode % log_every == 0:
                    rate = (
                        phase.scheduler.win_rate(stage) if phase.scheduler else float("nan")
                    )
                    print(
                        f"ep {episode:<6} frame {env.total_frames:<8} {phase.name:<13} "
                        f"{stage:<6} ret {ep_return:8.2f} won {int(won)} "
                        f"eps {epsilon:.3f} win {rate:.2%}"
                    )

                # ------------------------------------------------------ #
                # Evaluación full-level
                # ------------------------------------------------------ #
                if env.total_frames >= next_eval:
                    eval_env, metrics = _run_eval(
                        agent, eval_env, cfg, next_eval, env.total_frames,
                        episode, phase.name, eval_episodes,
                    )
                    evals_log.log(**metrics)
                    print(
                        f"  [eval] frame {metrics['frame']}: "
                        f"win_rate {metrics['win_rate']:.2%} "
                        f"retorno {metrics['return_mean']:.2f}"
                    )
                    next_eval += eval_every
                    loop["next_eval"] = next_eval

                if env.total_frames >= next_ckpt:
                    next_ckpt += ckpt_every
                    loop["next_ckpt"] = next_ckpt
                    _save_checkpoint(agent, phases, cfg, run_dir, env, episode, loop)

            _save_checkpoint(agent, phases, cfg, run_dir, env, episode, loop)
            if phase.scheduler is not None:
                print(phase.scheduler.summary())

        # -------------------------------------------------------------- #
        # Evaluación final (100 episodios): base para S_final y P2
        # -------------------------------------------------------------- #
        eval_env, metrics = _run_eval(
            agent, eval_env, cfg, int(train_cfg["total_frames"]), env.total_frames,
            episode, "final", final_eval_episodes,
        )
        evals_log.log(**metrics)
        save_json(metrics, run_dir / "final_eval.json")
        print(
            f"\n[final] {final_eval_episodes} episodios: "
            f"win_rate {metrics['win_rate']:.2%} "
            f"retorno {metrics['return_mean']:.2f}"
        )
        completed = True

    except KeyboardInterrupt:
        print("\nInterrumpido: guardando checkpoint final...")
    finally:
        if env.total_frames > 0:
            _save_checkpoint(agent, phases, cfg, run_dir, env, episode, loop)
        episodes_log.close()
        evals_log.close()
        env.close()
        if eval_env is not None:
            eval_env.close()
        if completed:
            print(f"\nListo. Logs en {run_dir}")


# --------------------------------------------------------------------------- #

def _eps_end(cfg: dict[str, Any]) -> float:
    return float(cfg["agent"].get("epsilon_end", 0.05))


def _stage_eps_start(cfg: dict[str, Any]) -> float:
    return float(cfg["agent"].get("epsilon_stage_start", 0.3))


def _stage_eps_decay(cfg: dict[str, Any]) -> int:
    return int(cfg["agent"].get("epsilon_stage_decay_frames", 150_000))


def _run_eval(
    agent: DDQNAgent,
    eval_env: KirbyEnv | None,
    cfg: dict[str, Any],
    frame: int,
    frame_actual: int,
    episode: int,
    phase: str,
    episodes: int,
) -> tuple[KirbyEnv, dict[str, Any]]:
    """Evaluación greedy; por defecto full-level (spawn -> jefe).

    `train.eval_stage` / `train.eval_goal` permiten evaluar una sola etapa
    (p. ej. el experimento `arena_only`). Los grupos del experimento principal
    no los definen, así que siguen evaluando el nivel completo.

    `frame` es el punto nominal de la grilla (múltiplo de eval_every) y
    `frame_actual` el contador real al evaluar.
    """
    if eval_env is None:
        eval_env = KirbyEnv(cfg["env"])

    stage = cfg["train"].get("eval_stage") or SPAWN
    # None = la meta local de la etapa (en arena: derrotar al jefe).
    goal = cfg["train"].get("eval_goal", BOSS_GOAL if stage == SPAWN else None)
    metrics = evaluate_policy(
        agent, eval_env, stage=stage, goal=goal, episodes=episodes, seed=EVAL_SEED
    )
    metrics.update(
        frame=int(frame), frame_actual=int(frame_actual), episode=episode, phase=phase
    )
    return eval_env, metrics


def _save_checkpoint(
    agent: DDQNAgent,
    phases: list[Phase],
    cfg: dict[str, Any],
    run_dir: Path,
    env: KirbyEnv,
    episode: int,
    loop: dict[str, Any] | None = None,
) -> None:
    extra: dict[str, Any] = {
        "config": config_to_dict(cfg),
        "total_frames": env.total_frames,
        "episode": episode,
        "loop": dict(loop or {}),
    }
    scheduler = phases[0].scheduler if phases else None
    if scheduler is not None:
        extra["scheduler"] = scheduler.state_dict()

    path = run_dir / "checkpoints" / "last.pt"
    agent.save(path, extra=extra)

    tagged = run_dir / "checkpoints" / f"frame_{env.total_frames:09d}.pt"
    if not tagged.exists():
        agent.save(tagged, extra=extra)

    print(f"  [checkpoint] {path} ({env.total_frames} frames)")


if __name__ == "__main__":
    main()
