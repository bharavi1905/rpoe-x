from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import Callable, Tuple

from models import (
    OrchestratorAction, OrchestratorObs,
    ZoneAction, ZoneObs, TaskResult,
)
from server.env import RPOEXEnv, _open_score, ZONES

# ---------------------------------------------------------------------------
# PART A — Greedy baseline agents
# ---------------------------------------------------------------------------

def greedy_orchestrator(obs: OrchestratorObs) -> OrchestratorAction:
    """
    Route to the zone with the lowest combined score of:
      queue_length + 10 * occupancy
    Tiebreak: lower zone_id wins.
    """
    best_zone = min(
        range(5),
        key=lambda z: obs.zone_queue_lengths[z] + 10.0 * obs.zone_occupancy[z]
    )
    return OrchestratorAction(action="route_to_zone", zone_id=best_zone)


def greedy_zone(obs: ZoneObs) -> ZoneAction:
    """
    Assign to the wheel with the lowest combined score of:
      wheel_queue_lengths[w] + est_rotation_cost[w]
    Tiebreak: lower wheel_id wins.
    """
    best_wheel = min(
        range(len(obs.wheel_occupancy)),
        key=lambda w: obs.wheel_queue_lengths[w] + obs.est_rotation_cost[w]
    )
    return ZoneAction(action="assign_to_wheel", wheel_id=best_wheel)


# ---------------------------------------------------------------------------
# PART B — Task runner type alias and shared runner
# ---------------------------------------------------------------------------

# Agent callable type:
# agent(obs, env) -> (OrchestratorAction, zone_id, ZoneAction)
AgentFn = Callable[[OrchestratorObs, RPOEXEnv], Tuple[OrchestratorAction, int, ZoneAction]]


def _run_episode(
    agent: AgentFn,
    seed: int,
    max_steps: int,
) -> dict:
    """
    Run one full episode. Returns raw metrics dict:
      total_parked, total_retrieved, total_overflowed,
      total_reward, step_count
    """
    env = RPOEXEnv(seed=seed, max_steps=max_steps)
    obs = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0
    while not obs.done:
        orch_action, zone_id, zone_action = agent(obs, env)
        obs = env.step(orch_action)
        total_reward += obs.reward
        step_count += 1
    return {
        "total_parked":     env._parked,
        "total_retrieved":  env._retrieved,
        "total_overflowed": env._overflowed,
        "total_reward":     round(total_reward, 6),
        "step_count":       step_count,
    }


# ---------------------------------------------------------------------------
# PART C — Task 1: easy (quiet demand, 200 steps)
# ---------------------------------------------------------------------------

TASK1_STEPS = 200


def run_task1(agent: AgentFn, seed: int = 42) -> TaskResult:
    """
    task1_easy — Quiet demand routing.
    Score = throughput_rate = total_parked / max_possible_parked
    max_possible_parked = TASK1_STEPS (one park per step upper bound)
    Pass threshold: 0.50
    """
    metrics = _run_episode(agent, seed=seed, max_steps=TASK1_STEPS)

    total_ops = metrics["total_parked"] + metrics["total_retrieved"]
    max_possible = TASK1_STEPS
    throughput_rate = total_ops / max_possible if max_possible > 0 else 0.0

    overflow_penalty = metrics["total_overflowed"] / max(1, metrics["total_parked"] + metrics["total_overflowed"])
    raw_score = throughput_rate * (1.0 - 0.5 * overflow_penalty)
    score = _open_score(raw_score)
    passed = score >= 0.50

    return TaskResult(
        task_id="task1_easy",
        score=score,
        metrics={
            "throughput_rate":  round(throughput_rate, 4),
            "overflow_penalty": round(overflow_penalty, 4),
            "total_parked":     float(metrics["total_parked"]),
            "total_retrieved":  float(metrics["total_retrieved"]),
            "total_overflowed": float(metrics["total_overflowed"]),
            "total_reward":     metrics["total_reward"],
        },
        passed=passed,
        notes=f"Quiet demand. throughput_rate={throughput_rate:.4f} overflow_penalty={overflow_penalty:.4f}",
    )


# ---------------------------------------------------------------------------
# PART A — Task 2: medium (peak hour surge, 400 steps)
# ---------------------------------------------------------------------------

TASK2_STEPS = 400


def run_task2(agent: AgentFn, seed: int = 42) -> TaskResult:
    """
    task2_medium — Peak hour surge (covers 8–10AM morning peak).
    Score = 0.60 * throughput_rate + 0.40 * balance_score
    balance_score = 1.0 - normalized zone imbalance
    Pass threshold: 0.55
    """
    import numpy as np

    env = RPOEXEnv(seed=seed, max_steps=TASK2_STEPS)
    obs = env.reset(seed=seed)
    total_reward = 0.0
    zone_occ_snapshots = []

    while not obs.done:
        orch_action, zone_id, zone_action = agent(obs, env)
        obs = env.step(orch_action)
        total_reward += obs.reward
        if env._step % 10 == 0:
            zone_occ_snapshots.append(list(obs.zone_occupancy))

    metrics_raw = {
        "total_parked":     env._parked,
        "total_retrieved":  env._retrieved,
        "total_overflowed": env._overflowed,
        "total_reward":     round(total_reward, 6),
    }

    total_ops = env._parked + env._retrieved
    max_possible = TASK2_STEPS
    throughput_rate = total_ops / max_possible if max_possible > 0 else 0.0

    if zone_occ_snapshots:
        imbalances = [float(np.std(snap)) for snap in zone_occ_snapshots]
        avg_imbalance = float(np.mean(imbalances))
    else:
        avg_imbalance = 0.0
    balance_score = max(0.0, 1.0 - avg_imbalance / 0.5)

    raw_score = 0.60 * throughput_rate + 0.40 * balance_score
    score = _open_score(raw_score)
    passed = score >= 0.55

    return TaskResult(
        task_id="task2_medium",
        score=score,
        metrics={
            "throughput_rate":  round(throughput_rate, 4),
            "balance_score":    round(balance_score, 4),
            "avg_imbalance":    round(avg_imbalance, 4),
            "total_parked":     float(env._parked),
            "total_retrieved":  float(env._retrieved),
            "total_overflowed": float(env._overflowed),
            "total_reward":     metrics_raw["total_reward"],
        },
        passed=passed,
        notes=f"Peak surge. throughput={throughput_rate:.4f} balance={balance_score:.4f}",
    )


# ---------------------------------------------------------------------------
# PART B — Task 3: hard (full 18-hour day, 1080 steps)
# ---------------------------------------------------------------------------

TASK3_STEPS = 1080


def run_task3(agent: AgentFn, seed: int = 42) -> TaskResult:
    """
    task3_hard — Full day simulation (covers all demand phases).
    Score = 0.50 * throughput_rate + 0.30 * balance_score + 0.20 * wait_score
    wait_score = 1.0 - clamp(avg_wait / 20.0, 0, 1)  # 20 steps = bad wait
    Pass threshold: 0.60
    """
    import numpy as np

    env = RPOEXEnv(seed=seed, max_steps=TASK3_STEPS)
    obs = env.reset(seed=seed)
    total_reward = 0.0
    zone_occ_snapshots = []
    wait_snapshots = []

    while not obs.done:
        orch_action, zone_id, zone_action = agent(obs, env)
        obs = env.step(orch_action)
        total_reward += obs.reward
        if env._step % 10 == 0:
            zone_occ_snapshots.append(list(obs.zone_occupancy))
            wait_snapshots.append(float(np.mean(obs.zone_avg_wait)))

    total_ops = env._parked + env._retrieved
    max_possible = TASK3_STEPS
    throughput_rate = total_ops / max_possible if max_possible > 0 else 0.0

    if zone_occ_snapshots:
        imbalances = [float(np.std(snap)) for snap in zone_occ_snapshots]
        avg_imbalance = float(np.mean(imbalances))
    else:
        avg_imbalance = 0.0
    balance_score = max(0.0, 1.0 - avg_imbalance / 0.5)

    avg_wait = float(np.mean(wait_snapshots)) if wait_snapshots else 0.0
    wait_score = max(0.0, 1.0 - min(avg_wait / 20.0, 1.0))

    raw_score = 0.50 * throughput_rate + 0.30 * balance_score + 0.20 * wait_score
    score = _open_score(raw_score)
    passed = score >= 0.60

    return TaskResult(
        task_id="task3_hard",
        score=score,
        metrics={
            "throughput_rate":  round(throughput_rate, 4),
            "balance_score":    round(balance_score, 4),
            "wait_score":       round(wait_score, 4),
            "avg_wait":         round(avg_wait, 4),
            "avg_imbalance":    round(avg_imbalance, 4),
            "total_parked":     float(env._parked),
            "total_retrieved":  float(env._retrieved),
            "total_overflowed": float(env._overflowed),
            "total_reward":     round(total_reward, 6),
        },
        passed=passed,
        notes=f"Full day. throughput={throughput_rate:.4f} balance={balance_score:.4f} wait={wait_score:.4f}",
    )


# ---------------------------------------------------------------------------
# PART C — TASKS registry
# ---------------------------------------------------------------------------

TASKS = {
    "task1_easy":   {"fn": run_task1, "steps": TASK1_STEPS, "threshold": 0.50},
    "task2_medium": {"fn": run_task2, "steps": TASK2_STEPS, "threshold": 0.55},
    "task3_hard":   {"fn": run_task3, "steps": TASK3_STEPS, "threshold": 0.60},
}


# ---------------------------------------------------------------------------
# PART D — Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def greedy_agent(obs, env):
        orch = greedy_orchestrator(obs)
        zone_obs = env.get_zone_obs(orch.zone_id)
        zone_act = greedy_zone(zone_obs)
        return orch, orch.zone_id, zone_act

    print("Running task1_easy...")
    r1 = run_task1(greedy_agent, seed=42)
    assert r1.task_id == "task1_easy"
    assert 0.001 <= r1.score <= 0.999
    print(f"  task1_easy  score={r1.score:.4f} passed={r1.passed}")

    print("Running task2_medium...")
    r2 = run_task2(greedy_agent, seed=42)
    assert r2.task_id == "task2_medium"
    assert 0.001 <= r2.score <= 0.999
    print(f"  task2_medium score={r2.score:.4f} passed={r2.passed}")

    print("Running task3_hard...")
    r3 = run_task3(greedy_agent, seed=42)
    assert r3.task_id == "task3_hard"
    assert 0.001 <= r3.score <= 0.999
    print(f"  task3_hard  score={r3.score:.4f} passed={r3.passed}")

    r1b = run_task1(greedy_agent, seed=42)
    assert r1.score == r1b.score, f"Determinism broken: {r1.score} != {r1b.score}"
    print("Determinism check passed.")

    print("ITEM 3.2 DONE")
