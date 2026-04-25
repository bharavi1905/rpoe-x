from __future__ import annotations

import json
import sys
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

import model_client
from model_client import API_BASE_URL, MODEL_NAME, HF_TOKEN  # Gate 2  # noqa: F401
try:
    from rpoe_x.models import OrchestratorObs, OrchestratorAction, ZoneObs, ZoneAction, TaskResult
    from rpoe_x.server.env import RPOEXEnv, _open_score
    from rpoe_x.tasks.graders import TASKS, greedy_orchestrator, greedy_zone
except ImportError:
    from models import OrchestratorObs, OrchestratorAction, ZoneObs, ZoneAction, TaskResult
    from server.env import RPOEXEnv, _open_score
    from tasks.graders import TASKS, greedy_orchestrator, greedy_zone

# ---------------------------------------------------------------------------
# Orchestrator agent
# ---------------------------------------------------------------------------

def llm_orchestrator(obs: OrchestratorObs) -> OrchestratorAction:
    system_prompt = (
        "You are an orchestrator agent for a rotary parking system in HITEC City, Hyderabad.\n"
        "Each step you route ONE car to ONE zone.\n"
        "RULE 1: Look at zone_queue_lengths. Route to the zone with the highest queue.\n"
        "RULE 2: Never route to a zone with queue_length=0 unless ALL zones are empty.\n"
        "RULE 3: If all queues are 0, pick zone 2 (largest buffer).\n"
        "Zones: 0=Cyber Towers(4 wheels), 1=Inorbit(4), 2=Metro(5), 3=Mindspace(4), 4=Kondapur(3).\n"
        'Respond ONLY with valid JSON: {"zone_id": <int 0-4>} /no_think'
    )

    user_msg = (
        f"zone_occupancy: {[round(x, 2) for x in obs.zone_occupancy]}\n"
        f"zone_queue_lengths: {obs.zone_queue_lengths}\n"
        f"zone_avg_wait: {[round(x, 1) for x in obs.zone_avg_wait]}\n"
        f"arrival_rate_ema: {[round(x, 3) for x in obs.arrival_rate_ema]}\n"
        f"time_of_day: {obs.time_of_day:.3f}  step: {obs.step}\n"
        f"Which zone_id (0-4) should the next car be routed to?"
    )

    try:
        raw     = model_client.call_model([{"role": "system", "content": system_prompt},
                                           {"role": "user",   "content": user_msg}], max_tokens=20)
        zone_id = int(json.loads(raw)["zone_id"])
        if not (0 <= zone_id <= 4):
            raise ValueError(f"zone_id {zone_id} out of range")
        return OrchestratorAction(action="route_to_zone", zone_id=zone_id)
    except Exception as e:
        print(f"[WARN] Orchestrator LLM fallback at step {obs.step}: {e}", file=sys.stderr)
        return greedy_orchestrator(obs)


# ---------------------------------------------------------------------------
# Zone agent
# ---------------------------------------------------------------------------

def llm_zone_agent(obs: ZoneObs) -> ZoneAction:
    n = len(obs.wheel_occupancy)
    system_prompt = (
        "You are a zone agent for a rotary parking system.\n"
        "Assign the incoming car to the best wheel in your zone.\n"
        "RULE: Never pick a full wheel (occupancy=1.0 or queue_length=12). Pick the least-occupied wheel.\n"
        'Respond ONLY with: {"wheel_id": <int>} /no_think'
    )

    user_msg = (
        f"zone_id: {obs.zone_id}\n"
        f"wheel_occupancy: {[round(x, 2) for x in obs.wheel_occupancy]}\n"
        f"wheel_queue_lengths: {obs.wheel_queue_lengths}\n"
        f"est_rotation_cost: {obs.est_rotation_cost}\n"
        f"time_of_day: {obs.time_of_day:.3f}  step: {obs.step}\n"
        f"Which wheel_id (0-{n - 1}) should the car be assigned to?"
    )

    try:
        raw      = model_client.call_model([{"role": "system", "content": system_prompt},
                                            {"role": "user",   "content": user_msg}], max_tokens=20)
        wheel_id = int(json.loads(raw)["wheel_id"])
        if not (0 <= wheel_id < n):
            raise ValueError(f"wheel_id {wheel_id} out of range for zone {obs.zone_id}")
        return ZoneAction(action="assign_to_wheel", wheel_id=wheel_id)
    except Exception as e:
        print(f"[WARN] Zone LLM fallback at step {obs.step} zone {obs.zone_id}: {e}", file=sys.stderr)
        return greedy_zone(obs)


# ---------------------------------------------------------------------------
# Multi-agent step — orchestrator picks zone, zone agent picks wheel
# ---------------------------------------------------------------------------

def hybrid_agent(obs: OrchestratorObs, env: RPOEXEnv, use_llm: bool = True):
    orch_action = llm_orchestrator(obs) if use_llm else greedy_orchestrator(obs)
    zone_obs    = env.get_zone_obs(orch_action.zone_id)
    zone_action = llm_zone_agent(zone_obs) if use_llm else greedy_zone(zone_obs)
    return orch_action, orch_action.zone_id, zone_action


# ---------------------------------------------------------------------------
# Task runner — mandatory [START]/[STEP]/[END] stdout format (Gate 4)
# ---------------------------------------------------------------------------

def run_task_with_logging(task_id: str, use_llm: bool, seed: int = 42) -> TaskResult:
    task_meta       = TASKS[task_id]
    max_steps       = task_meta["steps"]
    lambda_override = 0.05 if task_id == "task1_easy" else None

    t_start = time.time()
    print(f"[START] task_id={task_id} model={MODEL_NAME}")

    env = RPOEXEnv(seed=seed, max_steps=max_steps, lambda_override=lambda_override)
    obs = env.reset(seed=seed)

    zone_occ_snapshots: list = []
    wait_snapshots: list     = []

    while not obs.done:
        orch_action, zone_id, zone_action = hybrid_agent(obs, env, use_llm=use_llm)
        obs = env.step(orch_action, zone_action)
        print(f"[STEP] step={obs.step} action=route_to_zone:{zone_id} reward={obs.reward:.4f} done={obs.done}")
        print(f"[ZONE] step={obs.step} zone={zone_id} wheel={zone_action.wheel_id} "
              f"zone_reward={env._zone_parked[zone_id]*0.10 - env._zone_overflowed[zone_id]*0.50:.3f}",
              file=sys.stderr)
        if env._step % 10 == 0:
            zone_occ_snapshots.append(list(obs.zone_occupancy))
            wait_snapshots.append(float(np.mean(obs.zone_avg_wait)))

    total_ops      = env._parked + env._retrieved
    throughput_rate = total_ops / max_steps if max_steps > 0 else 0.0
    avg_imbalance  = float(np.mean([np.std(s) for s in zone_occ_snapshots])) if zone_occ_snapshots else 0.0
    balance_score  = max(0.0, 1.0 - avg_imbalance / 0.5)
    avg_wait       = float(np.mean(wait_snapshots)) if wait_snapshots else 0.0
    wait_score     = max(0.0, 1.0 - min(avg_wait / 20.0, 1.0))

    elapsed = time.time() - t_start

    if task_id == "task1_easy":
        total_arrived = env._parked + env._overflowed
        service_rate  = env._parked / max(1, total_arrived)
        score = _open_score(service_rate)
        return TaskResult(
            task_id=task_id, score=score, passed=score >= 0.50,
            metrics={"service_rate": round(service_rate, 4),
                     "total_parked": float(env._parked), "total_retrieved": float(env._retrieved),
                     "total_overflowed": float(env._overflowed), "total_arrived": float(total_arrived),
                     "elapsed_s": round(elapsed, 1)},
            notes=f"Quiet demand λ=0.05. service_rate={service_rate:.4f} elapsed={elapsed:.1f}s",
        )
    elif task_id == "task2_medium":
        score = _open_score(0.60 * throughput_rate + 0.40 * balance_score)
        return TaskResult(
            task_id=task_id, score=score, passed=score >= 0.55,
            metrics={"throughput_rate": round(throughput_rate, 4), "balance_score": round(balance_score, 4),
                     "avg_imbalance": round(avg_imbalance, 4), "total_parked": float(env._parked),
                     "total_retrieved": float(env._retrieved), "total_overflowed": float(env._overflowed),
                     "elapsed_s": round(elapsed, 1)},
            notes=f"throughput={throughput_rate:.4f} balance={balance_score:.4f} elapsed={elapsed:.1f}s",
        )
    else:  # task3_hard
        score = _open_score(0.50 * throughput_rate + 0.30 * balance_score + 0.20 * wait_score)
        return TaskResult(
            task_id=task_id, score=score, passed=score >= 0.60,
            metrics={"throughput_rate": round(throughput_rate, 4), "balance_score": round(balance_score, 4),
                     "wait_score": round(wait_score, 4), "avg_wait": round(avg_wait, 4),
                     "avg_imbalance": round(avg_imbalance, 4), "total_parked": float(env._parked),
                     "total_retrieved": float(env._retrieved), "total_overflowed": float(env._overflowed),
                     "elapsed_s": round(elapsed, 1)},
            notes=f"throughput={throughput_rate:.4f} balance={balance_score:.4f} wait={wait_score:.4f} elapsed={elapsed:.1f}s",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    results      = {}
    total_score  = 0.0
    run_start    = time.time()

    for task_id in ["task1_easy", "task2_medium", "task3_hard"]:
        result = run_task_with_logging(task_id, use_llm=True, seed=42)
        results[task_id] = result
        total_score += result.score
        elapsed = result.metrics.get("elapsed_s", 0.0)
        print(f"[END] task_id={task_id} score={result.score:.4f} avg_score={total_score / len(results):.4f} elapsed={elapsed:.1f}s")

    total_elapsed = time.time() - run_start
    avg_score = total_score / 3.0
    output = {
        "task1_easy":   round(results["task1_easy"].score, 4),
        "task2_medium": round(results["task2_medium"].score, 4),
        "task3_hard":   round(results["task3_hard"].score, 4),
        "avg_score":    round(avg_score, 4),
        "model":        MODEL_NAME,
        "total_elapsed_s": round(total_elapsed, 1),
    }
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBaseline scores written to baseline_scores.json")
    print(f"avg_score={avg_score:.4f}  total_elapsed={total_elapsed:.1f}s")
    for tid, r in results.items():
        print(f"  {tid}: score={r.score:.4f} passed={r.passed} elapsed={r.metrics.get('elapsed_s', 0):.1f}s")


if __name__ == "__main__":
    main()
