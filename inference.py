from __future__ import annotations

import os
import json
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from models import OrchestratorObs, OrchestratorAction, TaskResult
from server.env import RPOEXEnv, _open_score
from tasks.graders import (
    TASKS, greedy_orchestrator, greedy_zone,
)

# ---------------------------------------------------------------------------
# PART A — Environment variables (Gate 2 — exact format required)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")    # NO default — must be set externally

client = OpenAI(
    api_key=HF_TOKEN or "sk-placeholder",
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# PART B — LLM orchestrator agent
# ---------------------------------------------------------------------------

def llm_orchestrator(obs: OrchestratorObs) -> OrchestratorAction:
    """
    Uses LLM to decide which zone to route the next incoming car to.
    Falls back to greedy if LLM call fails or returns invalid zone_id.
    """
    system_prompt = """You are an orchestrator agent for a rotary parking system in HITEC City, Hyderabad.
Each step you park ONE car in ONE zone. If you route to a zone with no waiting cars, the step is wasted.
CRITICAL RULE: Always route to a zone that has cars waiting (zone_queue_lengths > 0).
If multiple zones have queued cars, prefer the zone with the longest queue to minimize overflow timeouts.
If no zone has queued cars, route to zone 2 (largest buffer).
Zones: 0=Cyber Towers, 1=Inorbit Mall, 2=Hitech City Metro (largest), 3=Mindspace, 4=Kondapur.
Respond with ONLY a JSON object: {"zone_id": <int 0-4>}
No explanation. No markdown. Just the JSON."""

    user_msg = f"""Current state:
zone_occupancy (0=empty, 1=full): {obs.zone_occupancy}
zone_queue_lengths (arrival+retrieval): {obs.zone_queue_lengths}
zone_avg_wait (steps): {obs.zone_avg_wait}
arrival_rate_ema (recent trend): {obs.arrival_rate_ema}
recent_delta_queue (positive=growing): {obs.recent_delta_queue}
time_of_day (0=7AM, 1=11PM): {obs.time_of_day:.3f}
step: {obs.step}

Which zone_id (0–4) should the next car be routed to?"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        zone_id = int(parsed["zone_id"])
        if zone_id < 0 or zone_id > 4:
            raise ValueError(f"zone_id {zone_id} out of range")
        return OrchestratorAction(action="route_to_zone", zone_id=zone_id)
    except Exception as e:
        print(f"[WARN] LLM fallback at step {obs.step}: {e}", file=sys.stderr)
        return greedy_orchestrator(obs)

# ---------------------------------------------------------------------------
# PART C — Hybrid agent (LLM orchestrator + greedy zone)
# ---------------------------------------------------------------------------

def hybrid_agent(obs: OrchestratorObs, env: RPOEXEnv, use_llm: bool = True):
    """
    Orchestrator: LLM (with greedy fallback)
    Zone:         Always greedy (deterministic, no LLM needed)
    Returns: (orch_action, zone_id, zone_action)
    """
    if use_llm:
        orch_action = llm_orchestrator(obs)
    else:
        orch_action = greedy_orchestrator(obs)
    zone_id = orch_action.zone_id
    zone_obs = env.get_zone_obs(zone_id)
    zone_action = greedy_zone(zone_obs)
    return orch_action, zone_id, zone_action

# ---------------------------------------------------------------------------
# PART D — Per-task runner with mandatory stdout logging (Gate 4)
# ---------------------------------------------------------------------------

def run_task_with_logging(
    task_id: str,
    use_llm: bool,
    seed: int = 42,
) -> TaskResult:
    """
    Runs a full task episode with [START]/[STEP]/[END] stdout logging.
    Stdout format is mandatory per competition spec — do not change.
    Single episode run — logging and scoring happen together.
    """
    task_meta = TASKS[task_id]
    max_steps = task_meta["steps"]
    lambda_override = 0.05 if task_id == "task1_easy" else None

    print(f"[START] task_id={task_id} model={MODEL_NAME}")

    env = RPOEXEnv(seed=seed, max_steps=max_steps, lambda_override=lambda_override)
    obs = env.reset(seed=seed)

    zone_occ_snapshots: list = []
    wait_snapshots: list = []

    while not obs.done:
        orch_action, zone_id, zone_action = hybrid_agent(obs, env, use_llm=use_llm)
        obs = env.step(orch_action, zone_action)
        print(f"[STEP] step={obs.step} action=route_to_zone:{zone_id} reward={obs.reward:.4f} done={obs.done}")
        if env._step % 10 == 0:
            zone_occ_snapshots.append(list(obs.zone_occupancy))
            wait_snapshots.append(float(np.mean(obs.zone_avg_wait)))

    total_ops = env._parked + env._retrieved
    throughput_rate = total_ops / max_steps if max_steps > 0 else 0.0

    if zone_occ_snapshots:
        imbalances = [float(np.std(snap)) for snap in zone_occ_snapshots]
        avg_imbalance = float(np.mean(imbalances))
    else:
        avg_imbalance = 0.0
    balance_score = max(0.0, 1.0 - avg_imbalance / 0.5)

    avg_wait = float(np.mean(wait_snapshots)) if wait_snapshots else 0.0
    wait_score = max(0.0, 1.0 - min(avg_wait / 20.0, 1.0))

    if task_id == "task1_easy":
        total_arrived = env._parked + env._overflowed
        service_rate = env._parked / max(1, total_arrived)
        score = _open_score(service_rate)
        return TaskResult(
            task_id=task_id, score=score, passed=score >= 0.50,
            metrics={
                "service_rate": round(service_rate, 4),
                "total_parked": float(env._parked),
                "total_retrieved": float(env._retrieved),
                "total_overflowed": float(env._overflowed),
                "total_arrived": float(total_arrived),
            },
            notes=f"Quiet demand λ=0.05. service_rate={service_rate:.4f}",
        )
    elif task_id == "task2_medium":
        raw = 0.60 * throughput_rate + 0.40 * balance_score
        score = _open_score(raw)
        return TaskResult(
            task_id=task_id, score=score, passed=score >= 0.55,
            metrics={
                "throughput_rate": round(throughput_rate, 4),
                "balance_score": round(balance_score, 4),
                "avg_imbalance": round(avg_imbalance, 4),
                "total_parked": float(env._parked),
                "total_retrieved": float(env._retrieved),
                "total_overflowed": float(env._overflowed),
            },
            notes=f"throughput={throughput_rate:.4f} balance={balance_score:.4f}",
        )
    else:  # task3_hard
        raw = 0.50 * throughput_rate + 0.30 * balance_score + 0.20 * wait_score
        score = _open_score(raw)
        return TaskResult(
            task_id=task_id, score=score, passed=score >= 0.60,
            metrics={
                "throughput_rate": round(throughput_rate, 4),
                "balance_score": round(balance_score, 4),
                "wait_score": round(wait_score, 4),
                "avg_wait": round(avg_wait, 4),
                "avg_imbalance": round(avg_imbalance, 4),
                "total_parked": float(env._parked),
                "total_retrieved": float(env._retrieved),
                "total_overflowed": float(env._overflowed),
            },
            notes=f"throughput={throughput_rate:.4f} balance={balance_score:.4f} wait={wait_score:.4f}",
        )

# ---------------------------------------------------------------------------
# PART E — Main entry point
# ---------------------------------------------------------------------------

def main():
    use_llm = HF_TOKEN is not None and HF_TOKEN.strip() != ""

    results = {}
    total_score = 0.0

    for task_id in ["task1_easy", "task2_medium", "task3_hard"]:
        result = run_task_with_logging(task_id, use_llm=use_llm, seed=42)
        results[task_id] = result
        total_score += result.score
        avg_so_far = total_score / len(results)
        print(f"[END] task_id={task_id} score={result.score:.4f} avg_score={avg_so_far:.4f}")

    avg_score = total_score / 3.0

    output = {
        "task1_easy":   round(results["task1_easy"].score, 4),
        "task2_medium": round(results["task2_medium"].score, 4),
        "task3_hard":   round(results["task3_hard"].score, 4),
        "avg_score":    round(avg_score, 4),
        "model":        MODEL_NAME,
    }
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBaseline scores written to baseline_scores.json")
    print(f"avg_score={avg_score:.4f}")
    for tid, r in results.items():
        print(f"  {tid}: score={r.score:.4f} passed={r.passed}")


if __name__ == "__main__":
    main()
