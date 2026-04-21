from __future__ import annotations

import os
import json
import time

from openai import OpenAI

from models import OrchestratorObs, OrchestratorAction, ZoneObs, ZoneAction, TaskResult
from server.env import RPOEXEnv
from tasks.graders import (
    TASKS, greedy_orchestrator, greedy_zone,
    run_task1, run_task2, run_task3,
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
    system_prompt = """You are an orchestrator agent for a multi-zone rotary parking system in HITEC City, Hyderabad.
Your job: choose which zone (0–4) to route the next incoming parking request to.
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
    except Exception:
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
    """
    task_meta = TASKS[task_id]
    max_steps = task_meta["steps"]

    print(f"[START] task_id={task_id} model={MODEL_NAME}")

    env = RPOEXEnv(seed=seed, max_steps=max_steps)
    obs = env.reset(seed=seed)

    step = 0
    while not obs.done:
        orch_action, zone_id, zone_action = hybrid_agent(obs, env, use_llm=use_llm)
        obs = env.step(orch_action)
        print(f"[STEP] step={obs.step} action=route_to_zone:{zone_id} reward={obs.reward:.4f} done={obs.done}")
        step += 1

    def agent_fn(o, e):
        return hybrid_agent(o, e, use_llm=False)

    result: TaskResult = task_meta["fn"](agent_fn, seed=seed)
    return result

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
