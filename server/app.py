from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import HTTPException

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Run: uv sync"
    ) from e

from models import ParkingAction, OrchestratorAction, OrchestratorObs, ZoneAction, TaskResult
from server.env import RPOEXEnv, ZONES
from tasks.graders import TASKS, greedy_orchestrator, greedy_zone


# ---------------------------------------------------------------------------
# Wrapper env: exposes ParkingAction(zone_id, wheel_id) to the Gradio UI
# ---------------------------------------------------------------------------

class RPOEXEnvUI(RPOEXEnv):
    def step(self, action: ParkingAction) -> OrchestratorObs:  # type: ignore[override]
        orch_action  = OrchestratorAction(action="route_to_zone", zone_id=action.zone_id)
        zone_action  = ZoneAction(action="assign_to_wheel", wheel_id=action.wheel_id)
        return super().step(orch_action, zone_action)


# ---------------------------------------------------------------------------
# Create app via openenv-core — enables HF Spaces web UI
# ---------------------------------------------------------------------------

app = create_app(
    RPOEXEnvUI,
    ParkingAction,
    OrchestratorObs,
    env_name="rpoe-x",
    max_concurrent_envs=1,
)


# ---------------------------------------------------------------------------
# Additional endpoints on top of the openenv base app
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "rpoe-x", "version": "0.1.0"}


@app.get("/tasks")
def list_tasks():
    return [
        {
            "id": tid,
            "steps": meta["steps"],
            "threshold": meta["threshold"],
        }
        for tid, meta in TASKS.items()
    ]


@app.get("/task/{task_id}")
def run_task(task_id: str, seed: int = 42):
    if task_id not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}"
        )

    def greedy_agent(obs, env):
        orch = greedy_orchestrator(obs)
        zone_obs = env.get_zone_obs(orch.zone_id)
        zone_act = greedy_zone(zone_obs)
        return orch, orch.zone_id, zone_act

    result: TaskResult = TASKS[task_id]["fn"](greedy_agent, seed=seed)
    return result.model_dump()


@app.get("/info")
def info():
    return {
        "name": "RPOE-X",
        "zones": ZONES,
        "total_wheels": sum(z["wheels"] for z in ZONES),
        "total_slots": sum(z["wheels"] * 12 for z in ZONES),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 7860):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
