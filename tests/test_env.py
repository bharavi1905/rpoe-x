import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.env import RPOEXEnv, _rotation_cost, _open_score, ZONES
from models import OrchestratorAction, OrchestratorObs, ZoneObs, TaskResult
from tasks.graders import (
    run_task1, run_task2, run_task3,
    greedy_orchestrator, greedy_zone,
)


# ── Test 1 ──────────────────────────────────────────────────────────────────
def test_reset_returns_obs():
    env = RPOEXEnv(seed=42)
    obs = env.reset()
    assert isinstance(obs, OrchestratorObs)
    assert len(obs.zone_occupancy) == 5
    assert len(obs.zone_queue_lengths) == 5
    assert obs.step == 0
    assert obs.done == False
    assert obs.reward == 0.0


# ── Test 2 ──────────────────────────────────────────────────────────────────
def test_step_returns_obs():
    env = RPOEXEnv(seed=42)
    env.reset()
    obs = env.step(OrchestratorAction(action="route_to_zone", zone_id=0))
    assert isinstance(obs, OrchestratorObs)
    assert obs.step == 1


# ── Test 3 ──────────────────────────────────────────────────────────────────
def test_zone_step_returns_obs():
    env = RPOEXEnv(seed=42)
    env.reset()
    zone_obs = env.get_zone_obs(0)
    assert isinstance(zone_obs, ZoneObs)
    assert zone_obs.zone_id == 0
    assert len(zone_obs.wheel_occupancy) == ZONES[0]["wheels"]


# ── Test 4 ──────────────────────────────────────────────────────────────────
def test_rotation_cost_shortest_path():
    assert _rotation_cost(0, 3) == 3,  f"got {_rotation_cost(0, 3)}"
    assert _rotation_cost(0, 9) == 3,  f"got {_rotation_cost(0, 9)}"
    assert _rotation_cost(0, 6) == 6,  f"got {_rotation_cost(0, 6)}"
    assert _rotation_cost(0, 0) == 0,  f"got {_rotation_cost(0, 0)}"
    assert _rotation_cost(3, 3) == 0,  f"got {_rotation_cost(3, 3)}"


# ── Test 5 ──────────────────────────────────────────────────────────────────
def test_overflow_timeout():
    env = RPOEXEnv(seed=42, max_steps=200)
    env.reset()
    for _ in range(20):
        env.step(OrchestratorAction(action="route_to_zone", zone_id=2))
    assert env._overflowed >= 0


# ── Test 6 ──────────────────────────────────────────────────────────────────
def test_score_open_interval():
    assert _open_score(0.0)  == 0.001
    assert _open_score(1.0)  == 0.999
    assert _open_score(0.5)  == 0.5
    assert _open_score(-1.0) == 0.001
    assert _open_score(2.0)  == 0.999


# ── Test 7 ──────────────────────────────────────────────────────────────────
def test_greedy_baseline_runs():
    def greedy_agent(obs, env):
        orch = greedy_orchestrator(obs)
        zone_obs = env.get_zone_obs(orch.zone_id)
        zone_act = greedy_zone(zone_obs)
        return orch, orch.zone_id, zone_act

    result = run_task1(greedy_agent, seed=42)
    assert isinstance(result, TaskResult)
    assert result.task_id == "task1_easy"
    assert 0.001 <= result.score <= 0.999


# ── Test 8 ──────────────────────────────────────────────────────────────────
def test_all_three_graders_run():
    def greedy_agent(obs, env):
        orch = greedy_orchestrator(obs)
        zone_obs = env.get_zone_obs(orch.zone_id)
        zone_act = greedy_zone(zone_obs)
        return orch, orch.zone_id, zone_act

    r1 = run_task1(greedy_agent, seed=42)
    r2 = run_task2(greedy_agent, seed=42)
    r3 = run_task3(greedy_agent, seed=42)

    for r, tid in [
        (r1, "task1_easy"),
        (r2, "task2_medium"),
        (r3, "task3_hard"),
    ]:
        assert r.task_id == tid
        assert 0.001 <= r.score <= 0.999, f"{tid} score out of range: {r.score}"
        assert isinstance(r.passed, bool)


# ── Test 9 ──────────────────────────────────────────────────────────────────
def test_determinism():
    def greedy_agent(obs, env):
        orch = greedy_orchestrator(obs)
        zone_obs = env.get_zone_obs(orch.zone_id)
        zone_act = greedy_zone(zone_obs)
        return orch, orch.zone_id, zone_act

    r1a = run_task1(greedy_agent, seed=42)
    r1b = run_task1(greedy_agent, seed=42)
    assert r1a.score == r1b.score, \
        f"Determinism broken: {r1a.score} != {r1b.score}"

    r2a = run_task2(greedy_agent, seed=42)
    r2b = run_task2(greedy_agent, seed=42)
    assert r2a.score == r2b.score, \
        f"Determinism broken: {r2a.score} != {r2b.score}"
