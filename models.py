from __future__ import annotations

from typing import Dict, List, Literal, Optional

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from pydantic import BaseModel as Action
    from pydantic import BaseModel as Observation
    from pydantic import BaseModel as State

from pydantic import BaseModel, Field


class OrchestratorAction(Action):
    action: Literal["route_to_zone"] = Field(description="Action type: always 'route_to_zone'")
    zone_id: int = Field(ge=0, le=4, description="Target zone index (0–4)")


class OrchestratorObs(Observation):
    zone_occupancy: List[float] = Field(description="Occupancy ratio 0.0–1.0 per zone (len=5)")
    zone_queue_lengths: List[int] = Field(description="Arrival + retrieval queue depth per zone (len=5)")
    zone_avg_wait: List[float] = Field(description="Mean wait time in steps per zone (len=5)")
    arrival_rate_ema: List[float] = Field(description="Exponential moving average of recent arrivals per zone (len=5)")
    recent_delta_queue: List[float] = Field(description="Queue trend per zone; positive = growing (len=5)")
    time_of_day: float = Field(description="Normalized time of day 0.0–1.0 (7AM=0.0, 11PM=1.0)")
    step: int = Field(description="Current simulation step")
    done: bool = Field(description="Whether the episode has ended")
    reward: float = Field(description="Reward received at this step")


class ZoneAction(Action):
    action: Literal["assign_to_wheel"] = Field(description="Action type: always 'assign_to_wheel'")
    wheel_id: int = Field(ge=0, description="Target wheel index within the zone (>= 0; upper bound enforced by env)")


class ZoneObs(Observation):
    zone_id: int = Field(description="Zone index this observation belongs to (0–4)")
    wheel_occupancy: List[float] = Field(description="Occupancy ratio 0.0–1.0 per wheel in this zone")
    wheel_queue_lengths: List[int] = Field(description="Pending request queue depth per wheel")
    est_rotation_cost: List[float] = Field(description="Estimated steps to service next request per wheel")
    local_arrival_rate_ema: float = Field(description="Zone-level exponential moving average of recent arrivals")
    time_of_day: float = Field(description="Normalized time of day 0.0–1.0 (7AM=0.0, 11PM=1.0)")
    step: int = Field(description="Current simulation step")
    done: bool = Field(description="Whether the episode has ended")
    reward: float = Field(description="Reward received at this step")


class CarState(BaseModel):
    car_id: str = Field(description="Unique car identifier")
    arrival_step: int = Field(description="Simulation step when the car arrived")
    zone_id: int = Field(description="Zone the car was routed to")
    wheel_id: Optional[int] = Field(default=None, description="Wheel the car is assigned to; None if not yet parked")
    slot_id: Optional[int] = Field(default=None, description="Slot the car occupies; None if not yet parked")
    status: Literal["queued", "parked", "retrieving"] = Field(description="Current lifecycle state of the car")


class WheelState(BaseModel):
    zone_id: int = Field(description="Zone this wheel belongs to")
    wheel_id: int = Field(description="Global wheel index (0–19)")
    local_wheel_id: int = Field(description="Wheel index within its zone (0 to zone.wheels-1)")
    slots: List[Optional[str]] = Field(description="car_id per slot or None if empty; len=12")
    front_slot: int = Field(description="Current front slot index (0–11)")
    queue_length: int = Field(description="Number of pending park/retrieve requests for this wheel")
    est_rotation_cost: float = Field(description="Average steps to reach the next target slot")


class ZoneState(BaseModel):
    zone_id: int = Field(description="Zone index (0–4)")
    name: str = Field(description="Human-readable zone name (real HITEC City location)")
    wheels: List[WheelState] = Field(description="All wheels in this zone")
    arrival_queue: List[CarState] = Field(description="Cars waiting to be parked in this zone")
    retrieval_queue: List[CarState] = Field(description="Cars waiting to be retrieved in this zone")
    occupancy: float = Field(description="Fraction of total slots filled (0.0–1.0)")
    avg_wait: float = Field(description="Mean wait time in steps across all queued cars in this zone")


class RPOEXState(State):
    zones: List[ZoneState] = Field(description="State of all 5 zones")
    step: int = Field(description="Current simulation step")
    hour: float = Field(description="Current hour offset from 7AM (0.0–16.0)")
    total_parked: int = Field(description="Cumulative cars successfully parked this episode")
    total_retrieved: int = Field(description="Cumulative cars successfully retrieved this episode")
    total_overflowed: int = Field(description="Cumulative cars that timed out and overflowed this episode")
    episode_id: str = Field(description="Unique identifier for this episode run")
    seed: int = Field(description="Random seed used for this episode (seed=42 for baselines)")


class TaskResult(BaseModel):
    task_id: str = Field(description="Task identifier (task1_easy / task2_medium / task3_hard)")
    score: float = Field(description="Final task score clamped to open interval (0.001, 0.999)")
    metrics: Dict[str, float] = Field(description="Raw metric breakdown (throughput, wait_score, zone_balance, etc.)")
    passed: bool = Field(description="True if score >= pass_threshold for this task")
    notes: str = Field(description="Human-readable summary of the run")


if __name__ == "__main__":
    a = OrchestratorAction(action="route_to_zone", zone_id=2)
    o = OrchestratorObs(
        zone_occupancy=[0.0] * 5,
        zone_queue_lengths=[0] * 5,
        zone_avg_wait=[0.0] * 5,
        arrival_rate_ema=[0.0] * 5,
        recent_delta_queue=[0.0] * 5,
        time_of_day=0.5,
        step=0,
        done=False,
        reward=0.0,
    )
    print("OrchestratorAction:", a)
    print("OrchestratorObs:", o)
    print("ITEM 1.1 DONE")

    za = ZoneAction(action="assign_to_wheel", wheel_id=1)
    zo = ZoneObs(
        zone_id=0,
        wheel_occupancy=[0.0] * 4,
        wheel_queue_lengths=[0] * 4,
        est_rotation_cost=[0.0] * 4,
        local_arrival_rate_ema=0.0,
        time_of_day=0.5,
        step=0,
        done=False,
        reward=0.0,
    )
    print("ZoneAction:", za)
    print("ZoneObs:", zo)
    print("ITEM 1.2 DONE")

    ws = WheelState(
        zone_id=0, wheel_id=0, local_wheel_id=0,
        slots=[None] * 12, front_slot=0, queue_length=0, est_rotation_cost=0.0
    )
    cs = CarState(car_id="car_001", arrival_step=5, zone_id=0, status="queued")
    zs = ZoneState(
        zone_id=0, name="Cyber Towers Junction",
        wheels=[ws], arrival_queue=[cs], retrieval_queue=[],
        occupancy=0.0, avg_wait=0.0
    )
    tr = TaskResult(
        task_id="task1_easy", score=0.5,
        metrics={"throughput": 0.5}, passed=True, notes="smoke test"
    )
    print("WheelState:", ws)
    print("CarState:", cs)
    print("ZoneState:", zs)
    print("TaskResult:", tr)
    print("ITEM 1.3 DONE")
