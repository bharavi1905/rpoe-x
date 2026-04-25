from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional

import numpy as np

try:
    from ..models import (
        OrchestratorObs, ZoneObs, ZoneAction, OrchestratorAction,
        CarState, WheelState, ZoneState, RPOEXState,
    )
except ImportError:
    from models import (
        OrchestratorObs, ZoneObs, ZoneAction, OrchestratorAction,
        CarState, WheelState, ZoneState, RPOEXState,
    )

try:
    from openenv.core.env_server.interfaces import Environment as _BaseEnv
except ImportError:
    _BaseEnv = object  # type: ignore

# ---------------------------------------------------------------------------
# PART A — Constants
# ---------------------------------------------------------------------------

ZONES = [
    {"id": 0, "name": "Cyber Towers Junction", "wheels": 4, "multiplier": 1.5},
    {"id": 1, "name": "Inorbit Mall Signal",   "wheels": 4, "multiplier": 1.2},
    {"id": 2, "name": "Hitech City Metro",     "wheels": 5, "multiplier": 1.0},
    {"id": 3, "name": "Mindspace Junction",    "wheels": 4, "multiplier": 1.2},
    {"id": 4, "name": "Kondapur / Whitefields","wheels": 3, "multiplier": 0.9},
]
WHEEL_SIZE = 12
MAX_QUEUE_PER_ZONE = 10
MAX_STEPS = 960
OVERFLOW_TIMEOUT = 10
DWELL_MU_STEPS  = 480
DWELL_STD_STEPS = 120
EMA_ALPHA = 0.15

ARRIVAL_RATES = [
    (0.0,  1.0,  0.05),
    (1.0,  3.0,  0.40),
    (3.0,  5.5,  0.12),
    (5.5,  7.5,  0.22),
    (7.5,  10.0, 0.10),
    (10.0, 13.0, 0.38),
    (13.0, 14.0, 0.08),
    (14.0, 16.0, 0.02),
]

# ---------------------------------------------------------------------------
# PART B — Helper functions
# ---------------------------------------------------------------------------

def _arrival_rate(hour_offset: float) -> float:
    hour_offset = min(hour_offset, 16.0)
    for start, end, lam in ARRIVAL_RATES:
        if start <= hour_offset < end:
            return lam
    return 0.0


def _sample_dwell(rng: np.random.Generator) -> int:
    mean = DWELL_MU_STEPS
    std  = DWELL_STD_STEPS
    mu_ln    = math.log(mean ** 2 / math.sqrt(mean ** 2 + std ** 2))
    sigma_ln = math.sqrt(math.log(1 + (std / mean) ** 2))
    sample = rng.lognormal(mean=mu_ln, sigma=sigma_ln)
    return max(60, int(sample))


def _current_hour_offset(step: int) -> float:
    return min(step / 60.0, 16.0)


def _rotation_cost(current_front: int, target_slot: int, wheel_size: int = 12) -> int:
    cw_dist  = (wheel_size - target_slot + current_front) % wheel_size
    ccw_dist = (target_slot - current_front) % wheel_size
    return min(cw_dist, ccw_dist)


def _open_score(score: float) -> float:
    return max(0.001, min(0.999, score))


# ---------------------------------------------------------------------------
# PART D — RPOEXEnv
# ---------------------------------------------------------------------------

class RPOEXEnv(_BaseEnv):

    def __init__(self, seed: int = 42, max_steps: int = MAX_STEPS, lambda_override: float = None):
        self._seed = seed
        self._max_steps = max_steps
        self._lambda_override = lambda_override
        self._rng = np.random.default_rng(seed)
        self._slots: List[List[List[Optional[str]]]] = []
        self._arrival_q: List[List[CarState]] = []
        self._retrieval_q: List[List[CarState]] = []
        self._dwell_timers: Dict[str, int] = {}
        self._front_slot: List[List[int]] = []
        self._ema: List[float] = [0.0] * 5
        self._prev_queue: List[int] = [0] * 5
        self._step: int = 0
        self._parked: int = 0
        self._retrieved: int = 0
        self._overflowed: int = 0
        self._zone_parked: List[int] = [0] * 5        # per-step (for [ZONE] log)
        self._zone_overflowed: List[int] = [0] * 5   # per-step (for [ZONE] log)
        self._zone_parked_total: List[int] = [0] * 5
        self._zone_overflowed_total: List[int] = [0] * 5
        self._episode_id: str = ""
        self._car_counter: int = 0
        if _BaseEnv is not object:
            super().__init__()
        self.reset()

    def reset(self, seed: int = None, episode_id: str = None) -> OrchestratorObs:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        self._slots = [
            [[None] * WHEEL_SIZE for _ in range(ZONES[z]["wheels"])]
            for z in range(5)
        ]
        self._front_slot = [
            [0] * ZONES[z]["wheels"]
            for z in range(5)
        ]
        self._arrival_q   = [[] for _ in range(5)]
        self._retrieval_q = [[] for _ in range(5)]
        self._dwell_timers = {}
        self._step        = 0
        self._parked      = 0
        self._retrieved   = 0
        self._overflowed  = 0
        self._car_counter = 0
        self._ema             = [0.0] * 5
        self._prev_queue      = [0] * 5
        self._zone_parked          = [0] * 5
        self._zone_overflowed      = [0] * 5
        self._zone_parked_total    = [0] * 5
        self._zone_overflowed_total= [0] * 5
        self._episode_id           = episode_id or str(uuid.uuid4())[:8]

        return self._make_orchestrator_obs(reward=0.0, done=False)

    def _make_orchestrator_obs(self, reward: float, done: bool) -> OrchestratorObs:
        occupancy = []
        queue_lengths = []
        avg_waits = []
        for z in range(5):
            total_slots = ZONES[z]["wheels"] * WHEEL_SIZE
            filled = sum(
                1 for w in range(ZONES[z]["wheels"])
                for s in range(WHEEL_SIZE)
                if self._slots[z][w][s] is not None
            )
            occupancy.append(round(filled / total_slots, 4))
            ql = len(self._arrival_q[z]) + len(self._retrieval_q[z])
            queue_lengths.append(ql)
            all_queued = self._arrival_q[z] + self._retrieval_q[z]
            if all_queued:
                avg_waits.append(
                    round(sum(self._step - c.arrival_step for c in all_queued) / len(all_queued), 4)
                )
            else:
                avg_waits.append(0.0)

        delta = [float(queue_lengths[z] - self._prev_queue[z]) for z in range(5)]

        return OrchestratorObs(
            zone_occupancy=occupancy,
            zone_queue_lengths=queue_lengths,
            zone_avg_wait=avg_waits,
            arrival_rate_ema=list(self._ema),
            recent_delta_queue=delta,
            time_of_day=round(_current_hour_offset(self._step) / 16.0, 4),
            step=self._step,
            done=done,
            reward=reward,
        )

    def _next_car_id(self) -> str:
        self._car_counter += 1
        return f"car_{self._episode_id}_{self._car_counter:04d}"

    def _process_arrivals(self) -> int:
        """Stochastic arrivals for current step. Returns total new arrivals."""
        hour = _current_hour_offset(self._step)
        base_rate = self._lambda_override if self._lambda_override is not None else _arrival_rate(hour)
        total_new = 0
        for z in range(5):
            lam = base_rate * ZONES[z]["multiplier"]
            n_arrivals = self._rng.poisson(lam)
            for _ in range(n_arrivals):
                if len(self._arrival_q[z]) >= MAX_QUEUE_PER_ZONE:
                    self._overflowed += 1
                    continue
                car = CarState(
                    car_id=self._next_car_id(),
                    arrival_step=self._step,
                    zone_id=z,
                    status="queued",
                )
                self._arrival_q[z].append(car)
                total_new += 1
            self._ema[z] = EMA_ALPHA * n_arrivals + (1 - EMA_ALPHA) * self._ema[z]
        return total_new

    def _process_overflow_timeout(self):
        """Remove cars that have waited too long in arrival queue (overflow timeout)."""
        for z in range(5):
            still_waiting = []
            for car in self._arrival_q[z]:
                waited = self._step - car.arrival_step
                if waited >= OVERFLOW_TIMEOUT:
                    self._overflowed += 1
                else:
                    still_waiting.append(car)
            self._arrival_q[z] = still_waiting

    def _trigger_retrievals(self):
        """Move parked cars whose dwell time has elapsed into retrieval queue."""
        expired = [car_id for car_id, due_step in self._dwell_timers.items()
                   if self._step >= due_step]
        for car_id in expired:
            del self._dwell_timers[car_id]
            for z in range(5):
                for w in range(ZONES[z]["wheels"]):
                    for s in range(WHEEL_SIZE):
                        if self._slots[z][w][s] == car_id:
                            car = CarState(
                                car_id=car_id,
                                arrival_step=self._step,
                                zone_id=z,
                                wheel_id=w,
                                slot_id=s,
                                status="retrieving",
                            )
                            self._retrieval_q[z].append(car)
                            return

    @property
    def state(self) -> RPOEXState:
        zones = []
        global_wheel_offset = 0
        for z in range(5):
            wheels = []
            all_q = self._arrival_q[z] + self._retrieval_q[z]
            avg_w = (
                sum(self._step - c.arrival_step for c in all_q) / len(all_q)
                if all_q else 0.0
            )
            total_slots = ZONES[z]["wheels"] * WHEEL_SIZE
            filled = sum(
                1 for w in range(ZONES[z]["wheels"])
                for s in range(WHEEL_SIZE)
                if self._slots[z][w][s] is not None
            )
            for w in range(ZONES[z]["wheels"]):
                wheels.append(WheelState(
                    zone_id=z,
                    wheel_id=global_wheel_offset + w,
                    local_wheel_id=w,
                    slots=list(self._slots[z][w]),
                    front_slot=self._front_slot[z][w],
                    queue_length=len(self._arrival_q[z]),
                    est_rotation_cost=float(_rotation_cost(self._front_slot[z][w], 0)),
                ))
            global_wheel_offset += ZONES[z]["wheels"]
            zones.append(ZoneState(
                zone_id=z,
                name=ZONES[z]["name"],
                wheels=wheels,
                arrival_queue=list(self._arrival_q[z]),
                retrieval_queue=list(self._retrieval_q[z]),
                occupancy=round(filled / total_slots, 4),
                avg_wait=round(avg_w, 4),
            ))
        return RPOEXState(
            zones=zones,
            step=self._step,
            hour=_current_hour_offset(self._step),
            total_parked=self._parked,
            total_retrieved=self._retrieved,
            total_overflowed=self._overflowed,
            episode_id=self._episode_id,
            seed=self._seed,
        )

    def step(self, action: OrchestratorAction, zone_action: Optional[ZoneAction] = None, timeout_s: Optional[float] = None, **kwargs) -> OrchestratorObs:
        """
        One environment step. Order of operations:
        1. Stochastic arrivals
        2. Overflow timeout check
        3. Dwell timer → retrieval queue
        4. Orchestrator routes zone; zone agent picks wheel within that zone
        5. Process one retrieval per zone (FIFO from retrieval queue)
        6. Compute reward
        7. Advance step counter
        8. Return OrchestratorObs
        """
        # Reset per-step zone counters
        self._zone_parked     = [0] * 5
        self._zone_overflowed = [0] * 5

        # 1. Arrivals
        self._process_arrivals()

        # 2. Overflow timeout
        self._process_overflow_timeout()

        # 3. Dwell timers → retrieval queue
        self._trigger_retrievals()

        # 4. Route one car from arrival queue of the chosen zone to a wheel
        z = action.zone_id
        throughput_this_step = 0
        if self._arrival_q[z]:
            car = self._arrival_q[z].pop(0)
            # Use zone agent's wheel choice if provided and valid; else fall back to least-occupied
            if zone_action is not None and 0 <= zone_action.wheel_id < ZONES[z]["wheels"]:
                chosen_wheel = zone_action.wheel_id
            else:
                chosen_wheel = min(
                    range(ZONES[z]["wheels"]),
                    key=lambda w: sum(1 for s in self._slots[z][w] if s is not None)
                )
            parked = False
            for s in range(WHEEL_SIZE):
                if self._slots[z][chosen_wheel][s] is None:
                    self._slots[z][chosen_wheel][s] = car.car_id
                    self._parked += 1
                    self._zone_parked[z] += 1
                    self._zone_parked_total[z] += 1
                    throughput_this_step += 1
                    dwell = _sample_dwell(self._rng)
                    self._dwell_timers[car.car_id] = self._step + dwell
                    parked = True
                    break
            if not parked:
                # Chosen wheel is full — overflow this car
                self._overflowed += 1
                self._zone_overflowed[z] += 1
                self._zone_overflowed_total[z] += 1

        # 5. Process one retrieval per zone (FIFO)
        for z in range(5):
            if self._retrieval_q[z]:
                car = self._retrieval_q[z].pop(0)
                if car.wheel_id is not None and car.slot_id is not None:
                    self._slots[z][car.wheel_id][car.slot_id] = None
                else:
                    for w in range(ZONES[z]["wheels"]):
                        for s in range(WHEEL_SIZE):
                            if self._slots[z][w][s] == car.car_id:
                                self._slots[z][w][s] = None
                                break
                self._retrieved += 1
                throughput_this_step += 1

        # 6. Compute reward
        all_queued = [c for z in range(5) for c in self._arrival_q[z] + self._retrieval_q[z]]
        avg_wait = (
            sum(self._step - c.arrival_step for c in all_queued) / len(all_queued)
            if all_queued else 0.0
        )
        zone_occ = []
        for z in range(5):
            total_slots = ZONES[z]["wheels"] * WHEEL_SIZE
            filled = sum(
                1 for w in range(ZONES[z]["wheels"])
                for s in range(WHEEL_SIZE)
                if self._slots[z][w][s] is not None
            )
            zone_occ.append(filled / total_slots)
        zone_imbalance = float(np.std(zone_occ))
        reward = (
            - avg_wait
            + 0.01 * throughput_this_step
            - 0.02 * zone_imbalance
        )

        # 7. Save prev queue lengths, advance step
        self._prev_queue = [
            len(self._arrival_q[z]) + len(self._retrieval_q[z]) for z in range(5)
        ]
        self._step += 1
        done = self._step >= self._max_steps

        return self._make_orchestrator_obs(reward=round(reward, 6), done=done)

    def get_zone_obs(self, zone_id: int) -> ZoneObs:
        z = zone_id
        n_wheels = ZONES[z]["wheels"]
        w_occ, w_ql, w_rc = [], [], []
        for w in range(n_wheels):
            filled = sum(1 for s in self._slots[z][w] if s is not None)
            w_occ.append(round(filled / WHEEL_SIZE, 4))
            # filled slots as effective queue depth — drives greedy away from full wheels
            w_ql.append(filled)
            # rotation cost to nearest empty slot; WHEEL_SIZE signals wheel is full
            rc = WHEEL_SIZE
            for d in range(1, WHEEL_SIZE + 1):
                cw  = (self._front_slot[z][w] + d) % WHEEL_SIZE
                ccw = (self._front_slot[z][w] - d) % WHEEL_SIZE
                if self._slots[z][w][cw] is None or self._slots[z][w][ccw] is None:
                    rc = d
                    break
            w_rc.append(float(rc))
        # Service rate over the episode: fraction of cars parked vs (parked + overflowed).
        # Shifted to [-0.5, +0.5] so 0 = break-even, positive = healthy, negative = overflowing.
        total_served = self._zone_parked_total[z] + self._zone_overflowed_total[z]
        zone_reward = round(
            self._zone_parked_total[z] / max(1, total_served) - 0.5, 6
        )
        return ZoneObs(
            zone_id=z,
            wheel_occupancy=w_occ,
            wheel_queue_lengths=w_ql,
            est_rotation_cost=w_rc,
            local_arrival_rate_ema=self._ema[z],
            time_of_day=round(_current_hour_offset(self._step) / 16.0, 4),
            step=self._step,
            done=self._step >= self._max_steps,
            reward=zone_reward,
        )


# ---------------------------------------------------------------------------
# PART C — Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    assert _rotation_cost(0, 3) == 3,  f"Expected 3, got {_rotation_cost(0, 3)}"
    assert _rotation_cost(0, 9) == 3,  f"Expected 3, got {_rotation_cost(0, 9)}"
    assert _rotation_cost(0, 6) == 6,  f"Expected 6, got {_rotation_cost(0, 6)}"
    assert _open_score(0.0) == 0.001
    assert _open_score(1.0) == 0.999
    assert _open_score(0.5) == 0.5
    assert _arrival_rate(1.5) == 0.40
    assert _current_hour_offset(120) == 2.0
    print("ITEM 2.1 DONE")

    import sys
    sys.path.insert(0, "..")
    env = RPOEXEnv(seed=42)
    obs = env.reset()
    assert isinstance(obs, OrchestratorObs), "reset() must return OrchestratorObs"
    assert len(obs.zone_occupancy) == 5
    assert obs.step == 0
    assert obs.done == False
    zone_obs = env.get_zone_obs(0)
    assert isinstance(zone_obs, ZoneObs)
    assert zone_obs.zone_id == 0
    print("ITEM 2.2 DONE")

    obs2 = env.step(OrchestratorAction(action="route_to_zone", zone_id=0))
    assert isinstance(obs2, OrchestratorObs)
    assert obs2.step == 1
    env2 = RPOEXEnv(seed=42, max_steps=200)
    env2.reset()
    for _ in range(16):
        env2.step(OrchestratorAction(action="route_to_zone", zone_id=2))
    assert env2._overflowed >= 0
    print("ITEM 2.3 DONE")
