"""RPOE-X Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ParkingAction, OrchestratorObs


class ParkingEnv(EnvClient[ParkingAction, OrchestratorObs, State]):
    """
    Client for the RPOE-X Rotary Parking Optimization Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with ParkingEnv.from_env("bharavivillu/rpoe-x") as env:
        ...     env.reset()
        ...     result = env.step(ParkingAction(zone_id=2, wheel_id=0))
        ...     print(result.observation.zone_occupancy)
    """

    def _step_payload(self, action: ParkingAction) -> Dict:
        return {
            "action":   action.action,
            "zone_id":  action.zone_id,
            "wheel_id": action.wheel_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[OrchestratorObs]:
        obs_data = payload.get("observation", payload)
        observation = OrchestratorObs(
            zone_occupancy      = obs_data.get("zone_occupancy",       [0.0] * 5),
            zone_queue_lengths  = obs_data.get("zone_queue_lengths",   [0] * 5),
            zone_avg_wait       = obs_data.get("zone_avg_wait",        [0.0] * 5),
            arrival_rate_ema    = obs_data.get("arrival_rate_ema",     [0.0] * 5),
            recent_delta_queue  = obs_data.get("recent_delta_queue",   [0.0] * 5),
            time_of_day         = obs_data.get("time_of_day",          0.0),
            step                = obs_data.get("step",                 0),
            done                = payload.get("done",                  False),
            reward              = payload.get("reward",                0.0),
        )
        return StepResult(
            observation = observation,
            reward      = payload.get("reward"),
            done        = payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id = payload.get("episode_id"),
            step_count = payload.get("step_count", 0),
        )
