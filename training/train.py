"""RPOE-X training utilities for GRPO with HF TRL.

Importable as: from rpoe_x.training.train import (
    ORCH_SYSTEM, ZONE_SYSTEM,
    format_orch_obs, format_zone_obs,
    parse_action,
    reward_total,
    rollout_once,
    build_rollout_func,
    plot_rewards,
)
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

import numpy as np

try:
    from ..models import ParkingAction
except ImportError:
    from models import ParkingAction

logger = logging.getLogger(__name__)

# ── Zone configuration ────────────────────────────────────────────────────────

ZONE_WHEEL_COUNTS = [4, 4, 5, 4, 3]  # wheels per zone, matches server/env.py
ZONE_NAMES = ["Cyber Towers", "Inorbit Mall", "Hitech Metro", "Mindspace", "Kondapur"]

# ── System prompts ────────────────────────────────────────────────────────────

ORCH_SYSTEM = """\
You are the Orchestrator Agent for RPOE-X, a rotary parking system in HITEC City, Hyderabad.

ZONES:
  0 = Cyber Towers Junction
  1 = Inorbit Mall Signal
  2 = Hitech City Metro
  3 = Mindspace Junction
  4 = Kondapur / Whitefields

YOUR JOB: Each step, route ONE incoming car to ONE zone by choosing its zone_id.
Goal: keep wait times low and throughput high across all zones.

OBSERVATIONS (received each step):
  zone_occupancy[z]      — fraction of slots filled in zone z (0.0 = empty, 1.0 = full)
  zone_queue_lengths[z]  — cars currently waiting to be parked in zone z
  zone_avg_wait[z]       — average steps cars have been waiting in zone z
  arrival_rate_ema[z]    — recent arrival rate trend for zone z
  time_of_day            — normalized time (0.0 = start, 1.0 = end of operating hours)

HARD CONSTRAINT:
  Do not route to a zone where zone_occupancy >= 0.98 — it has no remaining capacity.

OUTPUT — respond with exactly this JSON and nothing else:
{"zone_id": <integer 0 to 4>}
/no_think"""

ZONE_SYSTEM = """\
You are a Zone Agent for RPOE-X, a rotary parking system in HITEC City, Hyderabad.

YOUR JOB: A car has been routed to your zone. Assign it to the best available wheel.
Goal: minimize service time and avoid overflow penalties.

OBSERVATIONS (received each step):
  zone_id                  — which zone you are managing (0–4)
  wheel_occupancy[w]       — fraction of the 12 slots filled on wheel w (0.0 = empty, 1.0 = full)
  wheel_queue_lengths[w]   — filled slot count per wheel (12 means the wheel is completely full)
  est_rotation_cost[w]     — steps to rotate to the nearest empty slot (higher = slower service)
  time_of_day              — normalized time of day

HARD CONSTRAINT:
  Never assign to a wheel where wheel_occupancy >= 0.99 or wheel_queue_lengths >= 12.
  Doing so causes an overflow — the car is lost and a penalty is applied.

OUTPUT — respond with exactly this JSON and nothing else:
{"wheel_id": <integer>}
/no_think"""

# ── Observation formatters ────────────────────────────────────────────────────


def format_orch_obs(obs: dict[str, Any]) -> str:
    return (
        f"zone_occupancy: {[round(x, 2) for x in obs['zone_occupancy']]}\n"
        f"zone_queue_lengths: {obs['zone_queue_lengths']}\n"
        f"zone_avg_wait: {[round(x, 1) for x in obs['zone_avg_wait']]}\n"
        f"arrival_rate_ema: {[round(x, 3) for x in obs['arrival_rate_ema']]}\n"
        f"time_of_day: {obs['time_of_day']:.3f}  step: {obs['step']}\n"
        "Which zone_id (0-4) should the next car be routed to?"
    )


def format_zone_obs(obs: dict[str, Any]) -> str:
    n = len(obs["wheel_occupancy"])
    return (
        f"zone_id: {obs['zone_id']}\n"
        f"wheel_occupancy: {[round(x, 2) for x in obs['wheel_occupancy']]}\n"
        f"wheel_queue_lengths: {obs['wheel_queue_lengths']}\n"
        f"est_rotation_cost: {[round(x, 1) for x in obs['est_rotation_cost']]}\n"
        f"time_of_day: {obs['time_of_day']:.3f}  step: {obs['step']}\n"
        f"Which wheel_id (0-{n - 1}) should the car be assigned to?"
    )


# ── Action parser ─────────────────────────────────────────────────────────────


def parse_action(completion: str) -> dict[str, Any] | None:
    """Parse JSON action from model output, stripping Qwen <think> chains."""
    if not isinstance(completion, str):
        return None
    text = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    matches = re.findall(r"\{[^{}]*\}", text)
    if matches:
        try:
            result = json.loads(matches[-1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return None


# ── Pass-through TRL reward function ─────────────────────────────────────────


def reward_total(completions: list[str], **kwargs) -> list[float]:
    """Pass-through: rewards were captured during rollout, not recomputed here."""
    rewards = kwargs.get("total_reward")
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]


# ── Rollout — one full parking episode ───────────────────────────────────────


async def rollout_once(
    trainer,
    env,
    tokenizer,
    max_turns: int = 50,
) -> dict[str, list]:
    """Run one parking episode. Reward comes entirely from env.step() — no recomputation.

    Each turn: orchestrator picks zone_id, zone agent picks wheel_id, env.step() returns reward.
    Invalid-output penalties (-0.5 / -0.1) are applied here in rollout, not in reward_funcs.
    Token ids from both agents are accumulated into one sequence per episode.
    """
    from trl.experimental.openenv import generate_rollout_completions

    result = await env.reset()
    obs = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []

    for _ in range(max_turns):
        if result.done:
            break

        # ── Orchestrator turn ─────────────────────────────────────────────────
        orch_obs_dict = {
            "zone_occupancy":    list(obs.zone_occupancy),
            "zone_queue_lengths": list(obs.zone_queue_lengths),
            "zone_avg_wait":     list(obs.zone_avg_wait),
            "arrival_rate_ema":  list(obs.arrival_rate_ema),
            "time_of_day":       obs.time_of_day,
            "step":              obs.step,
        }
        orch_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": ORCH_SYSTEM},
             {"role": "user",   "content": format_orch_obs(orch_obs_dict)}],
            add_generation_prompt=True,
            tokenize=False,
        )
        orch_out = generate_rollout_completions(trainer, [orch_prompt])[0]
        prompt_ids.extend(orch_out["prompt_ids"])
        completion_ids.extend(orch_out["completion_ids"])
        logprobs.extend(orch_out["logprobs"])

        orch_text = orch_out.get("text") or tokenizer.decode(
            orch_out["completion_ids"], skip_special_tokens=True
        )
        parsed_orch = parse_action(orch_text)

        if parsed_orch is None or "zone_id" not in parsed_orch:
            step_rewards.append(-0.5)
            continue
        zone_id = int(parsed_orch["zone_id"])
        if not (0 <= zone_id <= 4):
            step_rewards.append(-1.0)
            continue

        # ── Zone agent turn ───────────────────────────────────────────────────
        n_wheels = ZONE_WHEEL_COUNTS[zone_id]
        base_occ = list(obs.zone_occupancy)[zone_id]
        wheel_occ = [
            max(0.0, min(1.0, base_occ + random.gauss(0, 0.15)))
            for _ in range(n_wheels)
        ]
        if all(w > 0.9 for w in wheel_occ):
            wheel_occ[random.randint(0, n_wheels - 1)] = random.uniform(0.1, 0.5)

        zone_obs_dict = {
            "zone_id":            zone_id,
            "wheel_occupancy":    wheel_occ,
            "wheel_queue_lengths": [max(0, int(w * 12)) for w in wheel_occ],
            "est_rotation_cost":  [max(1.0, (1.0 - w) * 12) for w in wheel_occ],
            "time_of_day":        obs.time_of_day,
            "step":               obs.step,
        }
        zone_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": ZONE_SYSTEM},
             {"role": "user",   "content": format_zone_obs(zone_obs_dict)}],
            add_generation_prompt=True,
            tokenize=False,
        )
        zone_out = generate_rollout_completions(trainer, [zone_prompt])[0]
        prompt_ids.extend(zone_out["prompt_ids"])
        completion_ids.extend(zone_out["completion_ids"])
        logprobs.extend(zone_out["logprobs"])

        zone_text = zone_out.get("text") or tokenizer.decode(
            zone_out["completion_ids"], skip_special_tokens=True
        )
        parsed_zone = parse_action(zone_text)

        if parsed_zone is None or "wheel_id" not in parsed_zone:
            step_rewards.append(-0.5)
            continue
        wheel_id = int(parsed_zone["wheel_id"])
        if not (0 <= wheel_id < n_wheels):
            step_rewards.append(-1.0)
            continue

        # ── Env step — sole reward source ─────────────────────────────────────
        try:
            result = await env.step(ParkingAction(zone_id=zone_id, wheel_id=wheel_id))
            step_rewards.append(float(result.reward or 0.0))
            obs = result.observation
        except Exception as e:
            logger.warning(f"env.step error: {e}")
            step_rewards.append(-0.1)
            break

    total_reward = sum(step_rewards) if step_rewards else -1.0
    return {
        "prompt_ids":     prompt_ids,
        "completion_ids": completion_ids,
        "logprobs":       logprobs,
        "total_reward":   total_reward,
    }


def build_rollout_func(env_url: str, tokenizer, max_turns: int = 50):
    """Return a GRPOTrainer-compatible rollout_func.

    Creates a fresh ParkingEnv connection per episode so rollout_func
    (which is called synchronously by TRL) can drive the async env.
    Requires nest_asyncio in Colab — call nest_asyncio.apply() before training.
    """
    import asyncio

    from ..client import ParkingEnv

    async def _run_episode(trainer) -> dict:
        async with ParkingEnv(base_url=env_url) as env:
            return await rollout_once(trainer, env, tokenizer, max_turns)

    def rollout_func(prompts: list[str], trainer) -> dict[str, list]:
        loop = asyncio.get_event_loop()
        ep_prompt_ids, ep_completion_ids, ep_logprobs, total_rewards = [], [], [], []
        for _ in prompts:
            episode = loop.run_until_complete(_run_episode(trainer))
            ep_prompt_ids.append(episode["prompt_ids"])
            ep_completion_ids.append(episode["completion_ids"])
            ep_logprobs.append(episode["logprobs"])
            total_rewards.append(episode["total_reward"])
            logger.info(f"episode reward={episode['total_reward']:.3f}")
        return {
            "prompt_ids":     ep_prompt_ids,
            "completion_ids": ep_completion_ids,
            "logprobs":       ep_logprobs,
            "total_reward":   total_rewards,
        }

    return rollout_func


# ── Reward curve plotter ──────────────────────────────────────────────────────


def plot_rewards(trainer, save_path: str) -> None:
    """Save the GRPO reward curve from trainer log history."""
    import matplotlib.pyplot as plt

    steps, rewards = [], []
    for log in trainer.state.log_history:
        if "step" in log and "reward" in log:
            steps.append(log["step"])
            rewards.append(log["reward"])
    if not rewards:
        print("No reward data to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rewards, color="steelblue", alpha=0.6, linewidth=1, label="Per-step")
    if len(rewards) >= 10:
        smoothed = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        ax.plot(steps[9:], smoothed, color="darkblue", linewidth=2.5, label="Smoothed (10-step)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("GRPO Reward")
    ax.set_title("RPOE-X — GRPO Training Reward Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Reward curve saved to {save_path}")
