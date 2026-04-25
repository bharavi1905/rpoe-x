"""RPOE-X training utilities for GRPO with HF TRL.

Importable as: from rpoe_x.training.train import (
    ORCH_SYSTEM, ZONE_SYSTEM,
    format_orch_obs, format_zone_obs,
    parse_action,
    format_reward, routing_reward, wheel_reward,
    collect_episode,
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

ORCH_SYSTEM = (
    "You are an orchestrator agent for a rotary parking system in HITEC City, Hyderabad.\n"
    "Each step you route ONE car to ONE zone. Routing to an empty zone wastes the step.\n"
    "RULE: Route to a zone with cars waiting (zone_queue_lengths > 0). Prefer the busiest zone.\n"
    "If no zone has cars, use zone 2 (Hitech Metro — largest buffer).\n"
    "Zones: 0=Cyber Towers, 1=Inorbit Mall, 2=Hitech Metro, 3=Mindspace, 4=Kondapur.\n"
    'Respond ONLY with: {"zone_id": <int 0-4>} /no_think'
)

ZONE_SYSTEM = (
    "You are a zone agent for a rotary parking system.\n"
    "Assign the incoming car to the best wheel in your zone.\n"
    "RULE: Never pick a full wheel (occupancy=1.0 or queue_length=12). Pick the least-occupied wheel.\n"
    'Respond ONLY with: {"wheel_id": <int>} /no_think'
)

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


# ── Reward functions ──────────────────────────────────────────────────────────


def format_reward(completions: list[str], agent_role: list[str], **kwargs) -> list[float]:
    """Reward valid JSON with the correct key for each agent role."""
    rewards = []
    for comp, role in zip(completions, agent_role):
        parsed = parse_action(comp)
        if parsed is None:
            rewards.append(-1.0)
        elif role == "orchestrator" and "zone_id" in parsed:
            z = int(parsed.get("zone_id", -1))
            rewards.append(0.5 if 0 <= z <= 4 else -1.0)
        elif role == "zone" and "wheel_id" in parsed:
            w = int(parsed.get("wheel_id", -1))
            rewards.append(0.5 if w >= 0 else -1.0)
        else:
            rewards.append(-1.0)
    return rewards


def routing_reward(
    completions: list[str],
    agent_role: list[str],
    zone_queue_lengths: list[str],
    **kwargs,
) -> list[float]:
    """Reward routing to zones with queued cars; penalise routing to empty zones.

    zone_queue_lengths is stored as a JSON string in the dataset to avoid
    TRL collation errors with variable-length lists.
    """
    rewards = []
    for comp, role, ql_raw in zip(completions, agent_role, zone_queue_lengths):
        ql = json.loads(ql_raw) if isinstance(ql_raw, str) else ql_raw
        if role != "orchestrator" or not ql:
            rewards.append(0.0)
            continue
        parsed = parse_action(comp)
        if parsed is None or "zone_id" not in parsed:
            rewards.append(-0.5)
            continue
        z = int(parsed["zone_id"])
        if not (0 <= z <= 4):
            rewards.append(-1.0)
            continue
        total = sum(ql)
        if total == 0:
            rewards.append(0.0)
        elif ql[z] == 0:
            rewards.append(-0.5)
        else:
            rewards.append(min(1.0, ql[z] / total + 0.3))
    return rewards


def wheel_reward(
    completions: list[str],
    agent_role: list[str],
    wheel_occupancy: list[str],
    n_wheels: list[int],
    **kwargs,
) -> list[float]:
    """Penalise full-wheel assignments; reward choosing low-occupancy wheels.

    wheel_occupancy is stored as a JSON string in the dataset to avoid
    TRL collation errors with variable-length lists.
    """
    rewards = []
    for comp, role, occ_raw, nw in zip(completions, agent_role, wheel_occupancy, n_wheels):
        occ = json.loads(occ_raw) if isinstance(occ_raw, str) else occ_raw
        if role != "zone" or not occ:
            rewards.append(0.0)
            continue
        parsed = parse_action(comp)
        if parsed is None or "wheel_id" not in parsed:
            rewards.append(-0.5)
            continue
        w = int(parsed["wheel_id"])
        if not (0 <= w < nw):
            rewards.append(-1.0)
            continue
        occ_val = occ[w]
        rewards.append(-0.8 if occ_val >= 0.99 else max(0.5, 1.0 - occ_val))
    return rewards


# ── Episode collector ─────────────────────────────────────────────────────────


async def collect_episode(env, tokenizer, max_turns: int = 50) -> list[dict]:
    """Run one greedy episode; return rows for GRPO training.

    The prompt column is pre-formatted via tokenizer.apply_chat_template so
    it's stored as a string — TRL's GRPOTrainer requires a string prompt column.

    Variable-length list columns (zone_queue_lengths, wheel_occupancy) are
    JSON-serialised to avoid TRL batch-collation errors.
    """
    rows = []
    result = await env.reset()
    obs = result.observation
    done = False

    for _ in range(max_turns):
        if done:
            break
        ql = list(obs.zone_queue_lengths)
        zo = list(obs.zone_occupancy)

        # Orchestrator training row
        orch_obs_dict = {
            "zone_occupancy": zo,
            "zone_queue_lengths": ql,
            "zone_avg_wait": list(obs.zone_avg_wait),
            "arrival_rate_ema": list(obs.arrival_rate_ema),
            "time_of_day": obs.time_of_day,
            "step": obs.step,
        }
        orch_messages = [
            {"role": "system", "content": ORCH_SYSTEM},
            {"role": "user", "content": format_orch_obs(orch_obs_dict)},
        ]
        rows.append({
            "prompt": tokenizer.apply_chat_template(
                orch_messages, add_generation_prompt=True, tokenize=False
            ),
            "agent_role": "orchestrator",
            "zone_queue_lengths": json.dumps(ql),
            "wheel_occupancy": json.dumps([]),
            "n_wheels": 0,
        })

        # Greedy zone selection; estimate wheel occupancy from zone-level signal
        zone_id = max(range(5), key=lambda z: ql[z])
        n_wheels = ZONE_WHEEL_COUNTS[zone_id]
        base_occ = zo[zone_id]
        wheel_occ = [
            max(0.0, min(1.0, base_occ + random.gauss(0, 0.15)))
            for _ in range(n_wheels)
        ]
        if all(w > 0.9 for w in wheel_occ):
            wheel_occ[random.randint(0, n_wheels - 1)] = random.uniform(0.1, 0.5)

        zone_obs_dict = {
            "zone_id": zone_id,
            "wheel_occupancy": wheel_occ,
            "wheel_queue_lengths": [max(0, int(w * 12)) for w in wheel_occ],
            "est_rotation_cost": [max(1.0, (1.0 - w) * 12) for w in wheel_occ],
            "time_of_day": obs.time_of_day,
            "step": obs.step,
        }
        zone_messages = [
            {"role": "system", "content": ZONE_SYSTEM},
            {"role": "user", "content": format_zone_obs(zone_obs_dict)},
        ]
        rows.append({
            "prompt": tokenizer.apply_chat_template(
                zone_messages, add_generation_prompt=True, tokenize=False
            ),
            "agent_role": "zone",
            "zone_queue_lengths": json.dumps([]),
            "wheel_occupancy": json.dumps(wheel_occ),
            "n_wheels": n_wheels,
        })

        wheel_id = min(range(n_wheels), key=lambda w: wheel_occ[w])
        step_result = await env.step(ParkingAction(zone_id=zone_id, wheel_id=wheel_id))
        obs = step_result.observation
        done = step_result.done

    return rows


# ── Reward curve plotter ──────────────────────────────────────────────────────


def plot_rewards(trainer, save_path: str) -> None:
    """Save and display the GRPO reward curve from trainer log history."""
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
