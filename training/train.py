"""RPOE-X training utilities for GRPO with HF TRL.

Importable as:
    from rpoe_x.training.train import (
        ZONE_WHEELS, ZONE_NAMES,
        ORCH_SYSTEM, ZONE_SYSTEM,
        RPOEXHTTPClient,
        _obs_to_text, _zone_obs_to_text,
        _greedy_zone_id, _greedy_wheel_id,
        _proxy_step_reward,
        reward_total,
        build_rollout_func,
        plot_rewards,
        parse_action,
    )
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ── Zone configuration ────────────────────────────────────────────────────────

ZONE_WHEELS = [4, 4, 5, 4, 3]   # wheels per zone — matches server/env.py ZONES
ZONE_NAMES  = [
    "Cyber Towers Junction",
    "Inorbit Mall Signal",
    "Hitech City Metro",
    "Mindspace Junction",
    "Kondapur / Whitefields",
]

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

# ── Sync HTTP client ──────────────────────────────────────────────────────────

class RPOEXHTTPClient:
    """Thin sync HTTP client for RPOE-X — used in rollout and evaluation."""

    def __init__(self, base_url: str, task_id: str = "task2_medium", timeout: int = 30):
        self.base_url   = base_url.rstrip("/")
        self.task_id    = task_id
        self.timeout    = timeout
        self.session_id = None

    def reset(self, seed: int | None = None) -> dict:
        payload: dict[str, Any] = {"task_id": self.task_id}
        if seed is not None:
            payload["seed"] = seed
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        self.session_id = data.get("session_id")
        return data

    def step(self, action: str, zone_id: int, wheel_id: int | None = None) -> dict:
        payload: dict[str, Any] = {"action": action, "zone_id": zone_id}
        if wheel_id is not None:
            payload["wheel_id"] = wheel_id
        if self.session_id:
            payload["session_id"] = self.session_id
        r = requests.post(f"{self.base_url}/step", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


# ── Observation formatters ────────────────────────────────────────────────────

def _obs_to_text(obs: dict) -> str:
    """Format orchestrator observation for the model prompt."""
    zone_occ = [round(x, 2) for x in obs.get("zone_occupancy", [])]
    zone_q   = obs.get("zone_queue_lengths", [])
    zone_w   = [round(x, 1) for x in obs.get("zone_avg_wait", [])]
    return (
        f"step={obs.get('step', 0)} time={round(obs.get('time_of_day', 0.0), 3)}\n"
        f"zone_occupancy={zone_occ}\nzone_queue_lengths={zone_q}\nzone_avg_wait={zone_w}\n"
        "Which zone_id (0-4) should the next car route to?"
    )


def _zone_obs_to_text(obs: dict, zone_id: int) -> str:
    """Format zone-level observation for the zone agent model prompt."""
    n = ZONE_WHEELS[zone_id]
    zone_obs_list = obs.get("zone_observations", [])
    if zone_id < len(zone_obs_list):
        z         = zone_obs_list[zone_id]
        wheel_occ = [round(x, 2) for x in z.get("wheel_occupancy", [0.0] * n)]
        wheel_q   = z.get("wheel_queue_lengths", [0] * n)
        rot_cost  = [round(x, 1) for x in z.get("est_rotation_cost", [3.0] * n)]
    else:
        # Approximate from orchestrator-level obs when zone_observations is absent
        occ       = obs.get("zone_occupancy", [0.0] * 5)[zone_id]
        wheel_occ = [round(occ, 2)] * n
        wheel_q   = [obs.get("zone_queue_lengths", [0] * 5)[zone_id] // max(n, 1)] * n
        rot_cost  = [3.0] * n
    tod = round(obs.get("time_of_day", 0.0), 3)
    return (
        f"zone_id={zone_id} n_wheels={n} time={tod}\n"
        f"wheel_occupancy={wheel_occ}\n"
        f"wheel_queue_lengths={wheel_q}\n"
        f"est_rotation_cost={rot_cost}\n"
        f"Which wheel_id (0-{n - 1}) should the car be assigned to?"
    )


# ── Greedy baselines ──────────────────────────────────────────────────────────

def _greedy_zone_id(obs: dict) -> int:
    """Route to the zone with the highest queue depth (lowest occupancy breaks ties)."""
    queues = obs.get("zone_queue_lengths", [0] * 5)
    occs   = obs.get("zone_occupancy", [0.0] * 5)
    max_q  = max(queues) if queues else 0
    if max_q == 0:
        return 2  # default to Metro buffer when system is idle
    candidates = [i for i, q in enumerate(queues) if q == max_q]
    return min(candidates, key=lambda i: occs[i])


def _greedy_wheel_id(zone_id: int, obs: dict) -> int:
    """Pick wheel with lowest filled-slots + rotation-cost composite."""
    n = ZONE_WHEELS[zone_id]
    zone_obs_list = obs.get("zone_observations", [])
    if zone_id < len(zone_obs_list):
        z        = zone_obs_list[zone_id]
        wheel_q  = z.get("wheel_queue_lengths", [0] * n)
        rot_cost = z.get("est_rotation_cost", [3.0] * n)
        return min(range(n), key=lambda w: wheel_q[w] + rot_cost[w])
    return 0


# ── Proxy step reward ─────────────────────────────────────────────────────────

def _proxy_step_reward(pre_obs: dict, post_obs: dict, zone_id: int) -> float:
    """Observation-based proxy reward — always in [0, 1], always has variance.

    Raw env step rewards are dominated by -avg_wait_time and sum to a large
    negative value over 50 steps, clamping to 0 after normalisation and
    producing zero GRPO gradient. This proxy uses observable state instead.

    routing_score : 1.0 if the chosen zone had queued cars before the step.
                    Direct proxy for task1 service rate. Has structural variance
                    because different model decisions pick different zones.
    balance_score : 1 - std(zone_occupancy) / 0.5. Always non-zero (floor ~0.3)
                    so reward_std never collapses to 0 in quiet periods.

    Expected range:
        early morning, bad routing  → ~0.24
        morning surge, good routing → ~0.70+
    """
    pre_q = pre_obs.get("zone_queue_lengths", [0] * 5)
    routing_score = 1.0 if zone_id < len(pre_q) and pre_q[zone_id] > 0 else 0.0

    occ      = post_obs.get("zone_occupancy", [0.5] * 5)
    mean_occ = sum(occ) / len(occ) if occ else 0.5
    std_occ  = (sum((x - mean_occ) ** 2 for x in occ) / len(occ)) ** 0.5 if occ else 0.0
    balance_score = max(0.0, 1.0 - std_occ / 0.5)

    return 0.7 * routing_score + 0.3 * balance_score


# ── TRL reward function ───────────────────────────────────────────────────────

def reward_total(completions: list, **kwargs) -> list[float]:
    """Pass-through reward: reads the reward baked into each completion dict.

    rollout_func returns completions as {"reward": float, "content": str}.
    reward_total extracts those values so TRL can log and use them for GRPO.
    """
    return [
        float(c.get("reward", 0.0)) if isinstance(c, dict) else 0.0
        for c in completions
    ]


# ── Rollout function ──────────────────────────────────────────────────────────

def build_rollout_func(env_url: str, tokenizer, max_turns: int = 50):
    """Return a GRPOTrainer-compatible rollout_func.

    Single shared model, dual role per env step:
      1. Called with ORCH_SYSTEM → picks zone_id  (orchestrator)
      2. Called with ZONE_SYSTEM → picks wheel_id (zone agent)

    Reward: _proxy_step_reward() summed over turns, divided by max_turns.
    Structurally in [0, 1] with real variance even during quiet periods.
    """
    import torch

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def _infer(model, tok, messages: list[dict], lo: int, hi: int) -> int:
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inp = tok(prompt, return_tensors="pt").to(_device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=16, temperature=0.9,
                do_sample=True, pad_token_id=tok.eos_token_id,
            )
        if was_training:
            model.train()
        text = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        for d in re.findall(r"\b(\d+)\b", text):
            if lo <= int(d) <= hi:
                return int(d)
        return lo  # fallback: lowest valid index

    def _rollout(prompts, model=None, processing_class=None, **kwargs):
        tok = processing_class or tokenizer
        all_completions, all_rewards = [], []
        for _ in prompts:
            client    = RPOEXHTTPClient(base_url=env_url, task_id="task2_medium")
            data      = client.reset()
            obs       = data.get("observation", {})
            ep_reward = 0.0
            turns     = 0
            text      = ""  # compact action log: "z,w;" — stays well under token limit
            while not obs.get("done", False) and turns < max_turns:
                pre_obs  = obs
                zone_id  = _infer(model, tok, [
                    {"role": "system", "content": ORCH_SYSTEM},
                    {"role": "user",   "content": _obs_to_text(obs)},
                ], lo=0, hi=4)
                n_wheels = ZONE_WHEELS[zone_id]
                wheel_id = _infer(model, tok, [
                    {"role": "system", "content": ZONE_SYSTEM},
                    {"role": "user",   "content": _zone_obs_to_text(obs, zone_id)},
                ], lo=0, hi=n_wheels - 1)
                resp      = client.step("route_to_zone", zone_id, wheel_id=wheel_id)
                obs       = resp.get("observation", {})
                ep_reward += _proxy_step_reward(pre_obs, obs, zone_id)
                text      += f"{zone_id},{wheel_id};"
                turns     += 1

            normalised = min(1.0, ep_reward / max(max_turns, 1))
            all_completions.append({"reward": normalised, "content": text})
            all_rewards.append(normalised)
        return all_completions, all_rewards

    return _rollout


# ── Reward curve plotter ──────────────────────────────────────────────────────

def plot_rewards(trainer, save_path: str | None = None) -> None:
    """Plot GRPO reward and loss curves from trainer log history."""
    import matplotlib.pyplot as plt
    import pandas as pd

    log_df      = pd.DataFrame(trainer.state.log_history)
    reward_cols = [c for c in log_df.columns if "reward" in c.lower()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for col in reward_cols:
        if "step" in log_df.columns:
            axes[0].plot(log_df["step"], log_df[col], alpha=0.7, label=col)
    if "step" in log_df.columns and reward_cols:
        smoothed = log_df[reward_cols[0]].rolling(10, min_periods=1).mean()
        axes[0].plot(log_df["step"], smoothed, linewidth=2.5, color="darkblue", label="smoothed")
    axes[0].set_title("GRPO Reward")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    if "loss" in log_df.columns and "step" in log_df.columns:
        axes[1].plot(log_df["step"], log_df["loss"], color="orange", label="loss")
        axes[1].set_title("Training Loss")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved to {save_path}")
    plt.show()


# ── JSON action parser ────────────────────────────────────────────────────────

def parse_action(completion: str) -> dict[str, Any] | None:
    """Parse JSON action from model output, stripping Qwen <think> blocks."""
    if not isinstance(completion, str):
        return None
    import json
    text = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
    text = re.sub(r"```.*?```",          "", text,       flags=re.DOTALL).strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    for match in re.findall(r"\{[^{}]*\}", text):
        try:
            result = json.loads(match)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue
    return None
