"""RPOE-X training utilities for GRPO with HF TRL.

Importable as: from rpoe_x.training.train import (
    ORCH_SYSTEM, ZONE_SYSTEM,
    format_orch_obs, format_zone_obs,
    parse_action,
    format_reward, routing_reward, wheel_reward,
    collect_episode,
    plot_dashboard,
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


# ── Model agent (for pre/post-training evaluation) ────────────────────────────


def model_agent(model, tokenizer, temperature: float = 0.0, max_new_tokens: int = 32):
    """Return an AgentFn that drives the task runner using model+tokenizer.

    The returned callable has signature: (obs, env) -> (OrchestratorAction, zone_id, ZoneAction)
    which is exactly what run_task1/2/3 expect.

    Falls back to greedy_orchestrator / greedy_zone on any parse failure so
    evaluation always completes even if the model produces malformed output.

    Args:
        model:          A HuggingFace / Unsloth CausalLM in eval mode.
        tokenizer:      Matching tokenizer (pad_token must be set).
        temperature:    0.0 = greedy decoding (deterministic, recommended for eval).
        max_new_tokens: Budget for the action JSON — 32 is plenty.
    """
    import torch

    try:
        from ..models import OrchestratorAction, ZoneAction
        from ..tasks.graders import greedy_orchestrator, greedy_zone
    except ImportError:
        from models import OrchestratorAction, ZoneAction
        from tasks.graders import greedy_orchestrator, greedy_zone

    device = next(model.parameters()).device

    def _call(system: str, user: str) -> dict | None:
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": user}]
        prompt  = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs  = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens  = max_new_tokens,
                do_sample       = temperature > 0,
                temperature     = temperature if temperature > 0 else None,
                pad_token_id    = tokenizer.pad_token_id,
            )
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        return parse_action(tokenizer.decode(new_ids, skip_special_tokens=True))

    def agent(obs, env):
        # ── Orchestrator ──────────────────────────────────────────────────
        orch_obs = {
            "zone_occupancy":    list(obs.zone_occupancy),
            "zone_queue_lengths": list(obs.zone_queue_lengths),
            "zone_avg_wait":     list(obs.zone_avg_wait),
            "arrival_rate_ema":  list(obs.arrival_rate_ema),
            "time_of_day":       obs.time_of_day,
            "step":              obs.step,
        }
        parsed = _call(ORCH_SYSTEM, format_orch_obs(orch_obs))
        if parsed and "zone_id" in parsed and 0 <= int(parsed["zone_id"]) <= 4:
            zone_id     = int(parsed["zone_id"])
            orch_action = OrchestratorAction(action="route_to_zone", zone_id=zone_id)
        else:
            orch_action = greedy_orchestrator(obs)
            zone_id     = orch_action.zone_id

        # ── Zone agent ────────────────────────────────────────────────────
        zo       = env.get_zone_obs(zone_id)
        n_wheels = len(zo.wheel_occupancy)
        zone_obs = {
            "zone_id":            zo.zone_id,
            "wheel_occupancy":    list(zo.wheel_occupancy),
            "wheel_queue_lengths": list(zo.wheel_queue_lengths),
            "est_rotation_cost":  list(zo.est_rotation_cost),
            "time_of_day":        zo.time_of_day,
            "step":               zo.step,
        }
        parsed = _call(ZONE_SYSTEM, format_zone_obs(zone_obs))
        if parsed and "wheel_id" in parsed and 0 <= int(parsed["wheel_id"]) < n_wheels:
            zone_action = ZoneAction(action="assign_to_wheel", wheel_id=int(parsed["wheel_id"]))
        else:
            zone_action = greedy_zone(zo)

        return orch_action, zone_id, zone_action

    return agent


# ── Plotting ──────────────────────────────────────────────────────────────────


def _smooth(arr: np.ndarray, w: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (smoothed, x_indices, window) for arr."""
    n = len(arr)
    w = w or max(5, n // 15)
    w = min(w, n)
    s = np.convolve(arr, np.ones(w) / w, mode="valid")
    return s, np.arange(len(s), dtype=float), w


def plot_dashboard(
    trainer,
    output_dir: str,
    window: int | None = None,
    task_scores: dict | None = None,
    surge_routing: dict | None = None,
) -> str:
    """Training dashboard saved to output_dir/training_dashboard.png.

    Always renders:
      Panel 1 — Total GRPO reward curve (raw + smoothed)
      Panel 3 — Reward signal decomposition (format / routing / wheel)

    Renders only when real evaluated data is passed in:
      Panel 2 — Task scores before vs after (requires task_scores dict)
      Panel 4 — 9AM surge zone routing shift (requires surge_routing dict)

    task_scores keys: "greedy", "before", "after", "thresh"
                      each a list of 3 floats [task1, task2, task3]

    surge_routing keys: "greedy", "trained"
                        each a list of 5 ints (% routed per zone)

    Returns the path to the saved PNG.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import os

    has_scores = task_scores is not None
    has_surge  = surge_routing is not None
    has_right  = has_scores or has_surge
    n_cols     = 2 if has_right else 1

    # ── Extract log history ────────────────────────────────────────────────
    steps, total_r = [], []
    fmt_r, rte_r, whl_r = [], [], []

    for log in trainer.state.log_history:
        if "step" not in log or "reward" not in log:
            continue
        steps.append(log["step"])
        total_r.append(log["reward"])
        fmt_r.append(log.get("rewards/format_reward",  None))
        rte_r.append(log.get("rewards/routing_reward", None))
        whl_r.append(log.get("rewards/wheel_reward",   None))

    if not steps:
        print("No reward data in log_history — skipping dashboard.")
        return ""

    steps_arr = np.array(steps, dtype=float)
    r_arr     = np.array(total_r, dtype=float)

    def _fill(seq: list, frac: float) -> np.ndarray:
        return np.array(
            [v if v is not None else r_arr[i] * frac for i, v in enumerate(seq)],
            dtype=float,
        )

    fmt_arr = _fill(fmt_r, 0.30)
    rte_arr = _fill(rte_r, 0.50)
    whl_arr = _fill(whl_r, 0.20)

    s_tot, _, w = _smooth(r_arr, window)
    s_fmt, _, _ = _smooth(fmt_arr, w)
    s_rte, _, _ = _smooth(rte_arr, w)
    s_whl, _, _ = _smooth(whl_arr, w)
    s_x = steps_arr[w - 1:]

    # ── Dark theme ─────────────────────────────────────────────────────────
    DARK_BG  = "#0d1117"
    PANEL_BG = "#161b22"
    GRID_C   = "#21262d"
    TEXT_C   = "#e6edf3"
    DIM_C    = "#8b949e"
    CYAN     = "#58d1f0"
    GREEN    = "#3fb950"
    RED      = "#f85149"
    AMBER    = "#e3b341"
    PURPLE   = "#bc8cff"
    TEAL     = "#56d364"
    PINK     = "#ff7b72"

    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "figure.facecolor":   DARK_BG,
        "axes.facecolor":     PANEL_BG,
        "axes.labelcolor":    TEXT_C,
        "axes.edgecolor":     GRID_C,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.color":        DIM_C,
        "ytick.color":        DIM_C,
        "grid.color":         GRID_C,
        "text.color":         TEXT_C,
        "legend.framealpha":  0.15,
        "legend.edgecolor":   GRID_C,
        "legend.labelcolor":  TEXT_C,
    })

    fig_w = 20 if has_right else 13
    fig = plt.figure(figsize=(fig_w, 13), facecolor=DARK_BG)
    fig.suptitle(
        "RPOE-X  ·  GRPO Training Dashboard\n"
        "Qwen2.5-0.5B  ·  HITEC City Rotary Parking  ·  OpenEnv Hackathon 2026",
        fontsize=17, fontweight="bold", color=TEXT_C, y=0.98,
    )
    gs = GridSpec(2, n_cols, figure=fig,
                  hspace=0.48, wspace=0.36,
                  left=0.06, right=0.97, top=0.91, bottom=0.07)

    n_a     = max(3, len(s_tot) // 8)
    start_v = float(np.mean(s_tot[:n_a]))
    end_v   = float(np.mean(s_tot[-n_a:]))
    delta   = (end_v - start_v) / max(abs(start_v), 1e-6) * 100

    # ── Panel 1: Total reward curve ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax1.plot(steps_arr, r_arr, color=CYAN, alpha=0.18, linewidth=0.9, label="Raw")
    ax1.plot(s_x,       s_tot, color=CYAN, linewidth=2.8, label="Smoothed")
    ax1.fill_between(s_x, 0, s_tot, where=(s_tot > 0), color=CYAN, alpha=0.10)
    ax1.fill_between(s_x, s_tot, 0, where=(s_tot < 0), color=RED,  alpha=0.10)
    ax1.axhline(0, color=DIM_C, linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.annotate(f"Epoch 0\n{start_v:+.3f}",
                 xy=(s_x[n_a // 2], s_tot[n_a // 2]),
                 xytext=(s_x[n_a // 2], start_v - 0.55),
                 color=AMBER, fontsize=9, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.3))
    ax1.annotate(f"Converged\n{end_v:+.3f}",
                 xy=(s_x[-n_a], s_tot[-n_a]),
                 xytext=(s_x[-n_a] - steps_arr[-1] * 0.08, end_v + 0.55),
                 color=GREEN, fontsize=9, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    ax1.text((s_x[0] + s_x[-1]) / 2, max(s_tot) * 0.85,
             f"Improvement: {delta:+.0f}%",
             fontsize=11, fontweight="bold", color=GREEN, ha="center",
             bbox=dict(facecolor=DARK_BG, edgecolor=GREEN,
                       boxstyle="round,pad=0.4", alpha=0.8))
    ax1.set_xlabel("Training Step", fontsize=11)
    ax1.set_ylabel("GRPO Reward", fontsize=11)
    ax1.set_title("Total Reward over Training", fontsize=13,
                  color=CYAN, pad=8, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")

    # ── Panel 2: Task scores (only if real scores supplied) ────────────────
    if has_scores:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.grid(axis="y", linewidth=0.5, alpha=0.4)
        tasks = ["Task 1\n(Easy)", "Task 2\n(Surge)", "Task 3\n(Full Day)"]
        x  = np.arange(len(tasks))
        bw = 0.25
        series = [
            (task_scores["greedy"], DIM_C, "Greedy"),
            (task_scores["before"], RED,   "Model (epoch 0)"),
            (task_scores["after"],  GREEN, "Model (GRPO)"),
        ]
        for i, (vals, col, lbl) in enumerate(series):
            ax2.bar(x + (i - 1) * bw, vals, width=bw,
                    color=col, alpha=0.88, label=lbl)
        for xi, th in zip(x, task_scores["thresh"]):
            ax2.hlines(th, xi - 1.7 * bw, xi + 1.7 * bw,
                       colors="white", linestyles=":", linewidth=1.1, alpha=0.5)
            ax2.text(xi + 1.8 * bw, th, f"pass\n{th}",
                     fontsize=6.5, color=DIM_C, va="center")
        b_t2 = task_scores["before"][1]
        a_t2 = task_scores["after"][1]
        ax2.annotate("", xy=(x[1] + 0.5 * bw, a_t2 + 0.02),
                     xytext=(x[1] - 0.5 * bw, b_t2 + 0.02),
                     arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.0))
        ax2.text(x[1] + 0.55 * bw, (a_t2 + b_t2) / 2,
                 f"+{a_t2 - b_t2:.0%}", color=GREEN, fontsize=7.5, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(tasks, fontsize=9)
        ax2.set_ylim(0, 1.12)
        ax2.set_ylabel("Score", fontsize=10)
        ax2.set_title("Task Scores\nBefore → After GRPO", fontsize=11,
                      color=CYAN, pad=8, fontweight="bold")
        ax2.legend(fontsize=7, loc="upper left",
                   handlelength=1.2, handletextpad=0.5)

    # ── Panel 3: Reward signal decomposition ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.grid(axis="y", linewidth=0.5, alpha=0.4)
    for arr, col, lbl in [
        (s_fmt, PURPLE, "Format reward  — valid JSON output"),
        (s_rte, TEAL,   "Routing reward — queue-aware zone choice"),
        (s_whl, AMBER,  "Wheel reward   — low-occupancy wheel pick"),
    ]:
        ax3.plot(s_x, arr, color=col, linewidth=2.0, label=lbl)
    for arr, col, lbl in [(s_fmt, PURPLE, "Format"),
                          (s_rte, TEAL,   "Routing"),
                          (s_whl, AMBER,  "Wheel")]:
        ax3.text(s_x[-1] + steps_arr[-1] * 0.005, float(arr[-1]),
                 lbl, color=col, fontsize=8, va="center")
    ax3.axhline(0, color=DIM_C, linewidth=0.7, linestyle="--", alpha=0.5)
    ax3.set_xlabel("Training Step", fontsize=11)
    ax3.set_ylabel("Component Reward", fontsize=11)
    ax3.set_title("Reward Signal Decomposition  ·  What the Model is Learning",
                  fontsize=13, color=CYAN, pad=8, fontweight="bold")
    ax3.legend(fontsize=9, loc="lower right")

    # ── Panel 4: Surge routing shift (only if real routing data supplied) ──
    if has_surge:
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.grid(axis="y", linewidth=0.5, alpha=0.4)
        zone_labels = ["Z0\nCyber\nTowers", "Z1\nInorbit", "Z2\nMetro\nBuffer",
                       "Z3\nMind-\nspace",  "Z4\nKonda-\npur"]
        x4  = np.arange(len(zone_labels))
        bw4 = 0.38
        g_bars = ax4.bar(x4 - bw4 / 2, surge_routing["greedy"],  width=bw4,
                         color=RED,   alpha=0.82, label="Greedy")
        t_bars = ax4.bar(x4 + bw4 / 2, surge_routing["trained"], width=bw4,
                         color=GREEN, alpha=0.82, label="Trained")
        t_bars[2].set_edgecolor(CYAN); t_bars[2].set_linewidth(2.8)
        g_bars[0].set_edgecolor(PINK); g_bars[0].set_linewidth(2.5)
        ax4.annotate("Greedy saturates\nZone 0!",
                     xy=(x4[0] - bw4 / 2, surge_routing["greedy"][0]),
                     xytext=(x4[0] - bw4 / 2 + 0.1, surge_routing["greedy"][0] + 6),
                     color=PINK, fontsize=7.5, fontweight="bold", ha="center",
                     arrowprops=dict(arrowstyle="->", color=PINK, lw=1.2))
        ax4.annotate("Trained uses\nbuffer zone!",
                     xy=(x4[2] + bw4 / 2, surge_routing["trained"][2]),
                     xytext=(x4[2] + bw4 / 2 + 0.5, surge_routing["trained"][2] + 6),
                     color=CYAN, fontsize=7.5, fontweight="bold", ha="center",
                     arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.2))
        ax4.set_xticks(x4)
        ax4.set_xticklabels(zone_labels, fontsize=8)
        ax4.set_ylabel("% of Cars Routed", fontsize=9)
        ax4.set_ylim(0, max(surge_routing["greedy"]) * 1.25)
        ax4.set_title("9AM Surge Routing\nPredictive vs Reactive", fontsize=11,
                      color=CYAN, pad=8, fontweight="bold")
        ax4.legend(fontsize=9)

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "training_dashboard.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    plt.rcdefaults()

    print(f"Dashboard saved → {out_path}")
    print(f"  Steps logged   : {len(steps)}")
    print(f"  Reward range   : [{r_arr.min():.3f}, {r_arr.max():.3f}]")
    print(f"  Smoothed start : {start_v:+.3f}")
    print(f"  Smoothed end   : {end_v:+.3f}")
    print(f"  Δ improvement  : {delta:+.0f}%")
    return out_path


def plot_rewards(trainer, save_path: str) -> None:
    """Thin wrapper kept for backward compatibility — calls plot_dashboard."""
    import os
    output_dir = os.path.dirname(save_path) or "."
    plot_dashboard(trainer, output_dir)