from __future__ import annotations

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.env import RPOEXEnv
from models import OrchestratorAction, OrchestratorObs
from tasks.graders import greedy_orchestrator, greedy_zone, run_task2

# ---------------------------------------------------------------------------
# PART A — Policy (linear softmax over orchestrator obs)
# ---------------------------------------------------------------------------

class OrchestratorPolicy:
    """
    Simple linear policy: obs_vector (dim=15) -> 5 logits -> softmax -> zone_id
    Obs vector = zone_occupancy(5) + zone_queue_lengths(5) + arrival_rate_ema(5)
    """
    def __init__(self, obs_dim: int = 15, n_zones: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (n_zones, obs_dim))  # (5, 15)
        self.b = np.zeros(n_zones)

    def _obs_to_vec(self, obs: OrchestratorObs) -> np.ndarray:
        return np.array(
            obs.zone_occupancy +
            [float(q) for q in obs.zone_queue_lengths] +
            obs.arrival_rate_ema,
            dtype=np.float32,
        )

    def forward(self, obs: OrchestratorObs) -> tuple[int, float]:
        """Returns (zone_id, log_prob)"""
        x = self._obs_to_vec(obs)
        logits = self.W @ x + self.b
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits) / np.exp(logits).sum()
        zone_id = int(np.random.choice(5, p=probs))
        log_prob = float(np.log(probs[zone_id] + 1e-8))
        return zone_id, log_prob

    def update(self, gradients: list[np.ndarray], lr: float = 1e-3):
        """Apply accumulated gradients to W and b."""
        for dW, db in gradients:
            self.W += lr * dW
            self.b += lr * db


# ---------------------------------------------------------------------------
# PART B — REINFORCE episode runner
# ---------------------------------------------------------------------------

def run_reinforce_episode(
    policy: OrchestratorPolicy,
    seed: int,
    max_steps: int = 400,
    gamma: float = 0.99,
) -> tuple[float, list]:
    """
    Run one episode with the policy.
    Returns (total_reward, list of (dW, db) gradient tuples).
    """
    env = RPOEXEnv(seed=seed, max_steps=max_steps)
    obs = env.reset(seed=seed)

    log_probs = []
    rewards = []

    while not obs.done:
        zone_id, log_prob = policy.forward(obs)
        action = OrchestratorAction(action="route_to_zone", zone_id=zone_id)
        obs = env.step(action)
        log_probs.append((log_prob, policy._obs_to_vec(obs), zone_id))
        rewards.append(obs.reward)

    # Compute discounted returns
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns, dtype=np.float32)

    # Normalize returns
    if returns.std() > 1e-6:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Compute gradients (REINFORCE: grad = log_prob * return)
    gradients = []
    for (lp, x_vec, z_id), G_t in zip(log_probs, returns):
        logits = policy.W @ x_vec + policy.b
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        d_logits = -probs.copy()
        d_logits[z_id] += 1.0
        dW = np.outer(d_logits, x_vec) * G_t
        db = d_logits * G_t
        gradients.append((dW, db))

    total_reward = float(sum(rewards))
    return total_reward, gradients


# ---------------------------------------------------------------------------
# PART C — Greedy baseline episode runner
# ---------------------------------------------------------------------------

def run_greedy_episode(seed: int, max_steps: int = 400) -> float:
    """Run one episode with greedy agent. Returns total reward."""
    env = RPOEXEnv(seed=seed, max_steps=max_steps)
    obs = env.reset(seed=seed)
    total_reward = 0.0
    while not obs.done:
        action = greedy_orchestrator(obs)
        obs = env.step(action)
        total_reward += obs.reward
    return float(total_reward)


# ---------------------------------------------------------------------------
# PART D — Main training loop
# ---------------------------------------------------------------------------

def train(
    n_episodes: int = 400,
    seeds: list[int] = [0, 1, 2],
    max_steps: int = 400,
    lr: float = 1e-3,
    gamma: float = 0.99,
    save_path: str = "training/curves.json",
):
    """
    Run REINFORCE training across 3 seeds.
    Save RL and greedy reward histories to curves.json.
    """
    print(f"Training: {n_episodes} episodes, {len(seeds)} seeds, lr={lr}")

    all_rl_curves = []
    all_greedy_curves = []

    for seed in seeds:
        print(f"\n── Seed {seed} ──────────────────────────────")
        np.random.seed(seed)
        policy = OrchestratorPolicy(seed=seed)

        rl_rewards = []
        greedy_rewards = []

        for ep in range(n_episodes):
            ep_seed = seed * 10000 + ep

            # RL episode
            total_reward, gradients = run_reinforce_episode(
                policy, seed=ep_seed, max_steps=max_steps, gamma=gamma
            )
            policy.update(gradients, lr=lr)
            rl_rewards.append(total_reward)

            # Greedy episode (same seed for fair comparison)
            g_reward = run_greedy_episode(seed=ep_seed, max_steps=max_steps)
            greedy_rewards.append(g_reward)

            if (ep + 1) % 50 == 0:
                rl_avg = np.mean(rl_rewards[-50:])
                gr_avg = np.mean(greedy_rewards[-50:])
                print(f"  ep={ep+1:4d} rl_avg={rl_avg:.4f} greedy_avg={gr_avg:.4f}")

        all_rl_curves.append(rl_rewards)
        all_greedy_curves.append(greedy_rewards)

    # Save curves
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    curves = {
        "rl":         all_rl_curves,
        "greedy":     all_greedy_curves,
        "seeds":      seeds,
        "n_episodes": n_episodes,
        "config":     {"lr": lr, "gamma": gamma, "max_steps": max_steps},
    }
    with open(save_path, "w") as f:
        json.dump(curves, f, indent=2)

    print(f"\nCurves saved to {save_path}")

    # Summary
    rl_final = np.mean([c[-50:] for c in all_rl_curves])
    gr_final = np.mean([c[-50:] for c in all_greedy_curves])
    print(f"Final RL avg reward:     {rl_final:.4f}")
    print(f"Final greedy avg reward: {gr_final:.4f}")
    improvement = (rl_final - gr_final) / (abs(gr_final) + 1e-8) * 100
    print(f"Improvement over greedy: {improvement:+.1f}%")


if __name__ == "__main__":
    train(
        n_episodes=400,
        seeds=[0, 1, 2],
        max_steps=400,
        lr=1e-3,
        gamma=0.99,
        save_path="training/curves.json",
    )
