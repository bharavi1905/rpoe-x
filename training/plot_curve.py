from __future__ import annotations

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# PART A — Load and smooth curves
# ---------------------------------------------------------------------------

def load_curves(path: str = "training/curves.json") -> dict:
    with open(path) as f:
        return json.load(f)


def smooth(values: list[float], window: int = 20) -> np.ndarray:
    """Simple moving average smoothing."""
    arr = np.array(values, dtype=np.float32)
    kernel = np.ones(window) / window
    # Pad edges to avoid boundary artifacts
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


# ---------------------------------------------------------------------------
# PART B — Plot function
# ---------------------------------------------------------------------------

def plot_curves(
    curves: dict,
    save_path: str = "training/training_curve.png",
):
    rl_curves     = np.array(curves["rl"],     dtype=np.float32)   # (3, N)
    greedy_curves = np.array(curves["greedy"], dtype=np.float32)   # (3, N)
    n_episodes    = curves["n_episodes"]
    episodes      = np.arange(1, n_episodes + 1)

    # Smooth each seed's curve
    rl_smooth     = np.array([smooth(rl_curves[i])     for i in range(3)])
    greedy_smooth = np.array([smooth(greedy_curves[i]) for i in range(3)])

    # Mean and std across seeds
    rl_mean     = rl_smooth.mean(axis=0)
    rl_std      = rl_smooth.std(axis=0)
    greedy_mean = greedy_smooth.mean(axis=0)

    # Final values for annotation
    rl_final     = float(rl_mean[-50:].mean())
    greedy_final = float(greedy_mean[-50:].mean())
    improvement  = (rl_final - greedy_final) / (abs(greedy_final) + 1e-8) * 100

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))

    # Greedy baseline — flat dashed line
    ax.axhline(
        y=greedy_final,
        color="#e74c3c",
        linewidth=1.8,
        linestyle="--",
        label=f"Greedy baseline  ({greedy_final:.2f})",
        zorder=2,
    )

    # RL agent — solid line with shaded ±1 std band
    ax.fill_between(
        episodes,
        rl_mean - rl_std,
        rl_mean + rl_std,
        alpha=0.18,
        color="#2ecc71",
        zorder=1,
    )
    ax.plot(
        episodes,
        rl_mean,
        color="#27ae60",
        linewidth=2.2,
        label=f"RL agent (REINFORCE, 3 seeds)  ({rl_final:.2f})",
        zorder=3,
    )

    # Annotation: improvement arrow at final episode
    ax.annotate(
        f"{improvement:+.1f}% vs greedy",
        xy=(n_episodes, rl_final),
        xytext=(n_episodes * 0.72, rl_final + (rl_final - greedy_final) * 0.4),
        fontsize=11,
        color="#27ae60",
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color="#27ae60",
            lw=1.5,
        ),
    )

    # Styling
    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Total Episode Reward", fontsize=13)
    ax.set_title(
        "RPOE-X Training Curve — REINFORCE vs Greedy Baseline\n"
        "HITEC City Multi-Zone Rotary Parking (5 zones, 20 wheels, 240 slots)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(1, n_episodes)

    # Subtle zone labels on right Y axis
    ax2 = ax.twinx()
    ax2.set_ylabel("← lower = more wait time", fontsize=9, color="gray")
    ax2.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved to {save_path}")
    print(f"RL final avg reward:     {rl_final:.4f}")
    print(f"Greedy final avg reward: {greedy_final:.4f}")
    print(f"Improvement:             {improvement:+.1f}%")
    return improvement


# ---------------------------------------------------------------------------
# PART C — Main block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    curves_path = "training/curves.json"
    save_path   = "training/training_curve.png"

    if not os.path.exists(curves_path):
        print(f"ERROR: {curves_path} not found. Run training/train.py first.")
        sys.exit(1)

    curves = load_curves(curves_path)
    improvement = plot_curves(curves, save_path=save_path)

    if improvement > 0:
        print("RL agent beats greedy — training curve is valid for demo.")
    else:
        print("WARNING: RL agent does not yet beat greedy — consider more episodes or tuning lr.")
