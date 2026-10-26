"""
flops_vs_timing_fig.py
=============================================================
Figure showing the dissociation between analytical FLOPs and real throughput,
used in the paper (Section "Analytical FLOPs Do Not Predict
Real-World Throughput").

UPDATED 2026-09-14: rewritten to compute the mean and SD
directly from research/results/raw_hardware/*.csv (10 official
seeds, 0-9), instead of the hard-coded HARDWARE_DATA list
of the previous version. With this the figure is reproducible
from the raw data and uses exactly the same methodology
as the hardware table (tab:hardware) of the paper: for each seed,
us_per_step = wall_time_s*1e6/steps is averaged over all its episodes,
and then the mean +/- SD across the 10 seeds is reported. The FLOPs
per update are taken from the flops_per_update column of the CSV itself
(not hard-coded). Verified: the 6 mean values match
those already used in the hardware table (25.03, 14.29, 34.09, 22.67, 596.3,
49.70), within that table's rounding.

The previous HARDWARE_DATA (recovered by hand after the original
script that generated it was lost; see the previous version of the
file, flops_vs_timing_fig.py.bak_pre_reproducible, not included in this repository)
is left commented out below only as a historical reference:

# HARDWARE_DATA_LEGACY = [
#     ("stream_ac", "gridworld",    84,     25.1),
#     ("stream_q",  "gridworld",   400,     13.4),
#     ("tiny_dqn",  "gridworld",  1152,     32.8),
#     ("stream_ac", "cartpole",     80,     22.7),
#     ("stream_q",  "cartpole",  41472,    597.1),
#     ("tiny_dqn",  "cartpole",   1152,     49.6),
# ]

Usage:
  python analysis/flops_vs_timing_fig.py

Output:
  results/figures/flops_vs_timing.png
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
HW_DIR      = os.path.join(RESULTS_DIR, "raw_hardware")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

_ORDER = [
    ("stream_ac", "gridworld"),
    ("stream_q",  "gridworld"),
    ("tiny_dqn",  "gridworld"),
    ("stream_ac", "cartpole"),
    ("stream_q",  "cartpole"),
    ("tiny_dqn",  "cartpole"),
]

AGENT_COLORS = {
    "stream_ac": "#1f77d0",
    "stream_q":  "#3aa155",
    "tiny_dqn":  "#e8622c",
}
ENV_MARKERS = {"gridworld": "o", "cartpole": "s"}
NICE_NAME = {"stream_ac": "Stream-AC", "stream_q": "Stream-Q", "tiny_dqn": "TinyDQN"}


def load_hardware_timing():
    """
    Loads research/results/raw_hardware/*.csv (10 official seeds, 0-9),
    computes us/step per episode, averages per seed, and then computes the
    mean +/- SD across the 10 seeds -- same methodology as the hardware table.
    Returns a dict {(agent, env): (flops, mean_us, sd_us)}.
    """
    rows = []
    for path in glob.glob(os.path.join(HW_DIR, "*.csv")):
        fname = os.path.basename(path)
        m = re.search(r"seed(\d+)", fname)
        if m and int(m.group(1)) >= 10:
            continue
        df = pd.read_csv(path)
        if df.empty or "agent" not in df.columns:
            continue
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df["us_per_step"] = df["wall_time_s"] * 1e6 / df["steps"]

    out = {}
    for (agent, env), g in df.groupby(["agent", "env"]):
        seed_means = g.groupby("seed")["us_per_step"].mean()
        flops = int(g["flops_per_update"].iloc[0])
        out[(agent, env)] = (flops, float(seed_means.mean()), float(seed_means.std(ddof=1)))
    return out


def build_figure(data_dict):
    hardware_data = [(a, e, data_dict[(a, e)][0], data_dict[(a, e)][1], data_dict[(a, e)][2])
                      for a, e in _ORDER]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    flops_all = [p[2] for p in hardware_data]

    # Proportionality line, calibrated on Stream-AC (GridWorld), recomputed value
    ref_flops, ref_us, _ = data_dict[("stream_ac", "gridworld")]
    k = ref_us / ref_flops
    x_line = np.logspace(np.log10(min(flops_all) * 0.5),
                          np.log10(max(flops_all) * 2), 200)
    y_line = k * x_line

    ax.fill_between(x_line, y_line, y_line * 50,
                     where=(y_line * 50 > 0), alpha=0.06, color="#c0392b")
    ax.fill_between(x_line, y_line / 50, y_line,
                     alpha=0.06, color="#2e8b57")

    ax.loglog(x_line, y_line, "--", color="#999", lw=1.3,
              label="Proportional to FLOPs", zorder=2)

    ax.text(x_line[15], y_line[15] * 8, "Slower than\nlinear reference",
            fontsize=9, color="#c0392b", style="italic", ha="left")
    ax.text(x_line[120], y_line[120] / 12, "Faster than\nlinear reference",
            fontsize=9, color="#2e8b57", style="italic", ha="left")

    legend_agents = {}
    for agent, env, flops, us_mean, us_sd in hardware_data:
        color  = AGENT_COLORS[agent]
        marker = ENV_MARKERS[env]
        ax.errorbar(flops, us_mean, yerr=us_sd, fmt=marker, color=color,
                     markersize=13, markeredgecolor="white", markeredgewidth=1.0,
                     ecolor=color, elinewidth=1.4, capsize=4, zorder=5)
        if agent not in legend_agents:
            legend_agents[agent] = plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color,
                markersize=10, label=NICE_NAME[agent],
            )

    env_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                   markersize=10, label="Environment: GridWorld"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#888",
                   markersize=10, label="Environment: CartPole"),
        plt.Line2D([0], [0], ls="--", color="#999", lw=1.3,
                   label="Proportional to FLOPs"),
    ]

    leg1 = ax.legend(handles=list(legend_agents.values()), title="Agent",
                      fontsize=9, title_fontsize=9, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=env_handles, title="Environment / reference",
              fontsize=8.5, title_fontsize=8.5, loc="lower right")

    ax.set_xlabel("Analytical FLOPs per update (log)", fontsize=11)
    ax.set_ylabel("Measured execution time (µs/step, log)", fontsize=11)
    ax.set_title(
        "Analytical FLOPs vs. measured execution time on ESP32-S3 @ 240 MHz\n"
        "GridWorld and CartPole — 3 lightweight RL agents (mean ± SD, 10 seeds)",
        fontsize=11, pad=12,
    )
    ax.grid(True, which="both", ls="--", alpha=0.3)
    # Presentation only (2026-10-25): fix the x range to the plotted reference band.
    # The error bars made matplotlib autoscale the x axis down to ~1 FLOP, leaving an
    # empty region. Data, markers and reference line are unchanged.
    ax.set_xlim(x_line[0] / 1.3, x_line[-1] * 1.3)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "flops_vs_timing.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path, hardware_data


def main():
    data_dict = load_hardware_timing()
    out_path, hardware_data = build_figure(data_dict)
    print(f"Figura guardada: {out_path}")
    print("\nValores usados (agent, env, flops, us_mean, us_sd):")
    for agent, env, flops, us_mean, us_sd in hardware_data:
        print(f"  {agent:10s} {env:10s} flops={flops:6d} us_mean={us_mean:8.2f} us_sd={us_sd:6.2f}")


if __name__ == "__main__":
    main()
