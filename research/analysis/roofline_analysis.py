"""
roofline_analysis.py
=============================================================
Roofline model for the embedded RL algorithms on the ESP32-S3.

The Roofline model (Williams, Waterman & Patterson, 2009) places each
compute kernel in the (Operational Intensity, Performance) space
and distinguishes whether it is limited by:
  - Memory (memory-bound): performance ≤ bandwidth × OI
  - Compute (compute-bound): performance ≤ peak_flops

Operational Intensity (OI) = FLOPs / bytes_accessed
Observed performance       = FLOPs / real_time  (effective MFLOPS)

ESP32-S3 hardware parameters (Xtensa LX7 @ 240 MHz):
  peak_flops  ≈ 240 MFLOPS  (1 FP op per clock cycle, no SIMD)
  sram_bw     ≈ 480 MB/s    (estimated: 240 MHz × 2 bytes/cycle for internal DRAM)
  l1_bw       ≈ 960 MB/s    (L1 cache at 1 cycle: 4 bytes × 240 MHz)
  l1_size     ≈ 32 KB       (Xtensa LX7 L1 instruction + data cache)
  ridge_l1    = peak_flops / l1_bw  ≈ 0.25 FLOP/byte  (L1 ridge point)
  ridge_sram  = peak_flops / sram_bw ≈ 0.50 FLOP/byte  (SRAM ridge point)

Analytical OI per agent × environment:
  - Stream-Q: every update() sweeps the WHOLE Q and traces array (n_states×n_actions).
    bytes = 4 × n_states × n_actions × sizeof(float32) [read+write of Q and traces]
    OI   = FLOPs / bytes = 4×n×a / (16×n×a) = 0.25 FLOP/byte (always)
    BUT: if the array fits in L1 (< 32 KB) → it runs at L1 bandwidth.
          if not (CartPole: 81 KB) → it runs at SRAM bandwidth → slower.

  - Stream-AC: accesses W_actor, w_critic and the normalizer (small arrays).
    The bottleneck is expf()/logf() (transcendental functions), not memory.
    OI > 0.35 FLOP/byte but the cost per FLOP is high (expf ≈ 20 cycles on the LX7).

  - TinyDQN: forward + backward over the network weights (small, cache-friendly arrays).
    OI ≈ 0.3–0.4 FLOP/byte, moderate performance.

Usage:
  python analysis/roofline_analysis.py
  python analysis/roofline_analysis.py --no-fig   # only prints the table

Output:
  results/figures/roofline_model.png
  results/raw_summaries/roofline_data.csv
"""

import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
# results/raw no longer exists (reorganized); this script's summary CSV
# lives next to the other summaries in raw_summaries/.
RAW_DIR     = os.path.join(RESULTS_DIR, "raw_summaries")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ESP32-S3 parameters (Xtensa LX7 @ 240 MHz)
# ─────────────────────────────────────────────────────────────────────────────

PEAK_FLOPS_MFLOPS = 240.0   # theoretical MFLOPS (1 FP/cycle, single core)
L1_BW_MB_S        = 960.0   # MB/s (4 bytes × 240 MHz, L1 at 1 cycle)
SRAM_BW_MB_S      = 480.0   # MB/s (estimated for internal DRAM without cache)
L1_SIZE_BYTES      = 32768   # 32 KB L1 (instructions + data, LX7 estimate)

# Ridge points
RIDGE_L1   = PEAK_FLOPS_MFLOPS / L1_BW_MB_S    # FLOP/byte at which L1 saturates compute
RIDGE_SRAM = PEAK_FLOPS_MFLOPS / SRAM_BW_MB_S  # FLOP/byte at which SRAM saturates compute


# ─────────────────────────────────────────────────────────────────────────────
# Analytical computation of OI (Operational Intensity)
# ─────────────────────────────────────────────────────────────────────────────

def compute_oi(agent, env, flops_per_update, memory_bytes):
    """
    Computes the Operational Intensity (FLOP/byte) for each agent × environment.

    For Stream-Q: update() traverses ALL state-actions in Q and traces.
      bytes_read    = n_states × n_actions × 4  (reading Q for the TD update)
      bytes_written = n_states × n_actions × 4  (writing the updated Q)
      + the same for traces → total = 4 × n_states × n_actions × 4 bytes
      But n_states × n_actions × 4 = memory_bytes / 2 (Q and traces take memory_bytes/2 each).
      → bytes_accessed = 4 × (memory_bytes / 2) = 2 × memory_bytes

    For Stream-AC and TinyDQN: update() reads and writes the network weights
      once per update. bytes_accessed ≈ 2 × memory_bytes (read + write).
      NOTE: the Welford normalizer (part of memory_bytes) is also accessed.
    """
    if agent == "stream_q":
        # Q and traces are read and written in full on every update
        bytes_accessed = 2.0 * memory_bytes
    else:
        # Weights are read once (forward) and written once (backward).
        # Factor 2 = read + write.
        bytes_accessed = 2.0 * memory_bytes

    if bytes_accessed < 1e-6:
        return float("nan")

    oi = flops_per_update / bytes_accessed
    return oi


def is_in_l1_cache(memory_bytes):
    """True if the working set fits in the LX7 L1 cache (~32 KB)."""
    return memory_bytes <= L1_SIZE_BYTES


# ─────────────────────────────────────────────────────────────────────────────
# Hardware timing data (from the paper, hardware table)
# The times are mean µs/step (act + update) measured on the ESP32-S3.
# ─────────────────────────────────────────────────────────────────────────────

# 2026-10-19: the times are no longer hard-coded. They are computed from
# results/raw_hardware/*.csv (seeds 0-9) with the same methodology as the
# tab:hardware table of the paper: per seed, mean of wall_time_s/steps over all its
# episodes; then the mean across the 10 seeds. Previous hand-written values
# (25.1, 13.4, 32.8, 22.7, 597.1, 49.6) are in roofline_analysis.py.bak_pre_v19 (not included).
def _load_hardware_data():
    import pandas as _pd
    hw_dir = os.path.join(RESULTS_DIR, "raw_hardware")
    data = []
    for env in ("gridworld", "cartpole"):
        for agent in ("stream_ac", "stream_q", "tiny_dqn"):
            seed_means = []
            for s in range(10):
                d = _pd.read_csv(os.path.join(hw_dir, f"{agent}_{env}_esp32_seed{s}.csv"))
                seed_means.append(float((d["wall_time_s"] * 1e6 / d["steps"]).mean()))
            data.append((agent, env, int(d["flops_per_update"].iloc[0]),
                         int(d["memory_bytes"].iloc[0]), sum(seed_means) / len(seed_means)))
    return data

HARDWARE_DATA = _load_hardware_data()


def compute_roofline_points(data):
    """Computes OI and effective MFLOPS for each Roofline point."""
    points = []
    for agent, env, flops, mem_bytes, us_per_step in data:
        oi         = compute_oi(agent, env, flops, mem_bytes)
        mflops_eff = flops / us_per_step  # Effective MFLOPS (FLOPs / µs = MFLOPS)
        in_l1      = is_in_l1_cache(mem_bytes)
        bottleneck = infer_bottleneck(agent, env, oi, in_l1)
        points.append({
            "agent": agent, "env": env,
            "flops": flops, "memory_bytes": mem_bytes,
            "us_per_step": us_per_step,
            "oi": oi,
            "mflops_eff": mflops_eff,
            "in_l1": in_l1,
            "bottleneck": bottleneck,
        })
    return points


def infer_bottleneck(agent, env, oi, in_l1):
    """Infers the dominant bottleneck for each point."""
    if agent == "stream_q":
        if not in_l1:
            return "Cache misses (SRAM)"
        return "Memory-bound (L1)"
    if agent == "stream_ac":
        return "Transcendental (expf/logf)"
    # TinyDQN
    return "ReLU + matmul"


# ─────────────────────────────────────────────────────────────────────────────
# Roofline figure
# ─────────────────────────────────────────────────────────────────────────────

AGENT_STYLES = {
    "stream_ac": dict(marker="^", color="#e05c3a", markersize=11, zorder=5),
    "stream_q":  dict(marker="s", color="#3a6eb5", markersize=11, zorder=5),
    "tiny_dqn":  dict(marker="D", color="#4daa6c", markersize=10, zorder=5),
}
ENV_FACECOLORS = {"gridworld": "white", "cartpole": None}  # None = filled


def build_roofline_figure(points):
    fig, ax = plt.subplots(figsize=(9, 6))

    # ── Roofline ceilings ────────────────────────────────────────────────
    oi_range = np.logspace(-3, 2, 500)  # 0.001 … 100 FLOP/byte

    # Compute ceiling (horizontal)
    compute_ceil = np.full_like(oi_range, PEAK_FLOPS_MFLOPS)

    # L1 memory ceiling (rising diagonal)
    l1_ceil   = np.minimum(L1_BW_MB_S   * oi_range, PEAK_FLOPS_MFLOPS)
    sram_ceil = np.minimum(SRAM_BW_MB_S * oi_range, PEAK_FLOPS_MFLOPS)

    ax.fill_between(oi_range, sram_ceil, compute_ceil,
                    alpha=0.08, color="#3a6eb5", label="_nolegend_")
    ax.fill_between(oi_range, l1_ceil, sram_ceil,
                    alpha=0.06, color="#e05c3a", label="_nolegend_")

    ax.loglog(oi_range, l1_ceil,   "--", color="#555", lw=1.4,
              label=f"L1 roof ({L1_BW_MB_S:.0f} MB/s)")
    ax.loglog(oi_range, sram_ceil, ":",  color="#888", lw=1.4,
              label=f"SRAM roof ({SRAM_BW_MB_S:.0f} MB/s)")
    ax.axhline(PEAK_FLOPS_MFLOPS, color="#222", lw=1.4, ls="-",
               label=f"Peak compute ({PEAK_FLOPS_MFLOPS:.0f} MFLOPS)")

    # Vertical lines at the ridge points
    ax.axvline(RIDGE_L1,   color="#555", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(RIDGE_SRAM, color="#888", lw=0.8, ls=":",  alpha=0.5)
    ax.text(RIDGE_L1,   ax.get_ylim()[0] if False else 1.2,
            f"Ref. L1\n{RIDGE_L1:.2f}", fontsize=7, color="#555",
            ha="center", va="bottom")
    ax.text(RIDGE_SRAM, 1.8,
            f"Ref. SRAM\n{RIDGE_SRAM:.2f}", fontsize=7, color="#888",
            ha="center", va="bottom")

    # ── Roofline points ───────────────────────────────────────────────
    legend_agents = {}
    for pt in points:
        style = dict(AGENT_STYLES[pt["agent"]])
        if pt["env"] == "gridworld":
            style["markerfacecolor"] = "white"  # hollow = GridWorld
            style["markeredgewidth"] = 2.0
        else:
            style["markeredgecolor"] = "white"
            style["markeredgewidth"] = 0.8

        ax.loglog(pt["oi"], pt["mflops_eff"],
                  linestyle="none",
                  **{k: v for k, v in style.items()
                     if k in ("marker","color","markersize","zorder",
                               "markerfacecolor","markeredgecolor","markeredgewidth")})

        # Label
        # Explicit offsets (in points) so that each label sits
        # next to its own marker; GW/CP as in the paper.
        env_tag = {"gridworld": "GW", "cartpole": "CP"}[pt["env"]]
        label_txt = f"{pt['agent']} ({env_tag})"
        LABEL_OFFSETS = {
            ("stream_ac", "gridworld"): (14, 10),  ("stream_ac", "cartpole"): (-78, -16),
            ("tiny_dqn",  "gridworld"): (12, 8),   ("tiny_dqn",  "cartpole"): (12, -14),
            ("stream_q",  "gridworld"): (-82, 4),  ("stream_q",  "cartpole"): (-80, 12),
        }
        ax.annotate(
            label_txt,
            xy=(pt["oi"], pt["mflops_eff"]),
            xytext=LABEL_OFFSETS[(pt["agent"], pt["env"])], textcoords="offset points",
            fontsize=7.5, color=AGENT_STYLES[pt["agent"]]["color"],
            arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.7),
        )

        # Legend per agent (only once)
        if pt["agent"] not in legend_agents:
            legend_agents[pt["agent"]] = plt.Line2D(
                [0], [0], marker=AGENT_STYLES[pt["agent"]]["marker"],
                color="w", markerfacecolor=AGENT_STYLES[pt["agent"]]["color"],
                markersize=9, label=pt["agent"].replace("_", "-"),
            )

    # Fill legend: GW (hollow) vs CP (solid)
    gw_patch = plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
                           markeredgecolor="#555", markeredgewidth=1.8,
                           markersize=9, label="GridWorld (outline)")
    cp_patch = plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#555",
                           markeredgecolor="white", markeredgewidth=0.8,
                           markersize=9, label="CartPole (solid)")

    handles_roof = [
        plt.Line2D([0],[0], ls="--", color="#555", lw=1.4,
                   label=f"L1 reference ({L1_BW_MB_S:.0f} MB/s)"),
        plt.Line2D([0],[0], ls=":",  color="#888", lw=1.4,
                   label=f"SRAM reference ({SRAM_BW_MB_S:.0f} MB/s)"),
        plt.Line2D([0],[0], ls="-",  color="#222", lw=1.4,
                   label=f"Compute reference ({PEAK_FLOPS_MFLOPS:.0f} MFLOPS)"),
    ]

    ax.legend(
        handles=list(legend_agents.values()) + [gw_patch, cp_patch] + handles_roof,
        fontsize=8, loc="upper left", framealpha=0.9, ncol=2,
    )

    ax.set_xlabel("FLOPs per estimated state-access byte (proxy)", fontsize=11)
    ax.set_ylabel("Effective performance (MFLOPS)", fontsize=11)
    ax.set_title(
        "Roofline-Inspired Analytical View — TinyRL on ESP32-S3 (Xtensa LX7 @ 240 MHz)",
        fontsize=10, pad=12,
    )
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_xlim(1e-3, 10)
    ax.set_ylim(0.5, PEAK_FLOPS_MFLOPS * 3)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "roofline_model.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Roofline model for TinyRL on the ESP32-S3.")
    parser.add_argument("--no-fig", action="store_true", help="Do not generate the figure")
    args = parser.parse_args()

    print("=== Roofline Model — ESP32-S3 (Xtensa LX7 @ 240 MHz) ===\n")
    print(f"Hardware parameters:")
    print(f"  Peak FLOPS     = {PEAK_FLOPS_MFLOPS:.0f} MFLOPS")
    print(f"  L1 BW          = {L1_BW_MB_S:.0f} MB/s  (ridge = {RIDGE_L1:.2f} FLOP/byte)")
    print(f"  SRAM BW        = {SRAM_BW_MB_S:.0f} MB/s  (ridge = {RIDGE_SRAM:.2f} FLOP/byte)")
    print(f"  L1 cache size  = {L1_SIZE_BYTES//1024} KB\n")

    points = compute_roofline_points(HARDWARE_DATA)

    # ── Results table ──
    print(f"{'Agent':12s} {'Env':10s} {'FLOPs':8s} {'Mem(B)':8s} "
          f"{'µs/step':8s} {'OI(F/B)':8s} {'MFLOPS':8s} {'in_L1':6s} {'Bottleneck'}")
    print("-" * 90)
    for p in points:
        l1_str = "✓" if p["in_l1"] else "✗"
        print(f"{p['agent']:12s} {p['env']:10s} {p['flops']:8d} {p['memory_bytes']:8d} "
              f"{p['us_per_step']:8.1f} {p['oi']:8.4f} {p['mflops_eff']:8.2f} "
              f"{l1_str:6s} {p['bottleneck']}")

    # ── CSV ──
    csv_path = os.path.join(RAW_DIR, "roofline_data.csv")
    fields = ["agent","env","flops","memory_bytes","us_per_step","oi","mflops_eff","in_l1","bottleneck"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(points)
    print(f"\nCSV saved: {csv_path}")

    # ── Key insights ──
    print("\n=== Roofline model insights ===")
    for p in points:
        if p["agent"] == "stream_q":
            if p["in_l1"]:
                print(f"  Stream-Q {p['env']:10s}: working set {p['memory_bytes']}B ≤ L1 ({L1_SIZE_BYTES//1024}KB) "
                      f"→ cache-friendly, {p['mflops_eff']:.1f} effective MFLOPS")
            else:
                ratio = p["memory_bytes"] / L1_SIZE_BYTES
                print(f"  Stream-Q {p['env']:10s}: working set {p['memory_bytes']//1024}KB = "
                      f"{ratio:.1f}× L1 → cache misses dominate, "
                      f"only {p['mflops_eff']:.1f} effective MFLOPS "
                      f"(vs {L1_BW_MB_S * p['oi']:.1f} theoretical MFLOPS in L1)")
        elif p["agent"] == "stream_ac":
            expected_mflops = min(L1_BW_MB_S * p["oi"], PEAK_FLOPS_MFLOPS)
            print(f"  Stream-AC {p['env']:9s}: OI={p['oi']:.3f} → expected ≥{expected_mflops:.1f} MFLOPS, "
                  f"actual={p['mflops_eff']:.1f} → bottleneck: expf()/logf() "
                  f"(~{expected_mflops/p['mflops_eff']:.1f}× slower than expected from OI)")

    print(f"\n  Ridge point L1:   OI ≥ {RIDGE_L1:.2f} FLOP/byte → compute-bound")
    print(f"  SRAM ridge point: OI ≥ {RIDGE_SRAM:.2f} FLOP/byte → compute-bound (no cache)")
    print(f"  All agents have OI < {RIDGE_L1:.2f} → memory-bound at L1,")
    print(f"  EXCEPT Stream-AC, which is compute-bound because of transcendental functions.")

    if not args.no_fig:
        out_path = build_roofline_figure(points)
        print(f"\nFigure saved: {out_path}")


if __name__ == "__main__":
    main()
