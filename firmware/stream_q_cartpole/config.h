#pragma once

// ── Seed ──────────────────────────────────────────────────────────────────────
#define SEED 9


// ── Environment ───────────────────────────────────────────────────────────────────
#define MAX_EPISODE_STEPS 500
#define N_EPISODES        1500

// ── CartPole discretizer (wrappers.py: CARTPOLE_DISCRETIZER) ────────────────
// bins = [6, 6, 12, 12]  →  n_states = 6*6*12*12 = 5184
// Q-table: 5184 * 2 * 4 B = 41 472 B  (+traces = 82 944 B total)
#define SQ_BINS_0   6
#define SQ_BINS_1   6
#define SQ_BINS_2  12
#define SQ_BINS_3  12
#define SQ_N_STATES 5184   // 6*6*12*12
#define SQ_N_ACTIONS   2

#define SQ_LOW_0  -2.4f
#define SQ_LOW_1  -3.0f
#define SQ_LOW_2  -0.21f
#define SQ_LOW_3  -3.0f

#define SQ_HIGH_0  2.4f
#define SQ_HIGH_1  3.0f
#define SQ_HIGH_2  0.21f
#define SQ_HIGH_3  3.0f

// ── Stream-Q hyperparameters (universal, config.yaml) ───────────────────────
// Corrected 2026-07-28: it had the old values from tuned_params.json
// (alpha=0.2, lambda=0.8, eps0=0.15, epsd=0.998), already identified in
// MEMORIA_PROYECTO.md sec. 17.7 as not representative of the official data.
// Verified bit-exact that the universal set from config.yaml reproduces
// raw_final/stream_q_cartpole_0.csv (1500/1500 identical ep, 2026-07-28) --
// Stream-Q uses the same universal set in both environments, without per-env tuning.
#define SQ_ALPHA         0.1f
#define SQ_GAMMA         0.99f
#define SQ_LAMBDA        0.9f
#define SQ_EPSILON_START 0.1f
#define SQ_EPSILON_DECAY 0.999f
