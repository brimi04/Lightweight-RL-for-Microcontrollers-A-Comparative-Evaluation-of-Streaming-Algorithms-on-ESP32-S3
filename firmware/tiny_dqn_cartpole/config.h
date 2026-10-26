#pragma once

// ── Seed ──────────────────────────────────────────────────────────────────────
#define SEED 8

// ── Environment ───────────────────────────────────────────────────────────────────
#define MAX_EPISODE_STEPS 500
#define N_EPISODES        1500

// ── TinyDQN hyperparameters (universal, config.yaml) ────────────────────────
// Network: 4 → 12 → 2
#define DQN_NF   4    // features: x, x_dot, theta, theta_dot
#define DQN_H    12   // hidden neurons
#define DQN_NA   2    // actions

// lr corrected 2026-07-28: it was 0.001 (old value from tuned_params.json, already
// flagged in MEMORIA_PROYECTO.md sec. 17.7 as not representative). Verified
// bit-exact that lr=0.002 (same as GridWorld -- TinyDQN is not tuned per
// environment) reproduces raw_final/tiny_dqn_cartpole_0.csv (1500/1500 identical ep).
#define DQN_LR            0.002f
#define DQN_GAMMA         0.99f
#define DQN_EPSILON_START 0.15f
#define DQN_EPSILON_DECAY 0.999f

// Mini replay buffer
#define DQN_BUF  32

// Clips (synchronized with nn_manual.py: GRAD_CLIP=5, WEIGHT_CLIP=10)
#define DQN_GRAD_CLIP   5.0f
#define DQN_WEIGHT_CLIP 10.0f
