#pragma once

// ── Seed ─────────────────────────────────────────────────────────────────────
// Change this value to run different seeds. It affects randomSeed() and the CSV.
#define SEED 9

// ── Environment ───────────────────────────────────────────────────────────────────
#define GRID_SIZE         5
#define MAX_EPISODE_STEPS 100
#define N_EPISODES        300

// ── TinyDQN hyperparameters (universal, config.yaml) ────────────────────────
// Corrected 2026-07-28: lr was 0.01 and epsilon_start 0.1 (old values, they did not
// matched config.yaml). Verified bit-exact that lr=0.002/eps0=0.15
// reproduce raw_final/tiny_dqn_gridworld_0.csv (300/300 identical ep).
#define DQN_LR            0.002f
#define DQN_GAMMA         0.99f
#define DQN_EPSILON_START 0.15f
#define DQN_EPSILON_DECAY 0.999f

// Network: n_features=2, hidden=12, n_actions=4
#define DQN_NF   2    // input features
#define DQN_H    12   // hidden neurons
#define DQN_NA   4    // output actions

// Mini-buffer
#define DQN_BUF  32   // replay buffer capacity

// Clips (nn_manual.py: GRAD_CLIP=5, WEIGHT_CLIP=10)
#define DQN_GRAD_CLIP   5.0f
#define DQN_WEIGHT_CLIP 10.0f
