#pragma once

// ── Seed ─────────────────────────────────────────────────────────────────────
// Change this value to run different seeds. It affects randomSeed() and the CSV.
#define SEED 9

// ── Environment ───────────────────────────────────────────────────────────────────
#define GRID_SIZE         5
#define MAX_EPISODE_STEPS 100
#define N_EPISODES        300

// ── Stream-Q hyperparameters ──────────────────────────────────────────────────
#define SQ_ALPHA         0.1f
#define SQ_GAMMA         0.99f
#define SQ_LAMBDA        0.9f
#define SQ_EPSILON_START 0.1f
#define SQ_EPSILON_DECAY 0.999f
