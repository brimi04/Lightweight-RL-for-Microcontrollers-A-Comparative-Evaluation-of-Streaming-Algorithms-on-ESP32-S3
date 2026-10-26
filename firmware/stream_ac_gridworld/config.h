#pragma once

// ── Seed ─────────────────────────────────────────────────────────────────────
// Change this value to run different seeds. It affects randomSeed() and the CSV.
#define SEED 9

// ── Environment ───────────────────────────────────────────────────────────────────
#define GRID_SIZE         5
#define MAX_EPISODE_STEPS 100
#define N_EPISODES        300

// ── Selected agent ───────────────────────────────────────────────────────
//   0 = Stream-Q   (tabular, Q-learning with traces)
//   1 = Stream-AC  (linear actor-critic, softmax policy)  <-- ACTIVE
#define AGENT_SELECT 1

// ── Stream-Q hyperparameters ──────────────────────────────────────────────────
#define SQ_ALPHA         0.1f
#define SQ_GAMMA         0.99f
#define SQ_LAMBDA        0.9f
#define SQ_EPSILON_START 0.1f
#define SQ_EPSILON_DECAY 0.999f

// ── Stream-AC hyperparameters ─────────────────────────────────────────────────
#define SAC_ALPHA_ACTOR  0.01f
#define SAC_ALPHA_CRITIC 0.02f
#define SAC_GAMMA        0.99f
// entropy_coef corrected 2026-07-28: it was 0.3 (pre-fix, undocumented value,
// never updated after MEMORIA_PROYECTO.md sec. 17.6). Verified bit-exact
// that 0.05 is the value that reproduces raw_final/stream_ac_gridworld_*.csv
// (episode-by-episode Python reproduction, 300/300 identical ep, 2026-07-28).
#define SAC_ENTROPY_COEF 0.05f
#define SAC_WEIGHT_CLIP  5.0f
// Gradient clip before applying the update (Bug 7 — synchronized with Python)
#define SAC_GRAD_CLIP    5.0f
