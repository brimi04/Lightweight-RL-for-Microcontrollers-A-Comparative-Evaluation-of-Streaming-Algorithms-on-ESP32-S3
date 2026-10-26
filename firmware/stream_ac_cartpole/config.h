#pragma once

// ── Seed ──────────────────────────────────────────────────────────────────────
// Change this value to run different seeds. It affects randomSeed() and the CSV.
#define SEED 9

// ── Environment ───────────────────────────────────────────────────────────────────
#define MAX_EPISODE_STEPS 500
#define N_EPISODES        1500

// ── Stream-AC hyperparameters (CartPole, tuned_params.json) ───────────────────
// Problem dimensions
#define SAC_NF   4    // input features (x, x_dot, theta, theta_dot)
#define SAC_NA   2    // actions (left, right)

#define SAC_ALPHA_ACTOR  0.005f
#define SAC_ALPHA_CRITIC 0.01f
#define SAC_GAMMA        0.99f
// entropy_coef corrected 2026-07-28: it was 0.5 (pre-fix, undocumented value,
// never updated after MEMORIA_PROYECTO.md sec. 17.7). Verified bit-exact
// that 0.05 is the value that reproduces raw_final/stream_ac_cartpole_*.csv
// (episode-by-episode Python reproduction, 1500/1500 identical ep, 2026-07-28).
#define SAC_ENTROPY_COEF 0.05f

// Clips (synchronized with stream_ac.py: GRAD_CLIP=5, WEIGHT_CLIP=5)
#define SAC_WEIGHT_CLIP  5.0f
#define SAC_GRAD_CLIP    5.0f
