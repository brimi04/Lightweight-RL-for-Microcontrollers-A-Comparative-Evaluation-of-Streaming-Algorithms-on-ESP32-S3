#pragma once
#include <math.h>
#include "config.h"

// One-step online actor-critic (TD(0)), linear softmax policy.
// Port of research/agents/stream_ac.py — no malloc, static arrays.
//
// Usage (GridWorld):
//   StreamACAgent<2, 4> agent;   // 2 features, 4 actions
//   agent.init();
//   float obs[2]; env.get_obs(obs);
//   int action = agent.act(obs);
//   env.step(action, &done); env.get_obs(next_obs);
//   agent.update(obs, action, reward, next_obs, done);

template<int NF, int NA>
class StreamACAgent {
 public:
  static const int NF1 = NF + 1;  // features + bias

  float W_actor[NA][NF1];   // logit_a = W_actor[a] . x
  float w_critic[NF1];      // V(x) = w_critic . x

  // Welford normalizer (embedded inline)
  float norm_mean[NF];
  float norm_var[NF];
  float norm_count;

  void init() {
    for (int a = 0; a < NA; a++)
      for (int f = 0; f < NF1; f++) W_actor[a][f] = 0.0f;
    for (int f = 0; f < NF1; f++) w_critic[f] = 0.0f;
    norm_count = 1e-4f;
    for (int f = 0; f < NF; f++) { norm_mean[f] = 0.0f; norm_var[f] = 1.0f; }
  }

  // Updates the Welford statistics AND normalizes -> x[NF1]
  void _norm_update(const float* obs, float* x) {
    norm_count += 1.0f;
    for (int f = 0; f < NF; f++) {
      float d1 = obs[f] - norm_mean[f];
      norm_mean[f] += d1 / norm_count;
      float d2 = obs[f] - norm_mean[f];
      norm_var[f] += (d1 * d2 - norm_var[f]) / norm_count;
      x[f] = (obs[f] - norm_mean[f]) / sqrtf(norm_var[f] + 1e-6f);
    }
    x[NF] = 1.0f;
  }

  // Only normalizes, without touching the statistics (for update())
  void _norm_only(const float* obs, float* x) const {
    for (int f = 0; f < NF; f++)
      x[f] = (obs[f] - norm_mean[f]) / sqrtf(norm_var[f] + 1e-6f);
    x[NF] = 1.0f;
  }

  void _softmax(const float* logits, float* probs) const {
    float mx = logits[0];
    for (int a = 1; a < NA; a++) if (logits[a] > mx) mx = logits[a];
    float s = 0.0f;
    for (int a = 0; a < NA; a++) { probs[a] = expf(logits[a] - mx); s += probs[a]; }
    for (int a = 0; a < NA; a++) probs[a] /= s;
  }

  void _clip() {
    for (int a = 0; a < NA; a++)
      for (int f = 0; f < NF1; f++) {
        if (W_actor[a][f] >  SAC_WEIGHT_CLIP) W_actor[a][f] =  SAC_WEIGHT_CLIP;
        if (W_actor[a][f] < -SAC_WEIGHT_CLIP) W_actor[a][f] = -SAC_WEIGHT_CLIP;
      }
    for (int f = 0; f < NF1; f++) {
      if (w_critic[f] >  SAC_WEIGHT_CLIP) w_critic[f] =  SAC_WEIGHT_CLIP;
      if (w_critic[f] < -SAC_WEIGHT_CLIP) w_critic[f] = -SAC_WEIGHT_CLIP;
    }
  }

  // Normalizes obs, computes softmax, samples the action
  int act(const float* obs) {
    float x[NF1];
    _norm_update(obs, x);
    float logits[NA];
    for (int a = 0; a < NA; a++) {
      logits[a] = 0.0f;
      for (int f = 0; f < NF1; f++) logits[a] += W_actor[a][f] * x[f];
    }
    float probs[NA];
    _softmax(logits, probs);
    // Sampling via the CDF
    float r = (float)random(0, 10000) / 10000.0f;
    float cum = 0.0f;
    for (int a = 0; a < NA; a++) {
      cum += probs[a];
      if (r < cum) return a;
    }
    return NA - 1;
  }

  // One TD(0) step: updates critic and actor
  void update(const float* obs, int action, float reward,
              const float* next_obs, bool terminated) {
    float x[NF1], xn[NF1];
    _norm_only(obs, x);
    _norm_only(next_obs, xn);

    // TD error
    float v = 0.0f, vn = 0.0f;
    for (int f = 0; f < NF1; f++) v += w_critic[f] * x[f];
    if (!terminated) for (int f = 0; f < NF1; f++) vn += w_critic[f] * xn[f];
    float delta = reward + SAC_GAMMA * vn - v;

    // Critic
    for (int f = 0; f < NF1; f++) w_critic[f] += SAC_ALPHA_CRITIC * delta * x[f];

    // Actor: current policy
    float logits[NA];
    for (int a = 0; a < NA; a++) {
      logits[a] = 0.0f;
      for (int f = 0; f < NF1; f++) logits[a] += W_actor[a][f] * x[f];
    }
    float probs[NA];
    _softmax(logits, probs);

    // Entropy H = -sum p log p
    float lp[NA], H = 0.0f;
    for (int a = 0; a < NA; a++) {
      lp[a] = logf(probs[a] + 1e-8f);
      H -= probs[a] * lp[a];
    }

    // Combined gradient: policy gradient + entropy
    for (int a = 0; a < NA; a++) {
      float g_pol = (a == action ? 1.0f : 0.0f) - probs[a];
      float g_ent = -probs[a] * (lp[a] + H);
      float coef  = SAC_ALPHA_ACTOR * (delta * g_pol + SAC_ENTROPY_COEF * g_ent);
      for (int f = 0; f < NF1; f++) W_actor[a][f] += coef * x[f];
    }
    _clip();
  }

  uint32_t memory_bytes() const {
    return sizeof(W_actor) + sizeof(w_critic) + sizeof(norm_mean) + sizeof(norm_var);
  }

  static uint32_t flops_per_update() {
    return 6UL * NA * NF1 + 4UL * NF1;
  }
};
