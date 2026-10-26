// TinyRL — TinyDQN on GridWorld for the ESP32
// 2-layer network (2->12->4), 32-transition mini-buffer, manual forward/backward
// Port of research/agents/tiny_dqn.py + core/nn_manual.py
// Class-free version — global arrays, C-style functions

#include <math.h>
#include "config.h"
#include "env_gridworld.h"

// ── Network weights (global) ────────────────────────────────────────────────
float W1[DQN_H][DQN_NF];   // layer 1: (H x NF)
float b1[DQN_H];
float W2[DQN_NA][DQN_H];   // layer 2: (NA x H)
float b2[DQN_NA];

// ── Welford normalizer ───────────────────────────────────────────────────────
float norm_mean[DQN_NF];
float norm_var[DQN_NF];
float norm_count;

// ── Mini replay buffer ────────────────────────────────────────────────────────
struct Transition {
  float obs[DQN_NF];
  int   action;
  float reward;
  float next_obs[DQN_NF];
  bool  terminated;
};
Transition buf[DQN_BUF];
int buf_size;   // valid elements (0..BUF_SIZE)
int buf_head;   // next slot to write (circular)

// ── Epsilon ───────────────────────────────────────────────────────────────────
float dqn_epsilon;

// ── Utility: random float in [-1, 1] ──────────────────────────────────────
static float randf() {
  return ((float)random(-32767, 32768)) / 32767.0f;
}

// ── Initialization ────────────────────────────────────────────────────────────
void agent_init() {
  dqn_epsilon = DQN_EPSILON_START;
  buf_size = 0;
  buf_head = 0;
  norm_count = 1e-4f;
  for (int f = 0; f < DQN_NF; f++) { norm_mean[f] = 0.0f; norm_var[f] = 1.0f; }

  // Xavier uniform: limit = sqrt(6 / (n_in + n_out))
  float lim1 = sqrtf(6.0f / (DQN_NF + DQN_H));
  for (int i = 0; i < DQN_H; i++) {
    b1[i] = 0.0f;
    for (int j = 0; j < DQN_NF; j++) W1[i][j] = randf() * lim1;
  }
  float lim2 = sqrtf(6.0f / (DQN_H + DQN_NA));
  for (int i = 0; i < DQN_NA; i++) {
    b2[i] = 0.0f;
    for (int j = 0; j < DQN_H; j++) W2[i][j] = randf() * lim2;
  }
}

// ── Normalizer: update statistics AND normalize ────────────────────────
void norm_update(const float* obs, float* x) {
  norm_count += 1.0f;
  for (int f = 0; f < DQN_NF; f++) {
    float d1 = obs[f] - norm_mean[f];
    norm_mean[f] += d1 / norm_count;
    float d2 = obs[f] - norm_mean[f];
    norm_var[f] += (d1 * d2 - norm_var[f]) / norm_count;
    float std = sqrtf(norm_var[f] > 0 ? norm_var[f] : 0.0f) + 1e-6f;
    x[f] = (obs[f] - norm_mean[f]) / std;
  }
}

// ── Normalizer: normalize only (without updating the stats) ──────────────────────
void norm_only(const float* obs, float* x) {
  for (int f = 0; f < DQN_NF; f++) {
    float std = sqrtf(norm_var[f] > 0 ? norm_var[f] : 0.0f) + 1e-6f;
    x[f] = (obs[f] - norm_mean[f]) / std;
  }
}

// ── Forward L1 (ReLU): h = relu(W1 x + b1) ───────────────────────────────────
//   stores z1 for the backward pass if save_cache=true
void forward_l1(const float* x, float* h, float* z1_out) {
  for (int i = 0; i < DQN_H; i++) {
    float z = b1[i];
    for (int j = 0; j < DQN_NF; j++) z += W1[i][j] * x[j];
    if (z1_out) z1_out[i] = z;
    h[i] = z > 0.0f ? z : 0.0f;
  }
}

// ── Forward L2 (linear): q = W2 h + b2 ───────────────────────────────────────
void forward_l2(const float* h, float* q) {
  for (int i = 0; i < DQN_NA; i++) {
    float z = b2[i];
    for (int j = 0; j < DQN_H; j++) z += W2[i][j] * h[j];
    q[i] = z;
  }
}

// ── One training step (SGD on one transition) ───────────────────────
void train_step(const float* obs, int action, float reward,
                const float* next_obs, bool terminated) {
  float x[DQN_NF], xn[DQN_NF];
  norm_only(obs, x);
  norm_only(next_obs, xn);

  // Inference forward on next_obs (no cache — for the TD target)
  float h_next[DQN_H], q_next[DQN_NA];
  forward_l1(xn, h_next, NULL);
  forward_l2(h_next, q_next);
  float q_next_max = q_next[0];
  for (int a = 1; a < DQN_NA; a++) if (q_next[a] > q_next_max) q_next_max = q_next[a];
  float target = reward + (terminated ? 0.0f : DQN_GAMMA * q_next_max);

  // Forward on obs (with cache z1 and h for backward)
  float z1[DQN_H], h[DQN_H], q[DQN_NA];
  forward_l1(x, h, z1);
  forward_l2(h, q);

  // TD error (only on the action taken)
  float td = q[action] - target;
  td = constrain(td, -DQN_GRAD_CLIP, DQN_GRAD_CLIP);

  // Backward L2: grad_out2[a] = td, 0 for the rest
  // grad_h[j] = sum_i W2[i][j] * grad_out2[i]  = W2[action][j] * td
  float grad_h[DQN_H];
  for (int j = 0; j < DQN_H; j++) grad_h[j] = W2[action][j] * td;

  // Update W2[action] and b2[action]
  for (int j = 0; j < DQN_H; j++) {
    W2[action][j] -= DQN_LR * td * h[j];
    W2[action][j] = constrain(W2[action][j], -DQN_WEIGHT_CLIP, DQN_WEIGHT_CLIP);
  }
  b2[action] -= DQN_LR * td;
  b2[action] = constrain(b2[action], -DQN_WEIGHT_CLIP, DQN_WEIGHT_CLIP);

  // Backward L1: ReLU gate + gradient of W1
  for (int i = 0; i < DQN_H; i++) {
    float g = (z1[i] > 0.0f ? grad_h[i] : 0.0f);  // ReLU gate
    g = constrain(g, -DQN_GRAD_CLIP, DQN_GRAD_CLIP);
    for (int j = 0; j < DQN_NF; j++) {
      W1[i][j] -= DQN_LR * g * x[j];
      W1[i][j] = constrain(W1[i][j], -DQN_WEIGHT_CLIP, DQN_WEIGHT_CLIP);
    }
    b1[i] -= DQN_LR * g;
    b1[i] = constrain(b1[i], -DQN_WEIGHT_CLIP, DQN_WEIGHT_CLIP);
  }
}

// ── Action (epsilon-greedy) ───────────────────────────────────────────────────
int agent_act(const float* obs) {
  float x[DQN_NF];
  norm_update(obs, x);   // updates the normalizer stats
  if ((float)random(0, 10000) / 10000.0f < dqn_epsilon)
    return (int)random(0, DQN_NA);
  float h[DQN_H], q[DQN_NA];
  forward_l1(x, h, NULL);
  forward_l2(h, q);
  int best = 0;
  for (int a = 1; a < DQN_NA; a++) if (q[a] > q[best]) best = a;
  return best;
}

// ── Store in the buffer (circular) ──────────────────────────────────────────────
void buf_push(const float* obs, int action, float reward,
              const float* next_obs, bool terminated) {
  Transition& t = buf[buf_head];
  for (int f = 0; f < DQN_NF; f++) { t.obs[f] = obs[f]; t.next_obs[f] = next_obs[f]; }
  t.action = action; t.reward = reward; t.terminated = terminated;
  buf_head = (buf_head + 1) % DQN_BUF;
  if (buf_size < DQN_BUF) buf_size++;
}

// ── Full update (online + one random replay) ────────────────────────────
void agent_update(const float* obs, int action, float reward,
                  const float* next_obs, bool terminated, bool episode_end) {
  // Online training
  train_step(obs, action, reward, next_obs, terminated);

  // Store in the buffer and replay if there are enough transitions
  buf_push(obs, action, reward, next_obs, terminated);
  if (buf_size >= 4) {
    int idx = (int)random(0, buf_size);
    Transition& t = buf[idx];
    train_step(t.obs, t.action, t.reward, t.next_obs, t.terminated);
  }

  if (episode_end) dqn_epsilon *= DQN_EPSILON_DECAY;
}

// ── Resource metrics ──────────────────────────────────────────────────────
uint32_t memory_bytes() {
  uint32_t weights = (uint32_t)(DQN_H * DQN_NF + DQN_H + DQN_NA * DQN_H + DQN_NA) * sizeof(float);
  uint32_t buffer  = (uint32_t)DQN_BUF * (DQN_NF * 2 + 3) * sizeof(float);
  uint32_t norm    = (uint32_t)DQN_NF * 2 * sizeof(float);
  return weights + buffer + norm;
}

uint32_t flops_per_update() {
  uint32_t fwd = 2UL * DQN_H * DQN_NF + 2UL * DQN_NA * DQN_H;   // 144
  uint32_t bwd = 4UL * DQN_H * DQN_NF + 4UL * DQN_NA * DQN_H;   // 288
  return 2UL * (2UL * fwd + bwd);  // 1152 (two _train_step calls per step: current transition + one from the mini-buffer, as in tiny_dqn.py)
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("# TinyDQN GridWorld ESP32 - starting...");
  Serial.flush();
  delay(500);

  randomSeed(SEED);
  agent_init();
  gw_reset();

  const uint32_t mem   = memory_bytes();
  const uint32_t flops = flops_per_update();

  Serial.println("agent,env,seed,episode,reward,steps,memory_bytes,flops_per_update,wall_time_s");
  Serial.flush();

  for (int episode = 0; episode < N_EPISODES; episode++) {
    gw_reset();
    float total_reward = 0.0f;
    int   steps        = 0;
    bool  done         = false;
    uint32_t t0        = micros();

    for (int t = 0; t < MAX_EPISODE_STEPS; t++) {
      float obs[DQN_NF], nobs[DQN_NF];
      gw_get_obs(obs);
      int action = agent_act(obs);
      float rew  = gw_step(action, &done);
      gw_get_obs(nobs);
      agent_update(obs, action, rew, nobs, done, done);

      total_reward += rew;
      steps++;
      if (done) break;
    }

    uint32_t elapsed_us = micros() - t0;

    Serial.print("tiny_dqn,gridworld,");
    Serial.print(SEED);
    Serial.print(",");
    Serial.print(episode);
    Serial.print(",");
    Serial.print(total_reward, 2);
    Serial.print(",");
    Serial.print(steps);
    Serial.print(",");
    Serial.print(mem);
    Serial.print(",");
    Serial.print(flops);
    Serial.print(",");
    Serial.println(elapsed_us / 1000000.0f, 6);
  }

  Serial.println("DONE");
  Serial.flush();
}

void loop() {}
