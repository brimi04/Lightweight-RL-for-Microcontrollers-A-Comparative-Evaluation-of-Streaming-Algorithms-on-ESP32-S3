// TinyRL — Stream-AC on GridWorld for the ESP32
// Template-free version for maximum compatibility with the Arduino IDE

#include "config.h"
#include "env_gridworld.h"

// ── Fixed dimensions for GridWorld ────────────────────────────────────────
#define NF  2   // features: row, column
#define NA  4   // actions: up, down, left, right
#define NF1 3   // NF + 1 (bias)

// ── Global weights (not on the stack) ─────────────────────────────────────────────
float W_actor[NA][NF1];
float w_critic[NF1];
float norm_mean[NF];
float norm_var[NF];
float norm_count;

GridWorldEnv env;

// ── Normalizer ──────────────────────────────────────────────────────────────
void norm_update(const float* obs, float* x) {
  norm_count += 1.0f;
  for (int f = 0; f < NF; f++) {
    float d1 = obs[f] - norm_mean[f];
    norm_mean[f] += d1 / norm_count;
    float d2 = obs[f] - norm_mean[f];
    norm_var[f] += (d1 * d2 - norm_var[f]) / norm_count;
    float std = sqrtf(norm_var[f] > 0 ? norm_var[f] : 0.0f) + 1e-6f;
    x[f] = (obs[f] - norm_mean[f]) / std;
  }
  x[NF] = 1.0f;
}

void norm_only(const float* obs, float* x) {
  for (int f = 0; f < NF; f++) {
    float std = sqrtf(norm_var[f] > 0 ? norm_var[f] : 0.0f) + 1e-6f;
    x[f] = (obs[f] - norm_mean[f]) / std;
  }
  x[NF] = 1.0f;
}

// ── Softmax ───────────────────────────────────────────────────────────────────
void softmax(const float* logits, float* probs) {
  float mx = logits[0];
  for (int a = 1; a < NA; a++) if (logits[a] > mx) mx = logits[a];
  float s = 0.0f;
  for (int a = 0; a < NA; a++) {
    probs[a] = expf(logits[a] - mx);
    s += probs[a];
  }
  for (int a = 0; a < NA; a++) probs[a] /= s;
}

// ── Weight clipping ─────────────────────────────────────────────────────────────
void clip_weights() {
  for (int a = 0; a < NA; a++)
    for (int f = 0; f < NF1; f++) {
      W_actor[a][f] = constrain(W_actor[a][f], -SAC_WEIGHT_CLIP, SAC_WEIGHT_CLIP);
    }
  for (int f = 0; f < NF1; f++)
    w_critic[f] = constrain(w_critic[f], -SAC_WEIGHT_CLIP, SAC_WEIGHT_CLIP);
}

// ── Agent ────────────────────────────────────────────────────────────────────
void agent_init() {
  for (int a = 0; a < NA; a++)
    for (int f = 0; f < NF1; f++) W_actor[a][f] = 0.0f;
  for (int f = 0; f < NF1; f++) w_critic[f] = 0.0f;
  norm_count = 1e-4f;
  for (int f = 0; f < NF; f++) { norm_mean[f] = 0.0f; norm_var[f] = 1.0f; }
}

int agent_act(const float* obs) {
  float x[NF1], logits[NA], probs[NA];
  norm_update(obs, x);
  for (int a = 0; a < NA; a++) {
    logits[a] = 0.0f;
    for (int f = 0; f < NF1; f++) logits[a] += W_actor[a][f] * x[f];
  }
  softmax(logits, probs);
  float r = (float)random(0, 10000) / 10000.0f;
  float cum = 0.0f;
  for (int a = 0; a < NA; a++) {
    cum += probs[a];
    if (r < cum) return a;
  }
  return NA - 1;
}

void agent_update(const float* obs, int action, float reward,
                  const float* next_obs, bool terminated) {
  float x[NF1], xn[NF1], logits[NA], probs[NA], lp[NA];
  norm_only(obs, x);
  norm_only(next_obs, xn);

  float v = 0.0f, vn = 0.0f;
  for (int f = 0; f < NF1; f++) v += w_critic[f] * x[f];
  if (!terminated) for (int f = 0; f < NF1; f++) vn += w_critic[f] * xn[f];
  float delta = reward + SAC_GAMMA * vn - v;

  for (int f = 0; f < NF1; f++) {
    float dc = SAC_ALPHA_CRITIC * delta * x[f];
    dc = constrain(dc, -SAC_GRAD_CLIP, SAC_GRAD_CLIP);
    w_critic[f] += dc;
  }

  for (int a = 0; a < NA; a++) {
    logits[a] = 0.0f;
    for (int f = 0; f < NF1; f++) logits[a] += W_actor[a][f] * x[f];
  }
  softmax(logits, probs);

  float H = 0.0f;
  for (int a = 0; a < NA; a++) {
    lp[a] = logf(probs[a] + 1e-8f);
    H -= probs[a] * lp[a];
  }

  for (int a = 0; a < NA; a++) {
    float g_pol = (a == action ? 1.0f : 0.0f) - probs[a];
    float g_ent = -probs[a] * (lp[a] + H);
    float coef  = SAC_ALPHA_ACTOR * (delta * g_pol + SAC_ENTROPY_COEF * g_ent);
    for (int f = 0; f < NF1; f++) {
      float da = constrain(coef * x[f], -SAC_GRAD_CLIP, SAC_GRAD_CLIP);
      W_actor[a][f] += da;
    }
  }
  clip_weights();
}

uint32_t memory_bytes() {
  return sizeof(W_actor) + sizeof(w_critic) + sizeof(norm_mean) + sizeof(norm_var);
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(2000);

  // Start-up confirmation
  Serial.println("# Stream-AC GridWorld ESP32 - starting...");
  Serial.flush();
  delay(500);

  randomSeed(SEED);
  agent_init();
  env.reset();

  const uint32_t mem   = memory_bytes();
  const uint32_t flops = 6UL * NA * NF1 + 4UL * NF1;

  Serial.println("agent,env,seed,episode,reward,steps,memory_bytes,flops_per_update,wall_time_s");
  Serial.flush();

  for (int episode = 0; episode < N_EPISODES; episode++) {
    env.reset();
    float total_reward = 0.0f;
    int   steps = 0;
    bool  done  = false;
    uint32_t t0 = micros();

    for (int t = 0; t < MAX_EPISODE_STEPS; t++) {
      float obs[NF], nobs[NF];
      env.get_obs(obs);
      int action = agent_act(obs);
      float rew  = env.step(action, &done);
      env.get_obs(nobs);
      agent_update(obs, action, rew, nobs, done);

      total_reward += rew;
      steps++;
      if (done) break;
    }

    uint32_t elapsed_us = micros() - t0;

    Serial.print("stream_ac,gridworld,");
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
