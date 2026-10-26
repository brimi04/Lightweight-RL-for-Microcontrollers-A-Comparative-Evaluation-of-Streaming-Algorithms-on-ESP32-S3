// TinyRL — Stream-Q (TD-lambda) on GridWorld for the ESP32
// Class-free version — global arrays, C-style functions
// Compatible with Arduino IDE 2.x + the ESP32 board package

#include "config.h"
#include "env_gridworld.h"

// ── Dimensions ───────────────────────────────────────────────────────────────
#define NS GW_N_STATES    // 25
#define NA GW_N_ACTIONS   // 4

// ── Q table and traces (global so as not to overflow the stack) ─────────────────────
float Q[NS][NA];
float traces[NS][NA];
float sq_epsilon;

// ── Agent initialization ─────────────────────────────────────────────────
void agent_init() {
  sq_epsilon = SQ_EPSILON_START;
  for (int s = 0; s < NS; s++)
    for (int a = 0; a < NA; a++) {
      Q[s][a]      = 0.0f;
      traces[s][a] = 0.0f;
    }
}

// ── Action selection (epsilon-greedy) ──────────────────────────────────────
int agent_act(int state) {
  if ((float)random(0, 10000) / 10000.0f < sq_epsilon)
    return (int)random(0, NA);
  int best = 0;
  for (int a = 1; a < NA; a++)
    if (Q[state][a] > Q[state][best]) best = a;
  return best;
}

// ── TD(lambda) update with replacing traces ─────────────────────────────
void agent_update(int state, int action, float reward, int next_state, bool done) {
  // Best next Q
  float best_next = 0.0f;
  if (!done) {
    best_next = Q[next_state][0];
    for (int a = 1; a < NA; a++)
      if (Q[next_state][a] > best_next) best_next = Q[next_state][a];
  }
  float td_error = reward + SQ_GAMMA * best_next - Q[state][action];

  // Decay all traces
  for (int s = 0; s < NS; s++)
    for (int a = 0; a < NA; a++)
      traces[s][a] *= SQ_GAMMA * SQ_LAMBDA;

  // Replacing trace: set the one of the current pair to 1
  traces[state][action] = 1.0f;

  // Update the whole Q table
  for (int s = 0; s < NS; s++)
    for (int a = 0; a < NA; a++)
      Q[s][a] += SQ_ALPHA * td_error * traces[s][a];

  // At the end of the episode: reset the traces and decay epsilon
  if (done) {
    for (int s = 0; s < NS; s++)
      for (int a = 0; a < NA; a++) traces[s][a] = 0.0f;
    sq_epsilon *= SQ_EPSILON_DECAY;
  }
}

// ── Resource metrics ──────────────────────────────────────────────────────
uint32_t memory_bytes()    { return (uint32_t)NS * NA * 2 * sizeof(float); }  // 800
uint32_t flops_per_update(){ return 4UL * NS * NA; }                           // 400

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("# Stream-Q GridWorld ESP32 - starting...");
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
      int state  = gw_state();
      int action = agent_act(state);
      float rew  = gw_step(action, &done);
      int nstate = gw_state();
      agent_update(state, action, rew, nstate, done);

      total_reward += rew;
      steps++;
      if (done) break;
    }

    uint32_t elapsed_us = micros() - t0;

    Serial.print("stream_q,gridworld,");
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
