#pragma once
#include "config.h"
#include "env_gridworld.h"

// Tabular Q-learning with eligibility traces (TD-lambda).
// Only used when AGENT_SELECT == 0.
class StreamQAgent {
 public:
  static const int N_STATES  = GridWorldEnv::N_STATES;
  static const int N_ACTIONS = GridWorldEnv::N_ACTIONS;

  float Q[N_STATES][N_ACTIONS];
  float traces[N_STATES][N_ACTIONS];
  float epsilon;

  void init() {
    epsilon = SQ_EPSILON_START;
    for (int s = 0; s < N_STATES; s++)
      for (int a = 0; a < N_ACTIONS; a++) {
        Q[s][a] = 0.0f;
        traces[s][a] = 0.0f;
      }
  }

  int act(int state) {
    if ((float)random(0, 10000) / 10000.0f < epsilon)
      return random(0, N_ACTIONS);
    int best = 0;
    for (int a = 1; a < N_ACTIONS; a++)
      if (Q[state][a] > Q[state][best]) best = a;
    return best;
  }

  void update(int state, int action, float reward, int next_state, bool done) {
    float best_next = 0.0f;
    if (!done) {
      best_next = Q[next_state][0];
      for (int a = 1; a < N_ACTIONS; a++)
        if (Q[next_state][a] > best_next) best_next = Q[next_state][a];
    }
    float td_error = reward + SQ_GAMMA * best_next - Q[state][action];

    for (int s = 0; s < N_STATES; s++)
      for (int a = 0; a < N_ACTIONS; a++)
        traces[s][a] *= SQ_GAMMA * SQ_LAMBDA;
    traces[state][action] = 1.0f;

    for (int s = 0; s < N_STATES; s++)
      for (int a = 0; a < N_ACTIONS; a++)
        Q[s][a] += SQ_ALPHA * td_error * traces[s][a];

    if (done) {
      for (int s = 0; s < N_STATES; s++)
        for (int a = 0; a < N_ACTIONS; a++) traces[s][a] = 0.0f;
      epsilon *= SQ_EPSILON_DECAY;
    }
  }

  uint32_t memory_bytes() const { return sizeof(Q) + sizeof(traces); }
};
