// TinyRL — Stream-Q on CartPole for the ESP32
// Tabular Q-learning with eligibility traces TD(lambda), bin discretizer.
// Port of research/agents/stream_q.py + envs/wrappers.py CARTPOLE_DISCRETIZER
//
// MEMORY NOTE: Q[5184][2] + traces[5184][2] = ~82 KB in SRAM.
// The ESP32-S3 has 512 KB of internal SRAM — enough.
// TIME NOTE: the sweep over the traces (O(n_states) per step) is slower
// than GridWorld. Expect ~2-4 min of total run time for 1500 episodes.

#include <math.h>
#include "config.h"
#include "env_cartpole.h"

// ── Q table and eligibility traces ─────────────────────────────────────────
float Q[SQ_N_STATES][SQ_N_ACTIONS];
float traces[SQ_N_STATES][SQ_N_ACTIONS];

float sq_epsilon;

// ── Discretizer ─────────────────────────────────────────────────────────────
static inline int _disc1(float v, float lo, float hi, int bins) {
    float ratio = (v - lo) / (hi - lo + 1e-8f);
    if (ratio < 0.0f) ratio = 0.0f;
    if (ratio > 1.0f) ratio = 1.0f;
    int idx = (int)(ratio * (bins - 1));
    if (idx >= bins) idx = bins - 1;
    return idx;
}

int obs_to_idx(const float* obs) {
    int i0 = _disc1(obs[0], SQ_LOW_0, SQ_HIGH_0, SQ_BINS_0);
    int i1 = _disc1(obs[1], SQ_LOW_1, SQ_HIGH_1, SQ_BINS_1);
    int i2 = _disc1(obs[2], SQ_LOW_2, SQ_HIGH_2, SQ_BINS_2);
    int i3 = _disc1(obs[3], SQ_LOW_3, SQ_HIGH_3, SQ_BINS_3);
    return ((i0 * SQ_BINS_1 + i1) * SQ_BINS_2 + i2) * SQ_BINS_3 + i3;
}

// ── Initialization ────────────────────────────────────────────────────────────
void agent_init() {
    sq_epsilon = SQ_EPSILON_START;
    for (int s = 0; s < SQ_N_STATES; s++)
        for (int a = 0; a < SQ_N_ACTIONS; a++) {
            Q[s][a]      = 0.0f;
            traces[s][a] = 0.0f;
        }
}

// ── Epsilon-greedy action ─────────────────────────────────────────────────────
int agent_act(int state_idx) {
    if ((float)random(0, 10000) / 10000.0f < sq_epsilon)
        return (int)random(0, SQ_N_ACTIONS);
    int best = 0;
    for (int a = 1; a < SQ_N_ACTIONS; a++)
        if (Q[state_idx][a] > Q[state_idx][best]) best = a;
    return best;
}

// ── TD(lambda) update ──────────────────────────────────────────────────
void agent_update(int s, int action, float reward, int s_next,
                  bool terminated, bool episode_end) {
    float best_next = 0.0f;
    if (!terminated) {
        best_next = Q[s_next][0];
        for (int a = 1; a < SQ_N_ACTIONS; a++)
            if (Q[s_next][a] > best_next) best_next = Q[s_next][a];
    }
    float td_error = reward + SQ_GAMMA * best_next - Q[s][action];

    // Decay all traces and mark the current one
    float decay = SQ_GAMMA * SQ_LAMBDA;
    for (int i = 0; i < SQ_N_STATES; i++)
        for (int a = 0; a < SQ_N_ACTIONS; a++)
            traces[i][a] *= decay;
    traces[s][action] = 1.0f;

    // Update Q
    float alpha_td = SQ_ALPHA * td_error;
    for (int i = 0; i < SQ_N_STATES; i++)
        for (int a = 0; a < SQ_N_ACTIONS; a++)
            Q[i][a] += alpha_td * traces[i][a];

    if (episode_end) {
        sq_epsilon *= SQ_EPSILON_DECAY;
        // Reset the traces at the start of each episode
        for (int i = 0; i < SQ_N_STATES; i++)
            for (int a = 0; a < SQ_N_ACTIONS; a++)
                traces[i][a] = 0.0f;
    }
}

// ── Metrics ──────────────────────────────────────────────────────────────────
uint32_t memory_bytes()    { return 2UL * SQ_N_STATES * SQ_N_ACTIONS * sizeof(float); }
uint32_t flops_per_update(){ return 4UL * SQ_N_STATES * SQ_N_ACTIONS; }

// ── Main setup ───────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("# Stream-Q CartPole ESP32 - starting...");
    Serial.println("# NOTE: ~2-4 min of run time (O(n_states) per step)");
    Serial.flush();
    delay(500);

    randomSeed(SEED);
    agent_init();

    const uint32_t mem   = memory_bytes();
    const uint32_t flops = flops_per_update();

    Serial.println("agent,env,seed,episode,reward,steps,memory_bytes,flops_per_update,wall_time_s");
    Serial.flush();

    float obs[4], nobs[4];

    for (int episode = 0; episode < N_EPISODES; episode++) {
        cp_reset();
        cp_get_obs(obs);
        int   state        = obs_to_idx(obs);
        float total_reward = 0.0f;
        int   steps        = 0;
        bool  done         = false;
        uint32_t t0        = micros();

        for (int t = 0; t < MAX_EPISODE_STEPS && !done; t++) {
            int action = agent_act(state);
            float rew  = cp_step(action, &done);
            cp_get_obs(nobs);
            int next_state = obs_to_idx(nobs);

            bool ep_end = done || (t + 1 >= MAX_EPISODE_STEPS);
            agent_update(state, action, rew, next_state, done, ep_end);

            state = next_state;
            total_reward += rew;
            steps++;
        }

        uint32_t elapsed_us = micros() - t0;

        Serial.print("stream_q,cartpole,");
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
