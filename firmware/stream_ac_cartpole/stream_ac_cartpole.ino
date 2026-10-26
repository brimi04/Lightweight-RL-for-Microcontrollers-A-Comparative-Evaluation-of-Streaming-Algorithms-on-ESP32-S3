// TinyRL — Stream-AC on CartPole for the ESP32
// Online linear actor-critic, softmax policy, Welford normalizer.
// Port de research/agents/stream_ac.py
// Environment: CartPole-v1 (embedded physics, no gymnasium)

#include <math.h>
#include "config.h"
#include "env_cartpole.h"
#include "agent_stream_ac.h"

// Agent instance: 4 input features, 2 actions
StreamACAgent<SAC_NF, SAC_NA> agent;

void setup() {
    Serial.begin(115200);
    delay(2000);

    Serial.println("# Stream-AC CartPole ESP32 - starting...");
    Serial.flush();
    delay(500);

    randomSeed(SEED);
    agent.init();

    const uint32_t mem   = agent.memory_bytes();
    const uint32_t flops = StreamACAgent<SAC_NF, SAC_NA>::flops_per_update();

    Serial.println("agent,env,seed,episode,reward,steps,memory_bytes,flops_per_update,wall_time_s");
    Serial.flush();

    for (int episode = 0; episode < N_EPISODES; episode++) {
        cp_reset();

        float total_reward = 0.0f;
        int   steps        = 0;
        bool  done         = false;
        uint32_t t0        = micros();

        float obs[SAC_NF], nobs[SAC_NF];

        for (int t = 0; t < MAX_EPISODE_STEPS && !done; t++) {
            cp_get_obs(obs);
            int action = agent.act(obs);
            float rew  = cp_step(action, &done);
            cp_get_obs(nobs);
            agent.update(obs, action, rew, nobs, done);

            total_reward += rew;
            steps++;
        }

        uint32_t elapsed_us = micros() - t0;

        Serial.print("stream_ac,cartpole,");
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
