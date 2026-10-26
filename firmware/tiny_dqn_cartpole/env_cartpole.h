#pragma once
#include <math.h>

// ── CartPole-v1 — embedded physics (Barto, Sutton & Anderson 1983) ─────────────
// Port of gymnasium CartPole-v1, standard Euler integration (same as gymnasium).
// Observation: [x, x_dot, theta, theta_dot]  (4 floats)
// Actions:     0 = push left, 1 = push right
// Reward:      +1.0 per step until termination
// Termination: |x| > 2.4 OR |theta| > 12° (0.2094 rad) OR steps >= MAX_EPISODE_STEPS

#define CP_GRAVITY       9.8f
#define CP_MASSCART      1.0f
#define CP_MASSPOLE      0.1f
#define CP_TOTAL_MASS    1.1f     // masscart + masspole
#define CP_POLELEN       0.5f     // half of the pole length
#define CP_POLEMASSLEN   0.05f    // masspole * polelen
#define CP_FORCE_MAG     10.0f
#define CP_TAU           0.02f    // dt (seconds)

#define CP_X_THRESH      2.4f
#define CP_THETA_THRESH  0.20943951f   // 12 degrees in radians

static float cp_x, cp_x_dot, cp_theta, cp_theta_dot;

// Resets the environment with a random state in [-0.05, 0.05]
// (same distribution as gymnasium CartPole-v1)
void cp_reset() {
    cp_x         = (float)random(-500, 501) / 10000.0f;
    cp_x_dot     = (float)random(-500, 501) / 10000.0f;
    cp_theta     = (float)random(-500, 501) / 10000.0f;
    cp_theta_dot = (float)random(-500, 501) / 10000.0f;
}

// Copies the current state into obs[4]
void cp_get_obs(float* obs) {
    obs[0] = cp_x;
    obs[1] = cp_x_dot;
    obs[2] = cp_theta;
    obs[3] = cp_theta_dot;
}

// Applies the action, integrates the physics, returns the reward and writes *done
float cp_step(int action, bool* done) {
    float force = (action == 1) ? CP_FORCE_MAG : -CP_FORCE_MAG;

    float cos_th = cosf(cp_theta);
    float sin_th = sinf(cp_theta);

    // Accelerations (equations of motion of the inverted pendulum)
    float temp    = (force + CP_POLEMASSLEN * cp_theta_dot * cp_theta_dot * sin_th)
                    / CP_TOTAL_MASS;
    float th_acc  = (CP_GRAVITY * sin_th - cos_th * temp)
                    / (CP_POLELEN * (4.0f/3.0f - CP_MASSPOLE * cos_th * cos_th / CP_TOTAL_MASS));
    float x_acc   = temp - CP_POLEMASSLEN * th_acc * cos_th / CP_TOTAL_MASS;

    // Standard Euler integration (same as gymnasium CartPole-v1, Bug 9 fix)
    // Position updated with the velocity from BEFORE the update.
    cp_x         += CP_TAU * cp_x_dot;
    cp_x_dot     += CP_TAU * x_acc;
    cp_theta     += CP_TAU * cp_theta_dot;
    cp_theta_dot += CP_TAU * th_acc;

    // Termination
    *done = (cp_x < -CP_X_THRESH   || cp_x > CP_X_THRESH ||
             cp_theta < -CP_THETA_THRESH || cp_theta > CP_THETA_THRESH);

    return 1.0f;  // constant reward until termination
}
