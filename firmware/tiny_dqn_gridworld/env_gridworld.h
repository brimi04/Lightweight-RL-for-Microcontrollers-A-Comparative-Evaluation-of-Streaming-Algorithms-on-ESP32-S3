#pragma once
#include "config.h"

// GridWorld 5x5 — continuous observation [row, col] as float for TinyDQN

int gw_row, gw_col;

void gw_reset() { gw_row = 0; gw_col = 0; }

void gw_get_obs(float* obs) {
  obs[0] = (float)gw_row;
  obs[1] = (float)gw_col;
}

float gw_step(int action, bool* done) {
  int dr = 0, dc = 0;
  switch (action) {
    case 0: dr = -1; break;
    case 1: dr =  1; break;
    case 2: dc = -1; break;
    case 3: dc =  1; break;
  }
  gw_row = constrain(gw_row + dr, 0, GRID_SIZE - 1);
  gw_col = constrain(gw_col + dc, 0, GRID_SIZE - 1);
  bool at_goal = (gw_row == GRID_SIZE - 1) && (gw_col == GRID_SIZE - 1);
  *done = at_goal;
  return at_goal ? 10.0f : -1.0f;
}
