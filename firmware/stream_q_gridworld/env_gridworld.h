#pragma once
#include "config.h"

// GridWorld 5x5: discrete state, 4 actions (up/down/left/right)
// State: linear index row*GRID_SIZE + col
// Reward: +10 at the goal (row=4, col=4), -1 on any other step

#define GW_N_STATES  (GRID_SIZE * GRID_SIZE)   // 25
#define GW_N_ACTIONS 4

int  gw_row;
int  gw_col;

void gw_reset() { gw_row = 0; gw_col = 0; }

int gw_state() { return gw_row * GRID_SIZE + gw_col; }

// Returns the reward; writes done
float gw_step(int action, bool* done) {
  int dr = 0, dc = 0;
  switch (action) {
    case 0: dr = -1; break;  // up
    case 1: dr =  1; break;  // down
    case 2: dc = -1; break;  // left
    case 3: dc =  1; break;  // right
  }
  gw_row = constrain(gw_row + dr, 0, GRID_SIZE - 1);
  gw_col = constrain(gw_col + dc, 0, GRID_SIZE - 1);
  bool at_goal = (gw_row == GRID_SIZE - 1) && (gw_col == GRID_SIZE - 1);
  *done = at_goal;
  return at_goal ? 10.0f : -1.0f;
}
