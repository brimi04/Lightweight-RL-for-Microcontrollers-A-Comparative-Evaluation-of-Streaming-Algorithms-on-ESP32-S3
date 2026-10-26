#pragma once
#include "config.h"

class GridWorldEnv {
 public:
  static const int N_ACTIONS  = 4;                    // up, down, left, right
  static const int N_STATES   = GRID_SIZE * GRID_SIZE;
  static const int N_FEATURES = 2;                    // observation: [row, col]

  int row, col;

  void reset() { row = 0; col = 0; }

  int state_index() const { return row * GRID_SIZE + col; }

  // Continuous observation [row, col] as float — for Stream-AC
  void get_obs(float* obs) const {
    obs[0] = (float)row;
    obs[1] = (float)col;
  }

  // Returns the reward; writes to *done if the episode ended
  float step(int action, bool* done) {
    int dr = 0, dc = 0;
    switch (action) {
      case 0: dr = -1; break;  // up
      case 1: dr =  1; break;  // down
      case 2: dc = -1; break;  // left
      case 3: dc =  1; break;  // right
    }
    row = constrain(row + dr, 0, GRID_SIZE - 1);
    col = constrain(col + dc, 0, GRID_SIZE - 1);
    bool at_goal = (row == GRID_SIZE - 1) && (col == GRID_SIZE - 1);
    *done = at_goal;
    return at_goal ? 10.0f : -1.0f;
  }
};
