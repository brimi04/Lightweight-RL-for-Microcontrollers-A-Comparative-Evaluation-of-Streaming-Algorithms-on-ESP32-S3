#pragma once

// Time and heap measurement — Arduino version (uses micros() and ESP.getFreeHeap(),
// available on any ESP32 board with the Arduino framework).

struct ProfileResult {
  uint32_t time_us;    // block duration in microseconds
  uint32_t heap_free;  // free heap before the block (bytes)
};

#define PROFILE_BLOCK(result, code)           \
  do {                                         \
    (result).heap_free = ESP.getFreeHeap();    \
    uint32_t _t0 = micros();                  \
    code;                                      \
    (result).time_us = micros() - _t0;        \
  } while (0)
