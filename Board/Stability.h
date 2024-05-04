#pragma once
#include "Board/Board.h"
#include <cstdint>

uint64_t StableEdges(const Position&);

// Stable stones of the opponent.
uint64_t StableStonesOpponent(const Position&);
