#pragma once
#include "Position.h"

[[nodiscard]]
BitBoard StableEdges(const Position&);

// Stable stones of the opponent.
[[nodiscard]]
BitBoard StableStones(const Position&);
