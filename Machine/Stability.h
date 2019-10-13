#pragma once
#include "Core/Position.h"
#include <cstdint>

[[nodiscard]]
BitBoard StableEdges(Position);

// Stable stones of the opponent.
[[nodiscard]]
BitBoard StableStones(Position);
