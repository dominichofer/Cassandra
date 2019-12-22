#pragma once
#include <cstdint>

[[nodiscard]]
uint64_t StableEdges(uint64_t P, uint64_t O);

// Stable stones of the opponent.
[[nodiscard]]
uint64_t StableStones(uint64_t P, uint64_t O);
