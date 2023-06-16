#pragma once
#include "Core/Core.h"
#include <cstdint>

//uint64_t StableC2(uint64_t); //TODO: Remove!

uint64_t StableEdges(const Position&);

// Stable stones of the opponent.
uint64_t StableStonesOpponent(const Position&);

// Stable corners + 1 of the opponent.
uint64_t StableCornersOpponent(const Position&);

int StabilityBasedMaxScore(const Position&);