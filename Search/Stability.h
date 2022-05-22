#pragma once
#include "Core/Core.h"

BitBoard StableEdges(const Position&);

// Stable stones of the opponent.
BitBoard StableStonesOpponent(const Position&);

// Stable corners + 1 of the opponent.
BitBoard StableCornersOpponent(const Position&);

int StabilityBasedMaxScore(const Position&);