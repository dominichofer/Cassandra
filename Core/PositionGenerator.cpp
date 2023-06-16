#include "PositionGenerator.h"
#include "Bit.h"
#include <cstdint>

Position RandomPositionGenerator::operator()() noexcept
{
	// Each field has a:
	//  25% chance to have a player's disc on it,
	//  25% chance to have an opponent's disc on it,
	//  50% chance to be empty.

	uint64_t a{ dist(rnd_engine) };
	uint64_t b{ dist(rnd_engine) };
	return { a & ~b, ~a & b };
}

Position RandomPosition(unsigned int seed)
{
	return RandomPositionGenerator{ seed }();
}
