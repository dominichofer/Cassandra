#include "PositionGenerator.h"
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

Position RandomPositionWithEmptyCount(int empty_count, unsigned int seed)
{
	std::mt19937_64 rnd_engine{ seed };
	std::uniform_int_distribution<int> boolean{ 0, 1 };

	Position pos{ 0, 0 };
	while (pos.EmptyCount() > empty_count)
	{
		int rnd = std::uniform_int_distribution<int>(0, pos.EmptyCount() - 1)(rnd_engine);

		// deposit bit on an empty field
		uint64_t bit = PDep(1ULL << rnd, pos.Empties());

		if (boolean(rnd_engine))
			pos = Position{ pos.Player() | bit, pos.Opponent() };
		else
			pos = Position{ pos.Player(), pos.Opponent() | bit };
	}
	return pos;
}
