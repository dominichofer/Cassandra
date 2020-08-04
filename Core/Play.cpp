#include "Play.h"
#include "Flips.h"

Position Play(const Position& pos, Field move, BitBoard flips)
{
	assert(flips); // flips something.
	assert((pos.O & flips) == flips); // only flipping opponent stones.

	return { pos.O ^ flips, pos.P ^ flips ^ BitBoard(move) };
}

Position Play(const Position& pos, Field move)
{
	assert(pos.Empties().Get(move)); // move field is free.

	const auto flips = Flips(pos, move);
	return Play(pos, move, flips);
}

Position PlayPass(const Position& pos)
{
	return { pos.O, pos.P };
}
