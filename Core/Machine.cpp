#include "Machine.h"
#include "Machine/CountLastFlip.h"
#include "Machine/Flips.h"
#include "Machine/PossibleMoves.h"
#include <cassert>

int CountLastFlip(Position pos, Field move)
{
	return CountLastFlip(pos.GetP(), move);
}

Position Play(Position pos, Field move)
{
	assert(TestBit(pos.Empties(), move)); // move field is free.

	const auto flips = Flips(pos.GetP(), pos.GetO(), move);

	assert(flips); // flips something.
	assert(TestBits(pos.GetO(), flips)); // only flipping opponent stones.

	return { pos.GetO() ^ flips, pos.GetP() ^ flips | Bit(move) };
}

Position PlayPass(Position pos)
{
	return { pos.GetO(), pos.GetP() };
}

Moves PossibleMoves(Position pos)
{
	return Moves(PossibleMoves(pos.GetP(), pos.GetO()));
}