#include "Machine.h"
#include <cassert>
// Adapter to Machine.
// Rises the level of abstraction.

// Forward declarations
[[nodiscard]] int CountLastFlip(uint64_t P, uint8_t move);
[[nodiscard]] uint64_t Flips(uint64_t P, uint64_t O, uint8_t move);
[[nodiscard]] uint64_t PossibleMoves(uint64_t P, uint64_t O);


int CountLastFlip(Position pos, Field move)
{
	return CountLastFlip(pos.GetP(), static_cast<uint8_t>(move));
}

BitBoard Flips(Board board, Field move)
{
	return BitBoard{ Flips(board.P, board.O, static_cast<uint8_t>(move)) };
}

Board Play(Board board, Field move, BitBoard flips)
{
	return { board.O ^ flips, board.P ^ flips ^ BitBoard{ move } };
}

Position Play(Position pos, Field move)
{
	assert(pos.Empties()[move]); // move field is free.

	const auto flips = Flips(pos, move);

	assert(flips); // flips something.
	assert((pos.GetO() & flips) == flips); // only flipping opponent stones.

	return Play(pos, move, flips);
}

Position PlayPass(Position pos)
{
	return { pos.GetO(), pos.GetP() };
}

Moves PossibleMoves(Position pos)
{
	return Moves{ BitBoard{ PossibleMoves(pos.GetP(), pos.GetO()) } };
}