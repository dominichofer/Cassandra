#include "Machine.h"
#include <cassert>
// Adapter to Machine.
// Rises the level of abstraction.

// Forward declarations
[[nodiscard]] int CountLastFlip(uint64_t P, uint8_t move);
[[nodiscard]] uint64_t Flips(uint64_t P, uint64_t O, uint8_t move);
[[nodiscard]] uint64_t PossibleMoves(uint64_t P, uint64_t O);
[[nodiscard]] uint64_t StableStones(uint64_t P, uint64_t O);


int CountLastFlip(const Position& pos, Field move)
{
	return CountLastFlip(pos.GetP(), static_cast<uint8_t>(move));
}

BitBoard Flips(const Position& pos, Field move)
{
	return BitBoard{ Flips(pos.GetP(), pos.GetO(), static_cast<uint8_t>(move)) };
}

Position Play(const Position& pos, Field move, BitBoard flips)
{
	return { pos.GetO() ^ flips, pos.GetP() ^ flips ^ BitBoard{ move } };
}

Position Play(const Position& pos, Field move)
{
	assert(pos.Empties()[move]); // move field is free.

	const auto flips = Flips(pos, move);

	assert(flips); // flips something.
	assert((pos.GetO() & flips) == flips); // only flipping opponent stones.

	return Play(pos, move, flips);
}

Position PlayPass(const Position& pos)
{
	return { pos.GetO(), pos.GetP() };
}

Moves PossibleMoves(const Position& board)
{
	return Moves{ BitBoard{ PossibleMoves(board.GetP(), board.GetO()) } };
}

BitBoard StableStones(const Position& pos)
{
	return StableStones(pos.GetP(), pos.GetO());
}
