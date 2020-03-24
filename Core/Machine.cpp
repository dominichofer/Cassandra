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
	return CountLastFlip(pos.P, static_cast<uint8_t>(move));
}

BitBoard Flips(const Position& pos, Field move)
{
	return BitBoard{ Flips(pos.P, pos.O, static_cast<uint8_t>(move)) };
}

Position Play(const Position& pos, Field move, BitBoard flips)
{
	return { pos.O ^ flips, pos.P ^ flips ^ to_BitBoard(move) };
}

Position Play(const Position& pos, Field move)
{
	assert(pos.Empties() & (1ULL << static_cast<int>(move))); // move field is free.

	const auto flips = Flips(pos, move);

	assert(flips); // flips something.
	assert((pos.O & flips) == flips); // only flipping opponent stones.

	return Play(pos, move, flips);
}

Position PlayPass(const Position& pos)
{
	return { pos.O, pos.P };
}

Moves PossibleMoves(const Position& board)
{
	return Moves{ BitBoard{ PossibleMoves(board.P, board.O) } };
}

Field GetFirstDisc(BitBoard board)
{
	return static_cast<Field>(BitScanLSB(board));
}

BitBoard StableStones(const Position& pos)
{
	return StableStones(pos.P, pos.O);
}
