#pragma once
#include "Bit.h"
#include "Moves.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <string_view>

class Position
{
	// A board where no field is taken by both players.

	uint64_t P{}, O{};
public:
	constexpr Position() noexcept = default;
	CUDA_CALLABLE constexpr Position(uint64_t P, uint64_t O) noexcept : P(P), O(O) { assert((P & O) == 0); }

	static Position Start();

	CUDA_CALLABLE constexpr bool operator==(const Position& o) const noexcept { return P == o.P && O == o.O; }
	CUDA_CALLABLE constexpr bool operator!=(const Position& o) const noexcept { return P != o.P || O != o.O; }
	CUDA_CALLABLE constexpr bool operator<(const Position& o) const noexcept { return P < o.P || (P == o.P && O < o.O); }

	CUDA_CALLABLE constexpr uint64_t Player() const noexcept { return P; }
	CUDA_CALLABLE constexpr uint64_t Opponent() const noexcept { return O; }

	CUDA_CALLABLE uint64_t Discs() const noexcept { return P | O; }
	CUDA_CALLABLE uint64_t Empties() const noexcept { return ~Discs(); }
	CUDA_CALLABLE int EmptyCount() const noexcept { return std::popcount(Empties()); }
};

std::string SingleLine(const Position&);
std::string MultiLine(const Position&);
std::string to_string(const Position&);

Position PositionFromString(std::string_view);

constexpr Position operator""_pos(const char* c, std::size_t size)
{
	assert(size == 120);

	uint64_t P{0};
	uint64_t O{0};
	for (int j = 0; j < 8; j++)
		for (int i = 0; i < 8; i++)
		{
			char symbol = c[119 - 15 * j - 2 * i];
			if (symbol == 'X')
				P |= 1ULL << (i + 8 * j);
			if (symbol == 'O')
				O |= 1ULL << (i + 8 * j);
		}
	return { P, O };
}

CUDA_CALLABLE int EndScore(const Position&) noexcept;

CUDA_CALLABLE Position Play(const Position&, Field move, uint64_t flips) noexcept;
CUDA_CALLABLE Position Play(const Position&, Field move) noexcept;

CUDA_CALLABLE Position PlayPass(const Position&) noexcept;

CUDA_CALLABLE Position PlayOrPass(const Position&, Field move) noexcept;

CUDA_CALLABLE uint64_t Flips(const Position&, Field move) noexcept;

int CountLastFlip(const Position&, Field move) noexcept;

CUDA_CALLABLE Moves PossibleMoves(const Position&) noexcept;

namespace detail
{
	#ifdef __AVX512F__
		Moves PossibleMoves_AVX512(const Position&) noexcept;
	#endif

	#ifdef __AVX2__
		Moves PossibleMoves_AVX2(const Position&) noexcept;
	#endif

	CUDA_CALLABLE Moves PossibleMoves_x64(const Position&) noexcept;
}

Position FlippedCodiagonal(const Position&) noexcept;
Position FlippedDiagonal(const Position&) noexcept;
Position FlippedHorizontal(const Position&) noexcept;
Position FlippedVertical(const Position&) noexcept;
Position FlippedToUnique(Position) noexcept;