#pragma once
#include "Base/Base.h"
#include "Field.h"
#include <cassert>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

class Position
{
	// A board where no field is taken by both players.

	uint64_t P{}, O{};
public:
	Position() noexcept = default;
	CUDA_CALLABLE constexpr Position(uint64_t P, uint64_t O) noexcept : P(P), O(O) { assert((P & O) == 0); }

	static Position FromString(std::string_view);
	static Position Start();

	CUDA_CALLABLE bool operator==(const Position& o) const noexcept { return P == o.P and O == o.O; }
	CUDA_CALLABLE bool operator!=(const Position& o) const noexcept { return P != o.P or O != o.O; }
	CUDA_CALLABLE bool operator<(const Position& o) const noexcept { return P < o.P or (P == o.P and O < o.O); }

	CUDA_CALLABLE uint64_t Player() const noexcept { return P; }
	CUDA_CALLABLE uint64_t Opponent() const noexcept { return O; }

	CUDA_CALLABLE uint64_t Discs() const noexcept { return P | O; }
	CUDA_CALLABLE uint64_t Empties() const noexcept { return ~Discs(); }
	CUDA_CALLABLE int EmptyCount() const noexcept { return std::popcount(Empties()); }
};

std::string SingleLine(const Position&);
std::string MultiLine(const Position&);
std::string to_string(const Position&);


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

CUDA_CALLABLE Position Play(const Position&, Field move, uint64_t flips) noexcept;
CUDA_CALLABLE Position Play(const Position&, Field move) noexcept;
CUDA_CALLABLE Position PlayPass(const Position&) noexcept;
CUDA_CALLABLE Position PlayOrPass(const Position&, Field move) noexcept;

Position FlippedCodiagonal(const Position&) noexcept;
Position FlippedDiagonal(const Position&) noexcept;
Position FlippedHorizontal(const Position&) noexcept;
Position FlippedVertical(const Position&) noexcept;
Position FlippedToUnique(const Position&) noexcept;
