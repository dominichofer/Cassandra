#pragma once
#include "BitBoard.h"
#include "Moves.h"
#include <cstdint>
#include <cstddef>
#include <compare>

using Score = int;

constexpr Score min_score{ -64 };
constexpr Score max_score{ +64 };
constexpr Score infinity{ +66 };

// A board where every field is either taken by exactly one player or is empty.
class Position
{
	BitBoard P{}, O{};
public:
	constexpr Position() noexcept = default;
	constexpr Position(BitBoard P, BitBoard O) noexcept : P(P), O(O) { assert((P & O).empty()); }

	static Position Start();
	static Position StartETH();

	static constexpr Position From(BitBoard P, BitBoard O) noexcept(false)
	{
		if ((P & O).empty())
			return { P, O };
		throw;
	}

	[[nodiscard]] constexpr auto operator<=>(const Position&) const noexcept = default;

	[[nodiscard]] BitBoard Player() const noexcept { return P; }
	[[nodiscard]] BitBoard Opponent() const noexcept { return O; }

	void FlipCodiagonal() noexcept;
	void FlipDiagonal() noexcept;
	void FlipHorizontal() noexcept;
	void FlipVertical() noexcept;
	void FlipToUnique() noexcept;

	[[nodiscard]] BitBoard Discs() const { return P | O; }
	[[nodiscard]] BitBoard Empties() const { return ~Discs(); }
	[[nodiscard]] int EmptyCount() const { return popcount(Empties()); }

	[[nodiscard]] BitBoard ParityQuadrants() const;
};

[[nodiscard]] Position FlipCodiagonal(Position) noexcept;
[[nodiscard]] Position FlipDiagonal(Position) noexcept;
[[nodiscard]] Position FlipHorizontal(Position) noexcept;
[[nodiscard]] Position FlipVertical(Position) noexcept;
[[nodiscard]] Position FlipToUnique(Position) noexcept;

[[nodiscard]]
constexpr Position operator""_pos(const char* c, std::size_t size)
{
	assert(size == 120);

	BitBoard P{0}, O{0};
	for (int j = 0; j < 8; j++)
		for (int i = 0; i < 8; i++)
		{
			char symbol = c[119 - 15 * j - 2 * i];
			if (symbol == 'X')
				P.Set(i, j);
			if (symbol == 'O')
				O.Set(i, j);
		}
	return { P, O };
}

[[nodiscard]]
Score EvalGameOver(const Position&);
