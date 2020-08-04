#pragma once
#include "BitBoard.h"
#include "Score.h"
#include <cstdint>
#include <cstddef>
#include <compare>

class Position;

class BitBoard_property
{
	friend class Position;
	BitBoard value{};

	constexpr BitBoard& operator=(const BitBoard& novum) { return value = novum; }
public:
	constexpr BitBoard_property() noexcept = default;
	constexpr BitBoard_property(const BitBoard& value) noexcept : value(value) {}

	[[nodiscard]] auto operator<=>(const BitBoard_property&) const noexcept = default;

	[[nodiscard]] constexpr operator BitBoard const&() const { return value; }
	[[nodiscard]] constexpr operator uint64_t() const { return value; }

	[[nodiscard]] bool Get(Field f) const noexcept { return value.Get(f); }
	[[nodiscard]] bool Get(int x, int y) const noexcept { return value.Get(x, y); }
};

constexpr BitBoard operator&(const BitBoard_property& l, const BitBoard_property& r) noexcept { return static_cast<BitBoard>(l) & static_cast<BitBoard>(r); }
constexpr BitBoard operator|(const BitBoard_property& l, const BitBoard_property& r) noexcept { return static_cast<BitBoard>(l) | static_cast<BitBoard>(r); }
constexpr BitBoard operator^(const BitBoard_property& l, const BitBoard_property& r) noexcept { return static_cast<BitBoard>(l) ^ static_cast<BitBoard>(r); }
constexpr BitBoard operator&(const BitBoard_property& l, uint64_t r) noexcept { return static_cast<BitBoard>(l) & r; }
constexpr BitBoard operator|(const BitBoard_property& l, uint64_t r) noexcept { return static_cast<BitBoard>(l) | r; }
constexpr BitBoard operator^(const BitBoard_property& l, uint64_t r) noexcept { return static_cast<BitBoard>(l) ^ r; }
constexpr BitBoard operator&(const BitBoard_property& l, const BitBoard& r) noexcept { return static_cast<BitBoard>(l) & r; }
constexpr BitBoard operator|(const BitBoard_property& l, const BitBoard& r) noexcept { return static_cast<BitBoard>(l) | r; }
constexpr BitBoard operator^(const BitBoard_property& l, const BitBoard& r) noexcept { return static_cast<BitBoard>(l) ^ r; }
constexpr BitBoard operator&(uint64_t l, const BitBoard_property& r) noexcept { return l & static_cast<BitBoard>(r); }
constexpr BitBoard operator|(uint64_t l, const BitBoard_property& r) noexcept { return l | static_cast<BitBoard>(r); }
constexpr BitBoard operator^(uint64_t l, const BitBoard_property& r) noexcept { return l ^ static_cast<BitBoard>(r); }
constexpr BitBoard operator&(const BitBoard& l, const BitBoard_property& r) noexcept { return l & static_cast<BitBoard>(r); }
constexpr BitBoard operator|(const BitBoard& l, const BitBoard_property& r) noexcept { return l | static_cast<BitBoard>(r); }
constexpr BitBoard operator^(const BitBoard& l, const BitBoard_property& r) noexcept { return l ^ static_cast<BitBoard>(r); }


// A board where every field is either taken by exactly one player or is empty.
class Position
{
public:
	BitBoard_property P{}, O{};

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

	[[nodiscard]]
	constexpr auto operator<=>(const Position&) const noexcept = default;

	void FlipCodiagonal() noexcept;
	void FlipDiagonal() noexcept;
	void FlipHorizontal() noexcept;
	void FlipVertical() noexcept;
	void FlipToUnique() noexcept;

	BitBoard Discs() const { return P | O; }
	BitBoard Empties() const { return ~Discs(); }
	int EmptyCount() const;
	
	uint64_t ParityQuadrants() const;
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

Score EvalGameOver(const Position&);