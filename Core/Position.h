#pragma once
#include "BitBoard.h"
#include <cstdint>
#include <cstddef>
#include <compare>

class Position;

class BitBoard_property
{
	friend class Position;
	BitBoard value = 0;

	constexpr BitBoard& operator=(const BitBoard& novum) { return value = novum; }
public:
	constexpr BitBoard_property() noexcept = default;
	constexpr BitBoard_property(const BitBoard& value) noexcept : value(value) {}

	[[nodiscard]] auto operator<=>(const BitBoard_property&) const noexcept = default;

	[[nodiscard]] constexpr operator BitBoard const& () const { return value; }
};

// A board where every field is either taken by a player or empty.
class Position
{
public:
	BitBoard_property P, O;

	constexpr Position() noexcept = default;
	constexpr Position(BitBoard P, BitBoard O) noexcept : P(P), O(O) { assert(Constrained(P, O)); }

	static Position Start();
	static Position StartETH();

	static constexpr Position TryCreate(BitBoard P, BitBoard O) noexcept(false)
	{
		if (Constrained(P, O))
			return { P, O };
		throw;
	}

	static constexpr bool Constrained(BitBoard P, BitBoard O) noexcept
	{
		return empty(P & O);
	}

	[[nodiscard]] auto operator<=>(const Position&) const noexcept = default;

	void FlipCodiagonal() noexcept;
	void FlipDiagonal() noexcept;
	void FlipHorizontal() noexcept;
	void FlipVertical() noexcept;
	void FlipToUnique() noexcept;

	BitBoard Empties() const { return ~(P.value | O.value); }
	std::size_t EmptyCount() const;
	
	uint64_t ParityQuadrants() const;
};

[[nodiscard]] Position FlipCodiagonal(Position pos) noexcept;
[[nodiscard]] Position FlipDiagonal(Position pos) noexcept;
[[nodiscard]] Position FlipHorizontal(Position pos) noexcept;
[[nodiscard]] Position FlipVertical(Position pos) noexcept;
[[nodiscard]] Position FlipToUnique(Position pos) noexcept;

[[nodiscard]] Position operator""_pos(const char*, std::size_t size);