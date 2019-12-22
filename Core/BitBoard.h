#pragma once
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <compare>

enum class Field : uint8_t
{
	A1, B1, C1, D1, E1, F1, G1, H1,
	A2, B2, C2, D2, E2, F2, G2, H2,
	A3, B3, C3, D3, E3, F3, G3, H3,
	A4, B4, C4, D4, E4, F4, G4, H4,
	A5, B5, C5, D5, E5, F5, G5, H5,
	A6, B6, C6, D6, E6, F6, G6, H6,
	A7, B7, C7, D7, E7, F7, G7, H7,
	A8, B8, C8, D8, E8, F8, G8, H8,
	invalid
};

class BitBoard; // forward declaration

class Bit_
{
	BitBoard& bb;
	Field f;

public:
	constexpr Bit_(BitBoard& bb, Field f) noexcept : bb(bb), f(f) {}

	constexpr Bit_& operator=(bool x) noexcept;
	Bit_& operator=(const Bit_&) noexcept = default;

	constexpr operator bool() const noexcept;
};

class BitBoard
{
	uint64_t b = 0;

public:
	using reference = Bit_;

	constexpr BitBoard() noexcept = default;
	constexpr BitBoard(uint64_t b) noexcept : b(b) {}
	constexpr explicit BitBoard(Field f) noexcept : b(1ULL << static_cast<uint8_t>(f)) { assert(f != Field::invalid); }

	constexpr operator uint64_t() const noexcept { return b; }

	[[nodiscard]] auto operator<=>(const BitBoard&) const noexcept = default;

	constexpr BitBoard& operator&=(BitBoard o) noexcept { b &= o.b; return *this; }
	constexpr BitBoard& operator|=(BitBoard o) noexcept { b |= o.b; return *this; }
	constexpr BitBoard& operator^=(BitBoard o) noexcept { b ^= o.b; return *this; }

	friend constexpr BitBoard operator&(BitBoard l, BitBoard r) noexcept { return l &= r; }
	friend constexpr BitBoard operator|(BitBoard l, BitBoard r) noexcept { return l |= r; }
	friend constexpr BitBoard operator^(BitBoard l, BitBoard r) noexcept { return l ^= r; }

	friend constexpr BitBoard operator&(uint64_t l, BitBoard r) noexcept { return BitBoard(l) &= r; }
	friend constexpr BitBoard operator|(uint64_t l, BitBoard r) noexcept { return BitBoard(l) |= r; }
	friend constexpr BitBoard operator^(uint64_t l, BitBoard r) noexcept { return BitBoard(l) ^= r; }

	friend constexpr BitBoard operator&(BitBoard l, uint64_t r) noexcept { return l &= BitBoard(r); }
	friend constexpr BitBoard operator|(BitBoard l, uint64_t r) noexcept { return l |= BitBoard(r); }
	friend constexpr BitBoard operator^(BitBoard l, uint64_t r) noexcept { return l ^= BitBoard(r); }

	[[nodiscard]] constexpr BitBoard operator~() const noexcept { return ~b; }
	
	constexpr bool operator[](Field f) const noexcept { return (*this & BitBoard(f)); }
	constexpr bool operator[](std::size_t i) const noexcept { assert(i < 64); return this->operator[](static_cast<Field>(i)); }
	constexpr reference operator[](Field f) noexcept { return { *this, f }; }
	constexpr reference operator[](std::size_t i) noexcept { assert(i < 64); return this->operator[](static_cast<Field>(i)); }


	[[nodiscard]] constexpr bool IsSubsetOf(BitBoard o) const noexcept { return (o.b & b) == b; }

	[[nodiscard]] constexpr bool empty() const noexcept { return !b; }

	[[nodiscard]] std::size_t PopCount() const noexcept;
	void RemoveFirstField() noexcept;
	[[nodiscard]] Field FirstField() const noexcept;

	void FlipCodiagonal() noexcept;
	void FlipDiagonal() noexcept;
	void FlipHorizontal() noexcept;
	void FlipVertical() noexcept;

	constexpr static BitBoard Middle();
	constexpr static BitBoard Edge();
};

[[nodiscard]]
inline std::size_t PopCount(BitBoard bb) noexcept { return bb.PopCount(); }
//
//inline void RemoveFirstField(BitBoard& bb) noexcept { bb.RemoveFirstField(); }
//
//[[nodiscard]]
//inline Field FirstField(const BitBoard& bb) noexcept { return bb.FirstField(); }

[[nodiscard]] BitBoard FlipCodiagonal(BitBoard) noexcept;
[[nodiscard]] BitBoard FlipDiagonal(BitBoard) noexcept;
[[nodiscard]] BitBoard FlipHorizontal(BitBoard) noexcept;
[[nodiscard]] BitBoard FlipVertical(BitBoard) noexcept;


constexpr BitBoard operator""_BitBoard(const char* c, std::size_t size)
{
	assert(size == 120);

	BitBoard bb;
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
		{
			char symbol = c[119 - 2 * j - 15 * i];
			bb[static_cast<Field>(i * 8 + j)] = (symbol != ' ');
		}
	return bb;
}

constexpr Bit_& Bit_::operator=(bool x) noexcept
{
	if (x)
		bb |= BitBoard(f);
	else
		bb &= ~BitBoard(f);
	return *this;
}

constexpr Bit_::operator bool() const noexcept
{
	return std::as_const(bb)[f];
}

constexpr BitBoard BitBoard::Middle()
{
	return
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - # # - - -"
		"- - - # # - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_BitBoard;
}

constexpr BitBoard BitBoard::Edge()
{
	return
		"# # # # # # # #"
		"# - - - - - - #"
		"# - - - - - - #"
		"# - - - - - - #"
		"# - - - - - - #"
		"# - - - - - - #"
		"# - - - - - - #"
		"# # # # # # # #"_BitBoard;
}