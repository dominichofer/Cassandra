#pragma once
#include <intrin.h>
#include <bit>
#include <cassert>
#include <cstdint>
#include <compare>
#include "Bit.h"

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

// An 8x8 board of binary integers.
class BitBoard
{
	uint64_t b{};
public:
	constexpr BitBoard() noexcept = default;
	constexpr BitBoard(uint64_t b) noexcept : b(b) {}
	constexpr explicit BitBoard(Field f) noexcept : BitBoard(1ULL << static_cast<int>(f)) {}

	//constexpr static BitBoard HorizontalLine(int i) noexcept { return 0xFFULL << (i * 8); }
	//constexpr static BitBoard VerticalLine(int i) noexcept { return 0x0101010101010101ULL << i; }
	//constexpr static BitBoard Edges() noexcept { return 0xFF818181818181FFULL; }

	constexpr operator uint64_t() const noexcept { return b; }

	constexpr BitBoard operator~() const noexcept { return ~b; }
	BitBoard& operator&=(const BitBoard& o) noexcept { b &= o.b; return *this; }
	BitBoard& operator|=(const BitBoard& o) noexcept { b |= o.b; return *this; }
	BitBoard& operator^=(const BitBoard& o) noexcept { b ^= o.b; return *this; }
	BitBoard& operator&=(uint64_t o) noexcept { b &= o; return *this; }
	BitBoard& operator|=(uint64_t o) noexcept { b |= o; return *this; }
	BitBoard& operator^=(uint64_t o) noexcept { b ^= o; return *this; }
	friend constexpr BitBoard operator&(const BitBoard& l, const BitBoard& r) noexcept { return l.b & r.b; }
	friend constexpr BitBoard operator|(const BitBoard& l, const BitBoard& r) noexcept { return l.b | r.b; }
	friend constexpr BitBoard operator^(const BitBoard& l, const BitBoard& r) noexcept { return l.b ^ r.b; }
	friend constexpr BitBoard operator&(uint64_t l, const BitBoard& r) noexcept { return l & r.b; }
	friend constexpr BitBoard operator|(uint64_t l, const BitBoard& r) noexcept { return l | r.b; }
	friend constexpr BitBoard operator^(uint64_t l, const BitBoard& r) noexcept { return l ^ r.b; }
	friend constexpr BitBoard operator&(const BitBoard& l, uint64_t r) noexcept { return l.b & r; }
	friend constexpr BitBoard operator|(const BitBoard& l, uint64_t r) noexcept { return l.b | r; }
	friend constexpr BitBoard operator^(const BitBoard& l, uint64_t r) noexcept { return l.b ^ r; }

	[[nodiscard]] auto operator<=>(const BitBoard&) const noexcept = default;

	[[nodiscard]] bool Get(Field f) const noexcept { return b & (1ULL << static_cast<int>(f)); }
	[[nodiscard]] bool Get(int x, int y) const noexcept { return b & (1ULL << (x + 8 * y)); }

	constexpr void Set(Field f) noexcept { b |= 1ULL << static_cast<int>(f); }
	constexpr void Set(int x, int y) noexcept { b |= 1ULL << (x + 8 * y); }

	void Clear(Field f) noexcept { b &= ~(1ULL << static_cast<int>(f)); }
	void Clear(int x, int y) noexcept { b &= ~(1ULL << (x + 8 * y)); }

	[[nodiscard]] constexpr bool empty() const noexcept { return b == 0; }

	[[nodiscard]] Field FirstSet() const noexcept { return static_cast<Field>(countr_zero(b)); }
	void ClearFirstSet() noexcept { RemoveLSB(b); }

	void FlipCodiagonal() noexcept;
	void FlipDiagonal() noexcept;
	void FlipHorizontal() noexcept;
	void FlipVertical() noexcept;
};

[[nodiscard]] BitBoard FlipCodiagonal(BitBoard) noexcept;
[[nodiscard]] BitBoard FlipDiagonal(BitBoard) noexcept;
[[nodiscard]] BitBoard FlipHorizontal(BitBoard) noexcept;
[[nodiscard]] BitBoard FlipVertical(BitBoard) noexcept;

//constexpr int countl_zero(const BitBoard& b) noexcept { return std::countl_zero(static_cast<uint64_t>(b)); }
//constexpr int countl_one(const BitBoard& b) noexcept { return std::countl_one(static_cast<uint64_t>(b)); }
//constexpr int countr_zero(const BitBoard& b) noexcept { return std::countr_zero(static_cast<uint64_t>(b)); }
//constexpr int countr_one(const BitBoard& b) noexcept { return std::countr_one(static_cast<uint64_t>(b)); }
//constexpr int popcount(const BitBoard& b) noexcept { return __builtin_popcountll(static_cast<uint64_t>(b)); }

constexpr BitBoard operator""_BitBoard(const char* c, std::size_t size)
{
	assert(size == 120);

	BitBoard bb;
	for (int y = 0; y < 8; y++)
		for (int x = 0; x < 8; x++)
		{
			char symbol = c[119 - 15 * y - 2 * x];
			if (symbol != '-')
				bb.Set(x, y);
		}
	return bb;
}
