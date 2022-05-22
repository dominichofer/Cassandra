#pragma once
#include <cassert>
#include <cstdint>
#include <string>
#include <ostream>
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

// Field::A1 -> "A1"
// etc
// Field::invalid -> "--"
std::string to_string(Field) noexcept;

inline std::ostream& operator<<(std::ostream& os, Field f) { return os << to_string(f); }


// An 8x8 board of bits.
class BitBoard
{
	uint64 b{};
public:
	constexpr BitBoard() noexcept = default;
	CUDA_CALLABLE constexpr BitBoard(uint64 b) noexcept : b(b) {}
	CUDA_CALLABLE constexpr explicit BitBoard(Field f) noexcept : BitBoard(1ULL << static_cast<uint8>(f)) {}

	static constexpr BitBoard Bit(int x, int y) noexcept { return BitBoard{ 1ULL << (x + y * 8) }; }

	static constexpr BitBoard All() noexcept { return BitBoard{ 0xFFFFFFFFFFFFFFFFULL }; }
	static constexpr BitBoard Empty() noexcept { return BitBoard{ 0 }; }

	static constexpr BitBoard HorizontalLine(int i) noexcept { return BitBoard{ 0xFFULL << (i * 8) }; }
	static constexpr BitBoard VerticalLine(int i) noexcept { return BitBoard{ 0x0101010101010101ULL << i }; }
	static constexpr BitBoard UpperDiagonalLine(int i) noexcept { return BitBoard{ 0x8040201008040201ULL << (i * 8) }; }
	static constexpr BitBoard LowerDiagonalLine(int i) noexcept { return BitBoard{ 0x8040201008040201ULL >> (i * 8) }; }
	static constexpr BitBoard UpperCodiagonalLine(int i) noexcept { return BitBoard{ 0x0102040810204080ULL << (i * 8) }; }
	static constexpr BitBoard LowerCodiagonalLine(int i) noexcept { return BitBoard{ 0x0102040810204080ULL >> (i * 8) }; }
	static constexpr BitBoard DiagonalLine(int i) noexcept { return (i > 0) ? UpperDiagonalLine(i) : LowerDiagonalLine(-i); }
	static constexpr BitBoard CodiagonalLine(int i) noexcept { return (i > 0) ? UpperCodiagonalLine(i) : LowerCodiagonalLine(-i); }

	static constexpr BitBoard HorizontalLine(Field f) noexcept { return BitBoard{ 0xFFULL << (static_cast<uint8>(f) & 0xF8) }; }
	static constexpr BitBoard VerticalLine(Field f) noexcept { return VerticalLine(static_cast<uint8>(f) % 8); }
	static constexpr BitBoard DiagonalLine(Field f) noexcept { return DiagonalLine((static_cast<uint8>(f) / 8) - (static_cast<uint8>(f) % 8)); }
	static constexpr BitBoard CodiagonalLine(Field f) noexcept { return CodiagonalLine((static_cast<uint8>(f) / 8) + (static_cast<uint8>(f) % 8) - 7); }

	static constexpr BitBoard Corners() noexcept { return BitBoard{ 0x8100000000000081ULL }; }
	static constexpr BitBoard Edges() noexcept { return BitBoard{ 0xFF818181818181FFULL }; }
	static constexpr BitBoard LeftHalf() noexcept { return BitBoard{ 0xF0F0F0F0F0F0F0F0ULL }; }
	static constexpr BitBoard RightHalf() noexcept { return BitBoard{ 0x0F0F0F0F0F0F0F0FULL }; }
	static constexpr BitBoard UpperHalf() noexcept { return BitBoard{ 0xFFFFFFFF00000000ULL }; }
	static constexpr BitBoard LowerHalf() noexcept { return BitBoard{ 0x00000000FFFFFFFFULL }; }
	static constexpr BitBoard StrictlyLeftUpper() noexcept { return BitBoard{ 0xFEFCF8F0E0C08000ULL }; }
	static constexpr BitBoard StrictlyLeftLower() noexcept { return BitBoard{ 0x0080C0E0F0F8FCFEULL }; }
	static constexpr BitBoard StrictlyRighUpper() noexcept { return BitBoard{ 0x7F3F1F0F07030100ULL }; }
	static constexpr BitBoard StrictlyRighLower() noexcept { return BitBoard{ 0x000103070F1F3F7FULL }; }

	CUDA_CALLABLE constexpr operator uint64() const noexcept { return b; }

	CUDA_CALLABLE constexpr bool empty() const noexcept { return b == 0; }

	CUDA_CALLABLE constexpr BitBoard operator~() const noexcept { return  BitBoard{ ~b }; }
	BitBoard& operator&=(BitBoard o) noexcept { b &= o.b; return *this; }
	BitBoard& operator|=(BitBoard o) noexcept { b |= o.b; return *this; }
	BitBoard& operator^=(BitBoard o) noexcept { b ^= o.b; return *this; }
	BitBoard& operator&=(uint64 o) noexcept { b &= o; return *this; }
	BitBoard& operator|=(uint64 o) noexcept { b |= o; return *this; }
	BitBoard& operator^=(uint64 o) noexcept { b ^= o; return *this; }
	friend CUDA_CALLABLE constexpr BitBoard operator&(BitBoard l, BitBoard r) noexcept { return BitBoard{ l.b & r.b }; }
	friend CUDA_CALLABLE constexpr BitBoard operator|(BitBoard l, BitBoard r) noexcept { return BitBoard{ l.b | r.b }; }
	friend CUDA_CALLABLE constexpr BitBoard operator^(BitBoard l, BitBoard r) noexcept { return BitBoard{ l.b ^ r.b }; }
	friend CUDA_CALLABLE constexpr BitBoard operator&(uint64 l, BitBoard r) noexcept { return BitBoard{ l & r.b }; }
	friend CUDA_CALLABLE constexpr BitBoard operator|(uint64 l, BitBoard r) noexcept { return BitBoard{ l | r.b }; }
	friend CUDA_CALLABLE constexpr BitBoard operator^(uint64 l, BitBoard r) noexcept { return BitBoard{ l ^ r.b }; }
	friend CUDA_CALLABLE constexpr BitBoard operator&(BitBoard l, uint64 r) noexcept { return BitBoard{ l.b & r }; }
	friend CUDA_CALLABLE constexpr BitBoard operator|(BitBoard l, uint64 r) noexcept { return BitBoard{ l.b | r }; }
	friend CUDA_CALLABLE constexpr BitBoard operator^(BitBoard l, uint64 r) noexcept { return BitBoard{ l.b ^ r }; }

	CUDA_CALLABLE constexpr bool operator==(BitBoard o) const noexcept { return b == o.b; }
	CUDA_CALLABLE constexpr bool operator!=(BitBoard o) const noexcept { return b != o.b; }

	CUDA_CALLABLE bool Get(Field f) const noexcept { return b & (1ULL << static_cast<uint8>(f)); }
	CUDA_CALLABLE bool Get(int x, int y) const noexcept { return b & (1ULL << (x + 8 * y)); }

	constexpr void Set(Field f) noexcept { b |= 1ULL << static_cast<uint8>(f); }
	constexpr void Set(int x, int y) noexcept { b |= 1ULL << (x + 8 * y); }

	void Clear(Field f) noexcept { b &= ~(1ULL << static_cast<uint8>(f)); }
	void Clear(int x, int y) noexcept { b &= ~(1ULL << (x + 8 * y)); }

	CUDA_CALLABLE BitBoard FirstSet() const noexcept { return BitBoard{ GetLSB(b) }; }
	CUDA_CALLABLE Field FirstSetField() const noexcept { return static_cast<Field>(std::countr_zero(b)); }
	CUDA_CALLABLE void ClearFirstSet() noexcept { RemoveLSB(b); }

	void FlipCodiagonal() noexcept;
	void FlipDiagonal() noexcept;
	void FlipHorizontal() noexcept;
	void FlipVertical() noexcept;
};

BitBoard FlipCodiagonal(BitBoard) noexcept;
BitBoard FlipDiagonal(BitBoard) noexcept;
BitBoard FlipHorizontal(BitBoard) noexcept;
BitBoard FlipVertical(BitBoard) noexcept;

BitBoard FourNeighbours(BitBoard) noexcept;
BitBoard FourNeighboursAndSelf(BitBoard) noexcept;
BitBoard EightNeighbours(BitBoard) noexcept;
BitBoard EightNeighboursAndSelf(BitBoard) noexcept;

CUDA_CALLABLE inline int countl_zero(BitBoard b) noexcept { return std::countl_zero(static_cast<uint64>(b)); }
CUDA_CALLABLE inline int countl_one(BitBoard b) noexcept { return std::countl_one(static_cast<uint64>(b)); }
CUDA_CALLABLE inline int countr_zero(BitBoard b) noexcept { return std::countr_zero(static_cast<uint64>(b)); }
CUDA_CALLABLE inline int countr_one(BitBoard b) noexcept { return std::countr_one(static_cast<uint64>(b)); }
CUDA_CALLABLE inline int popcount(BitBoard b) noexcept { return std::popcount(static_cast<uint64>(b)); }

std::string SingleLine(BitBoard);
std::string MultiLine(BitBoard);
inline std::string to_string(BitBoard bb) { return SingleLine(bb); }
inline std::ostream& operator<<(std::ostream& os, BitBoard bb) { return os << to_string(bb); }

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
