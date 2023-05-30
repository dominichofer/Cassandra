#pragma once
#include <cassert>
#include <cstdint>
#include <string>
#include "Algorithms.h"
#include "Bit.h"
#include "Field.h"


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
	static constexpr BitBoard RightLowerQuadrant() noexcept { return BitBoard{ 0x000000000F0F0F0FULL }; }
	static constexpr BitBoard LeftLowerQuadrant() noexcept { return BitBoard{ 0x00000000F0F0F0F0ULL }; }
	static constexpr BitBoard RightUpperQuadrant() noexcept { return BitBoard{ 0x0F0F0F0F00000000ULL }; }
	static constexpr BitBoard LeftUpperQuadrant() noexcept { return BitBoard{ 0xF0F0F0F000000000ULL }; }
	static constexpr BitBoard StrictlyLeftUpperTriangle() noexcept { return BitBoard{ 0xFEFCF8F0E0C08000ULL }; }
	static constexpr BitBoard StrictlyLeftLowerTriangle() noexcept { return BitBoard{ 0x0080C0E0F0F8FCFEULL }; }
	static constexpr BitBoard StrictlyRighUpperTriangle() noexcept { return BitBoard{ 0x7F3F1F0F07030100ULL }; }
	static constexpr BitBoard StrictlyRighLowerTriangle() noexcept { return BitBoard{ 0x000103070F1F3F7FULL }; }

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

	bool IsCodiagonallySymmetric() const noexcept;
	bool IsDiagonallySymmetric() const noexcept;
	bool IsHorizontallySymmetric() const noexcept;
	bool IsVerticallySymmetric() const noexcept;
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
