#include "BitBoard.h"
#include "Bit.h"

uint64_t FlippedCodiagonal(uint64_t b) noexcept
{
	// 9 x XOR, 6 x SHIFT, 3 x AND
	// 18 OPs

	// # # # # # # # /
	// # # # # # # / #
	// # # # # # / # #
	// # # # # / # # #
	// # # # / # # # #
	// # # / # # # # #
	// # / # # # # # #
	// / # # # # # # # <-LSB
	uint64_t t;
	t  =  b ^ (b << 36);
	b ^= (t ^ (b >> 36)) & 0xF0F0F0F00F0F0F0FULL;
	t  = (b ^ (b << 18)) & 0xCCCC0000CCCC0000ULL;
	b ^=  t ^ (t >> 18);
	t  = (b ^ (b <<  9)) & 0xAA00AA00AA00AA00ULL;
	b ^=  t ^ (t >>  9);
	return b;
}

uint64_t FlippedDiagonal(uint64_t b) noexcept
{
	// 9 x XOR, 6 x SHIFT, 3 x AND
	// 18 OPs

	// \ # # # # # # #
	// # \ # # # # # #
	// # # \ # # # # #
	// # # # \ # # # #
	// # # # # \ # # #
	// # # # # # \ # #
	// # # # # # # \ #
	// # # # # # # # \ <-LSB
	uint64_t t;
	t  = (b ^ (b >>  7)) & 0x00AA00AA00AA00AAULL;
	b ^=  t ^ (t <<  7);
	t  = (b ^ (b >> 14)) & 0x0000CCCC0000CCCCULL;
	b ^=  t ^ (t << 14);
	t  = (b ^ (b >> 28)) & 0x00000000F0F0F0F0ULL;
	b ^=  t ^ (t << 28);
	return b;
}

uint64_t FlippedHorizontal(uint64_t b) noexcept
{
	// 6 x SHIFT, 6 x AND, 3 x OR
	// 15 OPs

	// # # # #|# # # #
	// # # # #|# # # #
	// # # # #|# # # #
	// # # # #|# # # #
	// # # # #|# # # #
	// # # # #|# # # #
	// # # # #|# # # #
	// # # # #|# # # # <-LSB
	b = ((b >> 1) & 0x5555555555555555ULL) | ((b << 1) & 0xAAAAAAAAAAAAAAAAULL);
	b = ((b >> 2) & 0x3333333333333333ULL) | ((b << 2) & 0xCCCCCCCCCCCCCCCCULL);
	b = ((b >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((b << 4) & 0xF0F0F0F0F0F0F0F0ULL);
	return b;
}

uint64_t FlippedVertical(uint64_t b) noexcept
{
	// 1 x ByteSwap
	// 1 OP

	// # # # # # # # #
	// # # # # # # # #
	// # # # # # # # #
	// # # # # # # # #
	// ---------------
	// # # # # # # # #
	// # # # # # # # #
	// # # # # # # # #
	// # # # # # # # # <-LSB
	return BSwap(b);
}

bool IsCodiagonallySymmetric(uint64_t b) noexcept
{
	return FlippedCodiagonal(b) == b;
}

bool IsDiagonallySymmetric(uint64_t b) noexcept
{
	return FlippedDiagonal(b) == b;
}

bool IsHorizontallySymmetric(uint64_t b) noexcept
{
	return FlippedHorizontal(b) == b;
}

bool IsVerticallySymmetric(uint64_t b) noexcept
{
	return FlippedVertical(b) == b;
}


uint64_t ParityQuadrants(uint64_t b) noexcept
{
	// 4 x SHIFT, 4 x XOR, 1 x AND, 1 x MUL
	// = 10 OPs
	b ^= b >> 1;
	b ^= b >> 2;
	b ^= b >> 8;
	b ^= b >> 16;
	b &= 0x0000'0011'0000'0011ULL;
	return b * 0x0000'0000'0F0F'0F0FULL;
}

uint64_t EightNeighboursAndSelf(uint64_t b) noexcept
{
	b |= (b >> 8) | (b << 8);
	return b | ((b << 1) & 0xFEFEFEFEFEFEFEFEULL) | ((b >> 1) & 0x7F7F7F7F7F7F7F7FULL);
}