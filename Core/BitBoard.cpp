#include "BitBoard.h"
#include "Bit.h"
#include <array>
#include <string>

void BitBoard::FlipCodiagonal() noexcept
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
	// / # # # # # # #<-LSB
	uint64_t
	t  =  b ^ (b << 36);
	b ^= (t ^ (b >> 36)) & 0xF0F0F0F00F0F0F0FULL;
	t  = (b ^ (b << 18)) & 0xCCCC0000CCCC0000ULL;
	b ^=  t ^ (t >> 18);
	t  = (b ^ (b <<  9)) & 0xAA00AA00AA00AA00ULL;
	b ^=  t ^ (t >>  9);
}

void BitBoard::FlipDiagonal() noexcept
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
	// # # # # # # # \<-LSB
	uint64_t 
	t  = (b ^ (b >>  7)) & 0x00AA00AA00AA00AAULL;
	b ^=  t ^ (t <<  7);
	t  = (b ^ (b >> 14)) & 0x0000CCCC0000CCCCULL;
	b ^=  t ^ (t << 14);
	t  = (b ^ (b >> 28)) & 0x00000000F0F0F0F0ULL;
	b ^=  t ^ (t << 28);
}

void BitBoard::FlipHorizontal() noexcept
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
	// # # # #|# # # #<-LSB
	b = ((b >> 1) & 0x5555555555555555ULL) | ((b << 1) & 0xAAAAAAAAAAAAAAAAULL);
	b = ((b >> 2) & 0x3333333333333333ULL) | ((b << 2) & 0xCCCCCCCCCCCCCCCCULL);
	b = ((b >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((b << 4) & 0xF0F0F0F0F0F0F0F0ULL);
}

void BitBoard::FlipVertical() noexcept
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
	// # # # # # # # #<-LSB
	b = BSwap(b);
}

bool BitBoard::IsCodiagonallySymmetric() const noexcept
{
	return *this == ::FlipCodiagonal(*this);
}

bool BitBoard::IsDiagonallySymmetric() const noexcept
{
	return *this == ::FlipDiagonal(*this);

}

bool BitBoard::IsHorizontallySymmetric() const noexcept
{
	return *this == ::FlipHorizontal(*this);

}

bool BitBoard::IsVerticallySymmetric() const noexcept
{
	return *this == ::FlipVertical(*this);

}

BitBoard FlipCodiagonal(BitBoard b) noexcept
{
	b.FlipCodiagonal();
	return b;
}

BitBoard FlipDiagonal(BitBoard b) noexcept
{
	b.FlipDiagonal();
	return b;
}

BitBoard FlipHorizontal(BitBoard b) noexcept
{
	b.FlipHorizontal();
	return b;
}

BitBoard FlipVertical(BitBoard b) noexcept
{
	b.FlipVertical();
	return b;
}

BitBoard FourNeighbours(BitBoard b) noexcept
{
	return (b >> 8) | (b << 8) | ((b << 1) & ~BitBoard::VerticalLine(0)) | ((b >> 1) & ~BitBoard::VerticalLine(7));
}

BitBoard FourNeighboursAndSelf(BitBoard b) noexcept
{
	return FourNeighbours(b) | b;
}

BitBoard EightNeighbours(BitBoard b) noexcept
{
	return EightNeighboursAndSelf(b) ^ b;
}

BitBoard EightNeighboursAndSelf(BitBoard b) noexcept
{
	b |= (b >> 8) | (b << 8);
	return b | ((b << 1) & ~BitBoard::VerticalLine(0)) | ((b >> 1) & ~BitBoard::VerticalLine(7));
}

std::string SingleLine(BitBoard bb)
{
	std::string str(64, '-');
	for (int i = 0; i < 64; i++)
		if (bb.Get(static_cast<Field>(63 - i)))
			str[i] = L'#';
	return str;
}

std::string MultiLine(BitBoard bb)
{
	std::string board =
		"  H G F E D C B A  \n"
		"8 - - - - - - - - 8\n"
		"7 - - - - - - - - 7\n"
		"6 - - - - - - - - 6\n"
		"5 - - - - - - - - 5\n"
		"4 - - - - - - - - 4\n"
		"3 - - - - - - - - 3\n"
		"2 - - - - - - - - 2\n"
		"1 - - - - - - - - 1\n"
		"  H G F E D C B A  ";

	for (int i = 0; i < 64; i++)
		if (bb.Get(static_cast<Field>(63 - i)))
			board[22 + 2 * i + 4 * (i / 8)] = '#';
	return board;
}