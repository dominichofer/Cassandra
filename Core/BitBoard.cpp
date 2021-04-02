#include "BitBoard.h"
#include "Bit.h"
#include <array>

std::string to_string(Field f) noexcept
{
	static const std::array<std::string, 65> field_names = {
		"A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1",
		"A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2",
		"A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3",
		"A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4",
		"A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5",
		"A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6",
		"A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7",
		"A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8", "--"
	};
	return field_names[static_cast<uint8_t>(f)];
}

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

std::string SingleLine(const BitBoard& bb)
{
	std::string str(64, '-');
	for (int i = 0; i < 64; i++)
		if (bb.Get(static_cast<Field>(63 - i)))
			str[i] = L'#';
	return str;
}

std::string MultiLine(const BitBoard& bb)
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