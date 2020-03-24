#include "Position.h"
#include "Machine/BitTwiddling.h"
#include <algorithm>

void Position::FlipCodiagonal() noexcept { P.value = ::FlipCodiagonal(P.value); O.value = ::FlipCodiagonal(O.value); }
void Position::FlipDiagonal  () noexcept { P.value = ::FlipDiagonal  (P.value); O.value = ::FlipDiagonal  (O.value); }
void Position::FlipHorizontal() noexcept { P.value = ::FlipHorizontal(P.value); O.value = ::FlipHorizontal(O.value); }
void Position::FlipVertical  () noexcept { P.value = ::FlipVertical  (P.value); O.value = ::FlipVertical  (O.value); }

void Position::FlipToUnique() noexcept
{
	Position candidate = *this;
	Position min = candidate;
	candidate.FlipVertical();		if (candidate < min) min = std::min(min, candidate);
	candidate.FlipHorizontal();		if (candidate < min) min = std::min(min, candidate);
	candidate.FlipVertical();		if (candidate < min) min = std::min(min, candidate);
	candidate.FlipCodiagonal();		if (candidate < min) min = std::min(min, candidate);
	candidate.FlipVertical();		if (candidate < min) min = std::min(min, candidate);
	candidate.FlipHorizontal();		if (candidate < min) min = std::min(min, candidate);
	candidate.FlipVertical();		if (candidate < min) min = std::min(min, candidate);
	*this = min;
}

Position Position::Start()
{
	return
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - O X - - -"
		"- - - X O - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pos;
}

Position Position::StartETH()
{
	return
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - X X - - -"
		"- - - O O - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pos;
}


std::size_t Position::EmptyCount() const
{
	return PopCount(Empties());
}

uint64_t Position::ParityQuadrants() const
{
	// 4 x SHIFT, 4 x XOR, 1 x AND, 1 x NOT, 1x OR, 1 x MUL
	// = 12 OPs
	uint64_t E = Empties();
	E ^= E >> 1;
	E ^= E >> 2;
	E ^= E >> 8;
	E ^= E >> 16;
	E &= 0x0000'0011'0000'0011ULL;
	return E * 0x0000'0000'0F0F'0F0FULL;
}

Position operator""_pos(const char* c, std::size_t size)
{
	assert(size == 120);

	BitBoard P(0);
	BitBoard O(0);
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
		{
			char symbol = c[119 - 2 * j - 15 * i];
			if (symbol == 'X')
				Bit(P, i * 8 + j) = true;
			if (symbol == 'O')
				Bit(O, i * 8 + j) = true;
		}
	return { P, O };
}

Position FlipCodiagonal(Position pos) noexcept
{
	pos.FlipCodiagonal();
	return pos;
}

Position FlipDiagonal(Position pos) noexcept
{
	pos.FlipDiagonal();
	return pos;
}

Position FlipHorizontal(Position pos) noexcept
{
	pos.FlipHorizontal();
	return pos;
}

Position FlipVertical(Position pos) noexcept
{
	pos.FlipVertical();
	return pos;
}

Position FlipToUnique(Position pos) noexcept
{
	pos.FlipToUnique();
	return pos;
}
