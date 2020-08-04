#include "Position.h"
#include "Bit.h"
#include <algorithm>

void Position::FlipCodiagonal() noexcept { P = ::FlipCodiagonal(P); O = ::FlipCodiagonal(O); }
void Position::FlipDiagonal  () noexcept { P = ::FlipDiagonal  (P); O = ::FlipDiagonal  (O); }
void Position::FlipHorizontal() noexcept { P = ::FlipHorizontal(P); O = ::FlipHorizontal(O); }
void Position::FlipVertical  () noexcept { P = ::FlipVertical  (P); O = ::FlipVertical  (O); }

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


int Position::EmptyCount() const
{
	return popcount(Empties());
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

Score EvalGameOver(const Position& pos)
{
	const auto Ps = popcount(pos.P);
	const auto Os = popcount(pos.O);
	if (Ps > Os)
		return 64 - 2 * Os;
	if (Ps < Os)
		return 2 * Ps - 64;
	return 0;
}