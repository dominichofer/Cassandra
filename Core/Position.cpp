#include "Position.h"
#include "Format.h"

void Position::FlipCodiagonal() noexcept { P.FlipCodiagonal(); O.FlipCodiagonal(); }
void Position::FlipDiagonal  () noexcept { P.FlipDiagonal  (); O.FlipDiagonal  (); }
void Position::FlipHorizontal() noexcept { P.FlipHorizontal(); O.FlipHorizontal(); }
void Position::FlipVertical  () noexcept { P.FlipVertical  (); O.FlipVertical  (); }

void Position::FlipToUnique() noexcept
{
	Position candidate = *this;
	Position min = candidate;
	candidate.FlipVertical();		if (candidate < min) min = candidate;
	candidate.FlipHorizontal();		if (candidate < min) min = candidate;
	candidate.FlipVertical();		if (candidate < min) min = candidate;
	candidate.FlipCodiagonal();		if (candidate < min) min = candidate;
	candidate.FlipVertical();		if (candidate < min) min = candidate;
	candidate.FlipHorizontal();		if (candidate < min) min = candidate;
	candidate.FlipVertical();		if (candidate < min) min = candidate;
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

BitBoard Position::ParityQuadrants() const
{
	// 4 x SHIFT, 4 x XOR, 1 x AND, 1 x NOT, 1x OR, 1 x MUL
	// = 12 OPs
	BitBoard E = Empties();
	E ^= E >> 1;
	E ^= E >> 2;
	E ^= E >> 8;
	E ^= E >> 16;
	E &= 0x0000'0011'0000'0011ULL;
	return BitBoard{ E * 0x0000'0000'0F0F'0F0FULL };
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

std::string SingleLine(const Position& pos)
{
	std::string str = "---------------------------------------------------------------- X";
	for (int i = 0; i < 64; i++)
	{
		if (pos.Player().Get(static_cast<Field>(63 - i)))
			str[i] = 'X';
		else if (pos.Opponent().Get(static_cast<Field>(63 - i)))
			str[i] = 'O';
	}
	return str;
}

std::string MultiLine(const Position& pos)
{
	Moves moves = PossibleMoves(pos);
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
	{
		Field field = static_cast<Field>(63 - i);
		char symbol = '-';
		if (pos.Player().Get(field))
			symbol = 'X';
		else if (pos.Opponent().Get(field))
			symbol = 'O';
		else if (moves.contains(field))
			symbol = '+';
		board[22 + 2 * i + 4 * (i / 8)] = symbol;
	}
	return board;
}

int EvalGameOver(const Position& pos) noexcept
{
	int P = popcount(pos.Player());
	int O = popcount(pos.Opponent());
	if (P > O)
		return 32 - O;
	if (P < O)
		return P - 32;
	return 0;
}

CUDA_CALLABLE Position Play(const Position& pos, Field move, BitBoard flips) noexcept
{
	assert((pos.Opponent() & flips) == flips); // only flipping opponent stones.

	return { pos.Opponent() ^ flips, pos.Player() ^ flips ^ BitBoard(move) };
}

CUDA_CALLABLE Position Play(const Position& pos, Field move) noexcept
{
	//assert(pos.Empties().Get(move)); // move field is free.

	auto flips = Flips(pos, move);
	return Play(pos, move, flips);
}

CUDA_CALLABLE Position PlayPass(const Position& pos) noexcept
{
	return { pos.Opponent(), pos.Player() };
}

CUDA_CALLABLE Position PlayOrPass(const Position& pos, Field move) noexcept
{
	if (move == Field::invalid)
		return PlayPass(pos);
	return Play(pos, move);
}