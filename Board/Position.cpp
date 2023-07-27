#include "Position.h"
#include "BitBoard.h"
#include "Field.h"
#include "Flips.h"
#include "Moves.h"
#include "PossibleMoves.h"
#include <cassert>
#include <stdexcept>
#include <regex>

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

std::string SingleLine(const Position& pos)
{
	std::string str = "---------------------------------------------------------------- X";
	for (int i = 0; i < 64; i++)
	{
		uint64_t mask = 1ULL << (63 - i);
		if (pos.Player() & mask)
			str[i] = 'X';
		else if (pos.Opponent() & mask)
			str[i] = 'O';
	}
	return str;
}

std::string MultiLine(const Position& pos)
{
	Moves moves = PossibleMoves(pos);
	auto character = [pos, moves](int index) -> std::string {
		uint64_t mask = 1ULL << (63 - index);
		if (pos.Player() & mask)
			return "X";
		if (pos.Opponent() & mask)
			return "O";
		if (moves & mask)
			return "+";
		return "-";
	};

	std::string board = "  A B C D E F G H  \n";
	for (int i = 0; i < 8; i++)
	{
		board += std::to_string(i + 1) + " ";
		for (int j = 0; j < 8; j++)
			board += character(i * 8 + j) + " ";
		board += std::to_string(i + 1) + "\n";
	}
	board += "  A B C D E F G H  ";
	return board;
}

std::string to_string(const Position& pos)
{
	return SingleLine(pos);
}

Position PositionFromString(std::string_view s)
{
	if (s.length() < 66)
		throw std::runtime_error("Invalid position format");

	uint64_t P{ 0 }, O{ 0 };
	for (int i = 0; i < 64; i++)
		if (s[i] == 'X')
			P |= 1ULL << (63 - i);
		else if (s[i] == 'O')
			O |= 1ULL << (63 - i);

	if (s[65] == 'X')
		return { P, O };
	else
		return { O, P };
}

bool IsPosition(std::string_view str)
{
	std::regex pattern("[XO-]{64} [XO]");
	return std::regex_match(str.begin(), str.end(), pattern);
}

CUDA_CALLABLE Position Play(const Position& pos, Field move, uint64_t flips) noexcept
{
	assert((pos.Opponent() & flips) == flips); // only flips opponent discs.

	return { pos.Opponent() ^ flips, pos.Player() ^ flips ^ Bit(move) };
}

CUDA_CALLABLE Position Play(const Position& pos, Field move) noexcept
{
	//assert(pos.Empties().Get(move)); // move field is free. //TODO: Use it?

	auto flips = Flips(pos, move);
	return Play(pos, move, flips);
}

CUDA_CALLABLE Position PlayPass(const Position& pos) noexcept
{
	return { pos.Opponent(), pos.Player() };
}

CUDA_CALLABLE Position PlayOrPass(const Position& pos, Field move) noexcept
{
	if (move == Field::PS)
		return PlayPass(pos);
	return Play(pos, move);
}

Position FlippedCodiagonal(const Position& pos) noexcept
{
	return { FlippedCodiagonal(pos.Player()), FlippedCodiagonal(pos.Opponent()) };
}

Position FlippedDiagonal(const Position& pos) noexcept
{
	return { FlippedDiagonal(pos.Player()), FlippedDiagonal(pos.Opponent()) };
}

Position FlippedHorizontal(const Position& pos) noexcept
{
	return { FlippedHorizontal(pos.Player()), FlippedHorizontal(pos.Opponent()) };
}

Position FlippedVertical(const Position& pos) noexcept
{
	return { FlippedVertical(pos.Player()), FlippedVertical(pos.Opponent()) };
}

Position FlippedToUnique(Position pos) noexcept
{
	Position min = pos;
	pos = FlippedVertical(pos);		if (pos < min) min = pos;
	pos = FlippedHorizontal(pos);	if (pos < min) min = pos;
	pos = FlippedVertical(pos);		if (pos < min) min = pos;
	pos = FlippedCodiagonal(pos);	if (pos < min) min = pos;
	pos = FlippedVertical(pos);		if (pos < min) min = pos;
	pos = FlippedHorizontal(pos);	if (pos < min) min = pos;
	pos = FlippedVertical(pos);		if (pos < min) min = pos;
	return min;
}
