#include "Position.h"
#include "BitBoard.h"
#include "Flips.h"
#include "PossibleMoves.h"
#include <stdexcept>

Position Position::FromString(std::string_view str)
{
	if (str.length() < 66)
		throw std::runtime_error("Invalid position format");

	uint64_t P{ 0 }, O{ 0 };
	for (int i = 0; i < 64; i++)
		if (str[i] == 'X')
			P |= 1ULL << (63 - i);
		else if (str[i] == 'O')
			O |= 1ULL << (63 - i);

	if (str[65] == 'X')
		return { P, O };
	else
		return { O, P };
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
		else
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

CUDA_CALLABLE Position Play(const Position& pos, Field move, uint64_t flips) noexcept
{
	assert((pos.Opponent() & flips) == flips); // only flips opponent discs.

	return { pos.Opponent() ^ flips, pos.Player() ^ flips ^ Bit(move) };
}

CUDA_CALLABLE Position Play(const Position& pos, Field move) noexcept
{
	assert((pos.Empties() & Bit(move)) == Bit(move)); // move field is free.

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

Position FlippedToUnique(const Position& pos1) noexcept
{
	Position pos2 = FlippedCodiagonal(pos1);
	Position pos3 = FlippedDiagonal(pos1);
	Position pos4 = FlippedHorizontal(pos1);
	Position pos5 = FlippedVertical(pos1);
	Position pos6 = FlippedVertical(pos2);
	Position pos7 = FlippedVertical(pos3);
	Position pos8 = FlippedVertical(pos4);
	return std::min({ pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8 });
}
