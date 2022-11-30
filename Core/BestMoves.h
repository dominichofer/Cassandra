#pragma once
#include "BitBoard.h"
#include "Moves.h"
#include <array>

//class BestMoves
//{
//	Field move;
//public:
//	BestMoves() noexcept = default;
//	BestMoves(Field move) noexcept : move(move) {}
//
//	bool operator==(const BestMoves&) const noexcept = default;
//	bool operator!=(const BestMoves&) const noexcept = default;
//
//	operator Moves() const noexcept { return move == Field::invalid ? BitBoard() : BitBoard(move); }
//
//	void Add(Field novum) { move = novum; }
//	void Add(const BestMoves& novum) { move = novum.move; }
//};

class BestMoves
{
	// Constraint:
	// move1 != Field::invalid or move2 == Field::invalid

	Field move1 = Field::invalid;
	Field move2 = Field::invalid;
	bool add(Field novum)
	{
		if (novum == Field::invalid or novum == move1)
			return false;
		move2 = move1;
		move1 = novum;
		return true;
	}
public:
	BestMoves() noexcept = default;
	BestMoves(Field move1, Field move2) noexcept : move1(move1), move2(move2) {}

	bool operator==(const BestMoves&) const noexcept = default;
	bool operator!=(const BestMoves&) const noexcept = default;

	operator Moves() const noexcept { return Moves{ (move1 == Field::invalid ? BitBoard() : BitBoard(move1)) | (move2 == Field::invalid ? BitBoard() : BitBoard(move2)) }; }

	Field first() const noexcept { return move1; }
	Field second() const noexcept { return move2; }

	void Add(Field novum) { add(novum); }
	void Add(const BestMoves& novum)
	{
		bool added = add(novum.move1);
		if (not added or move2 == Field::invalid)
		{
			if (novum.move2 == Field::invalid or novum.move2 == move2)
				return;
			move2 = novum.move2;
		}
	}
};