#include "pch.h"

TEST(Moves, range_based_for_loop)
{
	Moves moves = PossibleMoves(Position::Start());

	std::vector<Field> possible_moves;
	for (Field move : moves)
		possible_moves.push_back(move);

	std::sort(possible_moves.begin(), possible_moves.end());
	ASSERT_EQ(possible_moves.size(), 4);
	ASSERT_EQ(possible_moves[0], Field::E6);
	ASSERT_EQ(possible_moves[1], Field::F5);
	ASSERT_EQ(possible_moves[2], Field::C4);
	ASSERT_EQ(possible_moves[3], Field::D3);
}