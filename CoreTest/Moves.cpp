#include "pch.h"

TEST(Moves, range_based_for_loop)
{
	const Moves moves = PossibleMoves(Position::Start());

	std::vector<Field> range_loop;
	for (const Field move : moves)
		range_loop.push_back(move);

	std::sort(range_loop.begin(), range_loop.end());
	ASSERT_EQ(range_loop.size(), 4);
	ASSERT_EQ(range_loop[0], Field::D3);
	ASSERT_EQ(range_loop[1], Field::C4);
	ASSERT_EQ(range_loop[2], Field::F5);
	ASSERT_EQ(range_loop[3], Field::E6);
}