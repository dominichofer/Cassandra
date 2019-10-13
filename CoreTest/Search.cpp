#include "pch.h"

TEST(EvalGameOver, full_of_player)
{
	Position pos =
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"_pos;

	ASSERT_EQ(EvalGameOver(pos), +64);
}

TEST(EvalGameOver, full_of_opponent)
{
	Position pos =
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"_pos;

	ASSERT_EQ(EvalGameOver(pos), -64);
}

TEST(EvalGameOver, half_half)
{
	Position pos =
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"_pos;

	ASSERT_EQ(EvalGameOver(pos), 0);
}

TEST(EvalGameOver, empties_count_toward_player)
{
	Position pos =
		"               "
		"               "
		"               "
		"      X X      "
		"      X X      "
		"               "
		"               "
		"               "_pos;

	ASSERT_EQ(EvalGameOver(pos), +64);
}

TEST(EvalGameOver, empties_count_toward_opponent)
{
	Position pos =
		"               "
		"               "
		"               "
		"      O O      "
		"      O O      "
		"               "
		"               "
		"               "_pos;

	ASSERT_EQ(EvalGameOver(pos), -64);
}