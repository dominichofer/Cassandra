#include "pch.h"

TEST(EvalGameOver, empty_board)
{
	Position pos = 
		"        "
		"        "
		"        "
		"        "
		"        "
		"        "
		"        "
		"        "_pos;

	ASSERT_EQ(EvalGameOver(pos), 0);
}

TEST(EvalGameOver, full_of_player)
{
	Position pos =
		"XXXXXXXX"
		"XXXXXXXX"
		"XXXXXXXX"
		"XXXXXXXX"
		"XXXXXXXX"
		"XXXXXXXX"
		"XXXXXXXX"
		"XXXXXXXX"_pos;

	ASSERT_EQ(EvalGameOver(pos), +64);
}

TEST(EvalGameOver, full_of_opponent)
{
	Position pos =
		"OOOOOOOO"
		"OOOOOOOO"
		"OOOOOOOO"
		"OOOOOOOO"
		"OOOOOOOO"
		"OOOOOOOO"
		"OOOOOOOO"
		"OOOOOOOO"_pos;

	ASSERT_EQ(EvalGameOver(pos), -64);
}

TEST(EvalGameOver, half_half)
{
	Position pos =
		"XXXXXXXX"
		"XXXXXXXX"
		"XXXXXXXX"
		"XXXXXXXX"
		"OOOOOOOO"
		"OOOOOOOO"
		"OOOOOOOO"
		"OOOOOOOO"_pos;

	ASSERT_EQ(EvalGameOver(pos), 0);
}

TEST(EvalGameOver, empties_count_toward_player)
{
	Position pos =
		"        "
		"        "
		"        "
		"        "
		"    X   "
		"        "
		"        "
		"        "_pos;

	ASSERT_EQ(EvalGameOver(pos), +64);
}

TEST(EvalGameOver, empties_count_toward_opponent)
{
	Position pos =
		"        "
		"        "
		"   O    "
		"        "
		"        "
		"        "
		"        "
		"        "_pos;

	ASSERT_EQ(EvalGameOver(pos), -64);
}