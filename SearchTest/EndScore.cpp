#include "pch.h"

TEST(EndScore, full_of_player)
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

	ASSERT_EQ(EndScore(pos), +64 / 2);
}

TEST(EndScore, full_of_opponent)
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

	ASSERT_EQ(EndScore(pos), -64 / 2);
}

TEST(EndScore, half_half)
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

	ASSERT_EQ(EndScore(pos), 0 / 2);
}

TEST(EndScore, empties_count_toward_player)
{
	Position pos =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - O X - - -"
		"- - - X X - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pos;

	ASSERT_EQ(EndScore(pos), +62 / 2);
}

TEST(EndScore, empties_count_toward_opponent)
{
	Position pos =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - X O - - -"
		"- - - O O - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pos;

	ASSERT_EQ(EndScore(pos), -62 / 2);
}
