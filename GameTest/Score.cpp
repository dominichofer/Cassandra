#include "pch.h"

TEST(Score, to_string)
{
    EXPECT_EQ(to_string(Score{ +32 }), "+64");
    EXPECT_EQ(to_string(Score{ +01 }), "+02");
    EXPECT_EQ(to_string(Score{ +00 }), "+00");
    EXPECT_EQ(to_string(Score{ -01 }), "-02");
    EXPECT_EQ(to_string(Score{ -32 }), "-64");
}

TEST(Score, FromString)
{
    EXPECT_EQ(Score{ +32 }, Score::FromString("+64"));
    EXPECT_EQ(Score{ +01 }, Score::FromString("+02"));
    EXPECT_EQ(Score{ +00 }, Score::FromString("+00"));
    EXPECT_EQ(Score{ -01 }, Score::FromString("-02"));
    EXPECT_EQ(Score{ -32 }, Score::FromString("-64"));
    EXPECT_THROW(Score::FromString("+A"), std::runtime_error);
}

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