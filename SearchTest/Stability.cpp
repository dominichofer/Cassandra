#include "pch.h"

TEST(StableStonesOpponent, start_position_has_none)
{
	const auto stables = StableStonesOpponent(Position::Start());

	ASSERT_EQ(stables, 0ULL);
}

TEST(StableStonesOpponent, corners_are_stable)
{
	const auto pos =
		"O - - - - - - O"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"O - - - - - - O"_pos;

	const auto stables =
		"# - - - - - - #"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"# - - - - - - #"_pattern;

	ASSERT_EQ(StableStonesOpponent(pos), stables);
}

TEST(StableStonesOpponent, filled_edges_are_stable)
{
	const auto pos =
		"O X O X O X O X"
		"X - - - - - - O"
		"X - - - - - - O"
		"X - - - - - - O"
		"X - - - - - - O"
		"X - - - - - - O"
		"X - - - - - - O"
		"O O O O X X X X"_pos;

	const auto stables =
		"# - # - # - # -"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - - #"
		"# # # # - - - -"_pattern;

	ASSERT_EQ(StableStonesOpponent(pos), stables);
}

TEST(StableStonesOpponent, unatackables_are_stable)
{
	const auto pos =
		"X - - X - - X -"
		"- X - X - X - -"
		"- - X X X - - -"
		"X X X O X X X X"
		"- - X X X - - -"
		"- X - X - X - -"
		"X - - X - - X -"
		"- - - X - - - X"_pos;

	const auto stables =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - # - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pattern;

	ASSERT_EQ(StableStonesOpponent(pos), stables);
}

TEST(StableStonesOpponent, undisturbed_skylines_are_stable)
{
	const auto pos =
		"O O O - - - - -"
		"O O - - - - - -"
		"O O - - - - - -"
		"O O - - - - - -"
		"O O - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pos;

	const auto stables =
		"# # # - - - - -"
		"# # - - - - - -"
		"# # - - - - - -"
		"# # - - - - - -"
		"# - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pattern;

	ASSERT_EQ(StableStonesOpponent(pos), stables);
}

TEST(StableStonesOpponent, disturbed_skylines_are_stable)
{
	const auto pos =
		"O O O - - - - -"
		"O O - - - - - -"
		"O O - - - - - -"
		"X O - - - - - -"
		"O - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pos;

	const auto stables =
		"# # # - - - - -"
		"# # - - - - - -"
		"# - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pattern;

	ASSERT_EQ(StableStonesOpponent(pos), stables);
}

TEST(StableStonesOpponent, undisturbed_corner_triangles_are_stable)
{
	const auto pos =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - O"
		"- - - - - - O O"
		"- - - - - O O O"
		"- - - - O O O O"_pos;

	const auto stables =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - # #"
		"- - - - - # # #"
		"- - - - # # # #"_pattern;

	ASSERT_EQ(StableStonesOpponent(pos), stables);
}

TEST(StableStonesOpponent, disturbed_corner_triangles_are_stable)
{
	const auto pos =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - O"
		"- - - - - - O X"
		"- - - - - O O O"
		"- - - - O O O O"_pos;

	const auto stables =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - # #"_pattern;

	ASSERT_TRUE((StableStonesOpponent(pos) & stables) == stables);
}