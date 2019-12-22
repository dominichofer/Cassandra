#include "pch.h"
#include "Core/Position.h"
#include "Core/Machine.h"

TEST(StableStones, start_position_has_none)
{
	const auto stables = StableStones(Position::Start());

	ASSERT_EQ(stables, BitBoard{0});
}

TEST(StableStones, corners_are_stable)
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
		"# - - - - - - #"_BitBoard;

	ASSERT_EQ(StableStones(pos), stables);
}

TEST(StableStones, filled_edges_are_stable)
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
		"# # # # - - - -"_BitBoard;

	ASSERT_EQ(StableStones(pos), stables);
}

TEST(StableStones, unatackables_are_stable)
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
		"- - - - - - - -"_BitBoard;

	ASSERT_EQ(StableStones(pos), stables);
}

TEST(StableStones, undisturbed_skylines_are_stable)
{
	const auto pos =
		"O O O - - - - -"
		"O O - - - - - -"
		"O O - - - - - -"
		"O O - - - - - -"
		"O - - - - - - -"
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
		"- - - - - - - -"_BitBoard;

	ASSERT_EQ(StableStones(pos), stables);
}

TEST(StableStones, disturbed_skylines_are_stable)
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
		"- - - - - - - -"_BitBoard;

	ASSERT_EQ(StableStones(pos), stables);
}

TEST(StableStones, undisturbed_corner_triangles_are_stable)
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
		"- - - - # # # #"_BitBoard;

	ASSERT_EQ(StableStones(pos), stables);
}

TEST(StableStones, disturbed_corner_triangles_are_stable)
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
		"- - - - - - # #"_BitBoard;

	ASSERT_TRUE(stables.IsSubsetOf(StableStones(pos)));
}