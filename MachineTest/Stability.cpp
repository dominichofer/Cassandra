#include "pch.h"
#include "Core/Position.h"
#include "Machine/Stability.h"

TEST(StableStones, start_position_has_none)
{
	const auto stables = StableStones(Position::Start());

	ASSERT_EQ(stables, 0ui64);
}

TEST(StableStones, filled_edges_are_stable)
{
	const auto pos =
		"O X O X O X O X"
		"X             O"
		"X             O"
		"X     O X     O"
		"X     X O     O"
		"X             O"
		"X             O"
		"O O O O X X X X"_pos;

	const auto stables =
		"#   #   #   #  "
		"              #"
		"              #"
		"              #"
		"              #"
		"              #"
		"              #"
		"# # # #        "_BitBoard;

	ASSERT_EQ(StableStones(pos), stables);
}