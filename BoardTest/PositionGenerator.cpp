#include "pch.h"

TEST(RandomPositionGenerator, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	Position pos_1 = RandomPositionGenerator{ seed }();
	Position pos_2 = RandomPositionGenerator{ seed }();

	ASSERT_EQ(pos_1, pos_2);
}

TEST(RandomPosition, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	Position pos_1 = RandomPosition(seed);
	Position pos_2 = RandomPosition(seed);

	ASSERT_EQ(pos_1, pos_2);
}

TEST(RandomPositionWithEmptyCount, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	Position pos_1 = RandomPositionWithEmptyCount(3, seed);
	Position pos_2 = RandomPositionWithEmptyCount(3, seed);

	ASSERT_EQ(pos_1, pos_2);
}
