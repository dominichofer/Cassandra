#include "pch.h"

TEST(RandomPosition, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	Position pos_1 = RandomPosition(seed);
	Position pos_2 = RandomPosition(seed);

	ASSERT_EQ(pos_1, pos_2);
}

TEST(RandomPositions, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	int count = 3; // arbitrary
	std::vector<Position> pos_1 = RandomPositions(count, seed);
	std::vector<Position> pos_2 = RandomPositions(count, seed);

	ASSERT_EQ(pos_1, pos_2);
}

TEST(RandomPositionGenerator, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	Position pos_1 = RandomPositionGenerator{ seed }();
	Position pos_2 = RandomPositionGenerator{ seed }();

	ASSERT_EQ(pos_1, pos_2);
}

TEST(RandomPositionWithEmptyCount, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	int empty_count = 15; // arbitrary
	Position pos_1 = RandomPositionWithEmptyCount(empty_count, seed);
	Position pos_2 = RandomPositionWithEmptyCount(empty_count, seed);

	ASSERT_EQ(pos_1, pos_2);
}

TEST(RandomPositionWithEmptyCount, returns_correct_empty_count)
{
	for (int empty_count = 0; empty_count <= 60; empty_count++)
	{
		Position pos = RandomPositionWithEmptyCount(empty_count);
		ASSERT_EQ(pos.EmptyCount(), empty_count);
	}
}

TEST(RandomPositionsWithEmptyCount, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	int count = 3; // arbitrary
	int empty_count = 15; // arbitrary
	std::vector<Position> pos_1 = RandomPositionsWithEmptyCount(count, empty_count, seed);
	std::vector<Position> pos_2 = RandomPositionsWithEmptyCount(count, empty_count, seed);

	ASSERT_EQ(pos_1, pos_2);
}

TEST(RandomPositionGeneratorWithEmptyCount, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	int empty_count = 15; // arbitrary
	Position pos_1 = RandomPositionGeneratorWithEmptyCount{ empty_count, seed }();
	Position pos_2 = RandomPositionGeneratorWithEmptyCount{ empty_count, seed }();

	ASSERT_EQ(pos_1, pos_2);
}

TEST(RandomPositionGeneratorWithEmptyCount, returns_correct_empty_count)
{
	for (int empty_count = 0; empty_count <= 60; empty_count++)
	{
		Position pos = RandomPositionGeneratorWithEmptyCount{ empty_count }();
		ASSERT_EQ(pos.EmptyCount(), empty_count);
	}
}

TEST(RandomlyPlayedPositionWithEmptyCount, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	int empty_count = 15; // arbitrary
	Position pos_1 = RandomlyPlayedPositionWithEmptyCount(empty_count, seed);
	Position pos_2 = RandomlyPlayedPositionWithEmptyCount(empty_count, seed);

	ASSERT_EQ(pos_1, pos_2);
}

TEST(RandomlyPlayedPositionWithEmptyCount, returns_correct_empty_count)
{
	for (int empty_count = 0; empty_count <= 60; empty_count++)
	{
		Position pos = RandomlyPlayedPositionWithEmptyCount(empty_count);
		ASSERT_EQ(pos.EmptyCount(), empty_count);
	}
}

TEST(RandomlyPlayedPositionsWithEmptyCount, is_deterministic)
{
	unsigned int seed = 42; // arbitrary
	int count = 3; // arbitrary
	int empty_count = 15; // arbitrary
	std::vector<Position> pos_1 = RandomlyPlayedPositionsWithEmptyCount(count, empty_count, seed);
	std::vector<Position> pos_2 = RandomlyPlayedPositionsWithEmptyCount(count, empty_count, seed);

	ASSERT_EQ(pos_1, pos_2);
}