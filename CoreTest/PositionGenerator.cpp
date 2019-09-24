#include "pch.h"

TEST(PosGen, Random_is_deterministic)
{
	PositionGenerator pg_1(42);
	PositionGenerator pg_2(42);

	ASSERT_EQ(pg_1.Random(), pg_2.Random());
}

TEST(PosGen, Random_with_empty_count_is_deterministic)
{
	PositionGenerator pg_1(42);
	PositionGenerator pg_2(42);

	ASSERT_EQ(pg_1.Random(8), pg_2.Random(8));
}

TEST(PosGen, Random_with_empty_count_returns_empty_count)
{
	PositionGenerator pg(42);

	for (std::size_t empty_count = 0; empty_count < 60; empty_count++)
		ASSERT_EQ(pg.Random(empty_count).EmptyCount(), empty_count);
}

class MockPlayer : public Player
{
	Position Play(Position in) noexcept(false) final
	{
		auto P = in.GetP();
		SetBit(P, BitScanLSB(in.Empties()));
		return { in.GetO(), P };
	}
};
TEST(PosGen, Played_returns_empty_count)
{
	PositionGenerator pg(42);
	MockPlayer mock;

	for (std::size_t empty_count = 0; empty_count <= 60; empty_count++)
		ASSERT_EQ(pg.Played(mock, empty_count).EmptyCount(), empty_count);
}