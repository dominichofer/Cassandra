#include "pch.h"

TEST(OpenIntervalTest, Equality)
{
	OpenInterval i1(1, 3);
	OpenInterval i2(1, 3);
	OpenInterval i3(2, 4);

	EXPECT_EQ(i1, i2);
	EXPECT_NE(i1, i3);
}

TEST(OpenIntervalTest, NegationOperator)
{
	OpenInterval i(1, 3);
	EXPECT_EQ(-i, OpenInterval(-3, -1));
}

TEST(OpenIntervalTest, Contains) {
	OpenInterval i(1, 3);

	EXPECT_FALSE(i.Contains(1));
	EXPECT_TRUE(i.Contains(2));
	EXPECT_FALSE(i.Contains(3));
}

TEST(OpenIntervalTest, Compare)
{
	OpenInterval i(1, 3);

	EXPECT_TRUE(1 < i);
	EXPECT_FALSE(2 < i);
	EXPECT_FALSE(2 > i);
	EXPECT_TRUE(3 > i);
	EXPECT_TRUE(i > 1);
	EXPECT_FALSE(i > 2);
	EXPECT_FALSE(i < 2);
	EXPECT_TRUE(i < 3);

	EXPECT_TRUE (OpenInterval(-3, 1) < i);
	EXPECT_FALSE(OpenInterval(-3, 2) < i);
	EXPECT_FALSE(OpenInterval(2, 6) > i);
	EXPECT_TRUE(OpenInterval(3, 6) > i);
	EXPECT_TRUE(i > OpenInterval(-3, 1));
	EXPECT_FALSE(i > OpenInterval(-3, 2));
	EXPECT_FALSE(i < OpenInterval(2, 6));
	EXPECT_TRUE(i < OpenInterval(3, 6));

	EXPECT_TRUE(ClosedInterval(-3, 1) < i);
	EXPECT_FALSE(ClosedInterval(-3, 2) < i);
	EXPECT_FALSE(ClosedInterval(2, 6) > i);
	EXPECT_TRUE(ClosedInterval(3, 6) > i);
	EXPECT_TRUE(i > ClosedInterval(-3, 1));
	EXPECT_FALSE(i > ClosedInterval(-3, 2));
	EXPECT_FALSE(i < ClosedInterval(2, 6));
	EXPECT_TRUE(i < ClosedInterval(3, 6));
}

TEST(ClosedIntervalTest, Equality)
{
	ClosedInterval i1(1, 3);
	ClosedInterval i2(1, 3);
	ClosedInterval i3(2, 4);

	EXPECT_EQ(i1, i2);
	EXPECT_NE(i1, i3);
}

TEST(ClosedIntervalTest, LessThanOperator)
{
	ClosedInterval i(1, 3);

	EXPECT_TRUE(0 < i);
	EXPECT_FALSE(1 < i);
	EXPECT_FALSE(3 > i);
	EXPECT_TRUE(4 > i);
	EXPECT_TRUE(i > 0);
	EXPECT_FALSE(i > 1);
	EXPECT_FALSE(i < 3);
	EXPECT_TRUE(i < 4);

	EXPECT_TRUE(OpenInterval(-3, 1) < i);
	EXPECT_FALSE(OpenInterval(-3, 2) < i);
	EXPECT_FALSE(OpenInterval(2, 6) > i);
	EXPECT_TRUE(OpenInterval(3, 6) > i);
	EXPECT_TRUE(i > OpenInterval(-3, 1));
	EXPECT_FALSE(i > OpenInterval(-3, 2));
	EXPECT_FALSE(i < OpenInterval(2, 6));
	EXPECT_TRUE(i < OpenInterval(3, 6));

	EXPECT_TRUE(ClosedInterval(-3, 0) < i);
	EXPECT_FALSE(ClosedInterval(-3, 1) < i);
	EXPECT_FALSE(ClosedInterval(3, 6) > i);
	EXPECT_TRUE(ClosedInterval(4, 6) > i);
	EXPECT_TRUE(i > ClosedInterval(-3, 0));
	EXPECT_FALSE(i > ClosedInterval(-3, 1));
	EXPECT_FALSE(i < ClosedInterval(3, 6));
	EXPECT_TRUE(i < ClosedInterval(4, 6));
}

TEST(ClosedIntervalTest, Contains) {
	ClosedInterval i(1, 3);

	EXPECT_FALSE(i.Contains(0));
	EXPECT_TRUE(i.Contains(1));
	EXPECT_TRUE(i.Contains(2));
	EXPECT_TRUE(i.Contains(3));
	EXPECT_FALSE(i.Contains(4));
}

TEST(ClosedIntervalTest, Overlaps) {
	ClosedInterval i(1, 3);

	EXPECT_FALSE(i.Overlaps(OpenInterval(-1, 1)));
	EXPECT_TRUE(i.Overlaps(OpenInterval(-1, 2)));
	EXPECT_TRUE(i.Overlaps(OpenInterval(2, 6)));
	EXPECT_FALSE(i.Overlaps(OpenInterval(3, 6)));
}