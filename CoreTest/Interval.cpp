#include "pch.h"

TEST(Intervals, Compare_less_Intervals_to_Score)
{
	ASSERT_TRUE(OpenInterval(2, 4) < 4);
	ASSERT_TRUE(2 < OpenInterval(2, 4));

	ASSERT_FALSE(ClosedInterval(2, 4) < 4);
	ASSERT_FALSE(2 < ClosedInterval(2, 4));
}

TEST(Intervals, Compare_greater_Intervals_to_Score)
{
	ASSERT_TRUE(4 > OpenInterval(2, 4));
	ASSERT_TRUE(OpenInterval(2, 4) > 2);

	ASSERT_FALSE(4 > ClosedInterval(2, 4));
	ASSERT_FALSE(ClosedInterval(2, 4) > 2);
}

TEST(Intervals, Compare_less_Intervals_to_Intervals)
{
	ASSERT_TRUE(OpenInterval(2, 4) < OpenInterval(3, 6)); // The overlap is empty.
	ASSERT_TRUE(OpenInterval(2, 4) < OpenInterval(4, 6)); // No overlap.

	ASSERT_TRUE(OpenInterval(2, 4) < ClosedInterval(4, 6)); // Intervals are touching but don't overlap.

	ASSERT_TRUE(ClosedInterval(2, 4) < OpenInterval(4, 6)); // Intervals are touching but don't overlap.

	ASSERT_FALSE(ClosedInterval(2, 4) < ClosedInterval(4, 6)); // The overlap contains one element.
	ASSERT_TRUE(ClosedInterval(2, 4) < ClosedInterval(5, 6)); // No overlap.
}

TEST(Intervals, Compare_greater_Intervals_to_Intervals)
{
	ASSERT_TRUE(OpenInterval(3, 6) > OpenInterval(2, 4));
	ASSERT_TRUE(OpenInterval(4, 6) > OpenInterval(2, 4));

	ASSERT_TRUE(ClosedInterval(4, 6) > OpenInterval(2, 4));

	ASSERT_TRUE(OpenInterval(4, 6) > ClosedInterval(2, 4));

	ASSERT_FALSE(ClosedInterval(3, 6) > ClosedInterval(2, 4));
	ASSERT_TRUE(ClosedInterval(5, 6) > ClosedInterval(2, 4));
}

TEST(Intervals, OpenInterval_Contains_Score)
{
	OpenInterval w(2, 6);

	ASSERT_FALSE(w.Contains(2));
	ASSERT_FALSE(w.Contains(6));
}

TEST(Intervals, ClosedInterval_Contains_Score)
{
	ClosedInterval w(2, 6);

	ASSERT_TRUE(w.Contains(2));
	ASSERT_TRUE(w.Contains(6));
}

TEST(Intervals, Interval_Contains_Interval)
{
	OpenInterval open(2, 6);
	ClosedInterval closed(2, 6);

	ASSERT_TRUE(open.Contains(open));
	ASSERT_FALSE(open.Contains(closed));
	ASSERT_TRUE(closed.Contains(open));
	ASSERT_TRUE(closed.Contains(OpenInterval(1, 5)));
	ASSERT_TRUE(closed.Contains(closed));
}

TEST(Intervals, Interval_Overlaps_Interval)
{
	ASSERT_TRUE(OpenInterval(2, 6).Overlaps(OpenInterval(4, 10)));
	ASSERT_FALSE(OpenInterval(2, 6).Overlaps(OpenInterval(5, 10)));
	ASSERT_FALSE(OpenInterval(2, 6).Overlaps(OpenInterval(6, 10)));

	ASSERT_TRUE(OpenInterval(2, 6).Overlaps(ClosedInterval(5, 10)));
	ASSERT_FALSE(OpenInterval(2, 6).Overlaps(ClosedInterval(6, 10)));

	ASSERT_TRUE(ClosedInterval(2, 6).Overlaps(OpenInterval(5, 10)));
	ASSERT_FALSE(ClosedInterval(2, 6).Overlaps(OpenInterval(6, 10)));

	ASSERT_TRUE(ClosedInterval(2, 6).Overlaps(ClosedInterval(6, 10)));
	ASSERT_FALSE(ClosedInterval(2, 6).Overlaps(ClosedInterval(7, 10)));
}

TEST(Intervals, Overlap)
{	
	ASSERT_EQ(Overlap(OpenInterval(2, 6), OpenInterval(4, 8)), OpenInterval(4, 6));
	ASSERT_EQ(Overlap(ClosedInterval(2, 6), ClosedInterval(4, 8)), ClosedInterval(4, 6));
}

TEST(Intervals, OpenInterval_subtract_assign)
{
	OpenInterval open(2, 6);
	ClosedInterval closed(4, 8);

	open -= closed;

	ASSERT_EQ(open, OpenInterval(2, 4));
}

TEST(Intervals, ClosedInterval_subtract_assign)
{
	OpenInterval open(2, 6);
	ClosedInterval closed(4, 8);

	closed -= open;

	ASSERT_EQ(closed, ClosedInterval(6, 8));
}

TEST(Intervals, Subtract)
{
	ASSERT_EQ(OpenInterval(2, 6) - ClosedInterval(4, 8), OpenInterval(2, 4));
	ASSERT_EQ(ClosedInterval(2, 6) - OpenInterval(4, 8), ClosedInterval(2, 4));
}

TEST(Intervals, empty)
{
	ASSERT_TRUE(OpenInterval(2, 3).empty());
	ASSERT_FALSE(OpenInterval(2, 4).empty());
}

TEST(Intervals, Negation_flips_window)
{
	ASSERT_TRUE(-OpenInterval(2, 10) == OpenInterval(-10, -2));
	ASSERT_TRUE(-ClosedInterval(2, 10) == ClosedInterval(-10, -2));
}