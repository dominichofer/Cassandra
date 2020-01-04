#include "pch.h"

TEST(ClosedInterval, Equality)
{
	ClosedInterval a{ 1, 2 };
	ClosedInterval b{ 1, 3 };

	ASSERT_TRUE(a == a);
	ASSERT_FALSE(a != a);
	ASSERT_FALSE(a == b);
	ASSERT_TRUE(a != b);
}

TEST(ClosedInterval, Compare)
{
	ClosedInterval a{ 1, 2 };
	ClosedInterval b{ 2, 3 };
	ClosedInterval c{ 3, 4 };

	ASSERT_FALSE(a < a);
	ASSERT_FALSE(a < b);
	ASSERT_TRUE(a < c);
	ASSERT_FALSE(b < b);
	ASSERT_FALSE(b < c);
	ASSERT_FALSE(c < c);

	ASSERT_FALSE(a > a);
	ASSERT_FALSE(b > a);
	ASSERT_TRUE(c > a);
	ASSERT_FALSE(b > b);
	ASSERT_FALSE(c > b);
	ASSERT_FALSE(c > c);
}

TEST(ClosedInterval, Compare_with_score)
{
	ClosedInterval i{ 1, 2 };

	ASSERT_TRUE(0 < i);
	ASSERT_FALSE(1 < i);
	ASSERT_FALSE(i < 2);
	ASSERT_TRUE(i < 3);

	ASSERT_TRUE(i > 0);
	ASSERT_FALSE(i > 1);
	ASSERT_FALSE(2 > i);
	ASSERT_TRUE(3 > i);
}

TEST(ClosedInterval, Intersection)
{
	ClosedInterval a{ 1, 3 };
	ClosedInterval b{ 2, 4 };
	ClosedInterval c{ 2, 3 };

	auto i = Intersection(a, b);
	
	ASSERT_TRUE(i == c);
}

TEST(OpenInterval, Equality)
{
	OpenInterval a{ 1, 2 };
	OpenInterval b{ 1, 3 };

	ASSERT_TRUE(a == a);
	ASSERT_FALSE(a != a);
	ASSERT_FALSE(a == b);
	ASSERT_TRUE(a != b);
}

TEST(OpenInterval, Compare)
{
	OpenInterval a{ 1, 2 };
	OpenInterval b{ 2, 4 };
	OpenInterval c{ 3, 5 };

	ASSERT_FALSE(a < a);
	ASSERT_TRUE(a < b);
	ASSERT_TRUE(a < c);
	ASSERT_FALSE(b < b);
	ASSERT_FALSE(b < c);
	ASSERT_FALSE(c < c);

	ASSERT_FALSE(a > a);
	ASSERT_TRUE(b > a);
	ASSERT_TRUE(c > a);
	ASSERT_FALSE(b > b);
	ASSERT_FALSE(c > b);
	ASSERT_FALSE(c > c);
}

TEST(OpenInterval, Compare_with_score)
{
	OpenInterval i{ 1, 2 };

	ASSERT_TRUE(0 < i);
	ASSERT_TRUE(1 < i);
	ASSERT_FALSE(2 < i);
	ASSERT_FALSE(i < 1);
	ASSERT_TRUE(i < 2);
	ASSERT_TRUE(i < 3);

	ASSERT_TRUE(i > 0);
	ASSERT_TRUE(i > 1);
	ASSERT_FALSE(i > 2);
	ASSERT_FALSE(1 > i);
	ASSERT_TRUE(2 > i);
	ASSERT_TRUE(3 > i);
}

TEST(OpenInterval, Intersection)
{
	OpenInterval a{ 1, 3 };
	OpenInterval b{ 2, 4 };
	OpenInterval c{ 2, 3 };

	auto i = Intersection(a, b);

	ASSERT_TRUE(i == c);
}
