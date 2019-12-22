#include "pch.h"

TEST(InclusiveInterval, Equality)
{
	InclusiveInterval a{ 1, 2 };
	InclusiveInterval b{ 1, 3 };

	ASSERT_TRUE(a == a);
	ASSERT_FALSE(a != a);
	ASSERT_FALSE(a == b);
	ASSERT_TRUE(a != b);
}

TEST(InclusiveInterval, Compare)
{
	InclusiveInterval a{ 1, 2 };
	InclusiveInterval b{ 2, 3 };
	InclusiveInterval c{ 3, 4 };

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

TEST(InclusiveInterval, Compare_with_score)
{
	InclusiveInterval i{ 1, 2 };

	ASSERT_TRUE(0 < i);
	ASSERT_FALSE(1 < i);
	ASSERT_FALSE(i < 2);
	ASSERT_TRUE(i < 3);

	ASSERT_TRUE(i > 0);
	ASSERT_FALSE(i > 1);
	ASSERT_FALSE(2 > i);
	ASSERT_TRUE(3 > i);
}

TEST(InclusiveInterval, Intersection)
{
	InclusiveInterval a{ 1, 3 };
	InclusiveInterval b{ 2, 4 };
	InclusiveInterval c{ 2, 3 };

	auto i = Intersection(a, b);
	
	ASSERT_TRUE(i == c);
}

TEST(ExclusiveInterval, Equality)
{
	ExclusiveInterval a{ 1, 2 };
	ExclusiveInterval b{ 1, 3 };

	ASSERT_TRUE(a == a);
	ASSERT_FALSE(a != a);
	ASSERT_FALSE(a == b);
	ASSERT_TRUE(a != b);
}

TEST(ExclusiveInterval, Compare)
{
	ExclusiveInterval a{ 1, 2 };
	ExclusiveInterval b{ 2, 4 };
	ExclusiveInterval c{ 3, 5 };

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

TEST(ExclusiveInterval, Compare_with_score)
{
	ExclusiveInterval i{ 1, 2 };

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

TEST(ExclusiveInterval, Intersection)
{
	ExclusiveInterval a{ 1, 3 };
	ExclusiveInterval b{ 2, 4 };
	ExclusiveInterval c{ 2, 3 };

	auto i = Intersection(a, b);

	ASSERT_TRUE(i == c);
}
