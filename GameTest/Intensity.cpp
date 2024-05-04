#include "pch.h"

TEST(Intensity, str_without_confidence_level)
{
	Intensity original{ 10 };
	EXPECT_EQ(original, Intensity::FromString(original.to_string()));
}

TEST(Intensity, str_with_confidence_level)
{
	Intensity original{ 10, 1.0f };
	EXPECT_EQ(original, Intensity::FromString(original.to_string()));
}

TEST(Intensity, compare)
{
	EXPECT_LT(Intensity(5), Intensity(10));
	EXPECT_LT(Intensity(10, 1.0f), Intensity(10));
}

TEST(Intensity, add)
{
	EXPECT_EQ(Intensity(5) + 5, Intensity(10));
	EXPECT_EQ(Intensity(5, 1.0f) + 5, Intensity(10, 1.0f));
}

TEST(Intensity, sub)
{
	EXPECT_EQ(Intensity(10) - 5, Intensity(5));
	EXPECT_EQ(Intensity(10, 1.0f) - 5, Intensity(5, 1.0f));
}

TEST(Intensity, IsExact)
{
	EXPECT_TRUE(Intensity(10).IsExact());
	EXPECT_FALSE(Intensity(10, 1.0f).IsExact());
}
