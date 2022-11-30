#include "pch.h"

TEST(ParseIntensity, certain)
{
	Intensity in{ 10 };
	Intensity out = ParseIntensity(to_string(in));
	ASSERT_EQ(in, out);
}

TEST(ParseIntensity, uncertain)
{
	Intensity in{ 10, Confidence(1.1) };
	Intensity out = ParseIntensity(to_string(in));
	ASSERT_EQ(in, out);
}