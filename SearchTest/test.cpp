#include "pch.h"

using namespace Search;

TEST(Selectivity, None_represents_infinit_sigmas)
{
	ASSERT_TRUE(Selectivity::None == Selectivity(std::numeric_limits<decltype(Selectivity::quantile)>::infinity()));
}

TEST(Selectivity, Infinit_represents_zero_sigmas)
{
	ASSERT_TRUE(Selectivity::Infinit == Selectivity(0));
}

TEST(Selectivity, Equality)
{
	constexpr auto inf = std::numeric_limits<decltype(Selectivity::quantile)>::infinity();

	ASSERT_TRUE(Selectivity(0 /*sigmas*/) == Selectivity(0 /*sigmas*/));
	ASSERT_TRUE(Selectivity(1 /*sigmas*/) == Selectivity(1 /*sigmas*/));
	ASSERT_TRUE(Selectivity(inf /*sigmas*/) == Selectivity(inf /*sigmas*/));

	ASSERT_FALSE(Selectivity(0 /*sigmas*/) != Selectivity(0 /*sigmas*/));
	ASSERT_FALSE(Selectivity(1 /*sigmas*/) != Selectivity(1 /*sigmas*/));
	ASSERT_FALSE(Selectivity(inf /*sigmas*/) != Selectivity(inf /*sigmas*/));
}

TEST(Selectivity, Bigger_sigma_is_less_selective)
{
	ASSERT_TRUE(Selectivity(1 /*sigmas*/) > Selectivity(2 /*sigmas*/));
}

TEST(Selectivity, Compare)
{
	constexpr auto inf = std::numeric_limits<decltype(Selectivity::quantile)>::infinity();

	ASSERT_TRUE(Selectivity(1 /*sigmas*/) < Selectivity(0 /*sigmas*/));
	ASSERT_TRUE(Selectivity(inf /*sigmas*/) < Selectivity(0 /*sigmas*/));

	ASSERT_TRUE(Selectivity(0 /*sigmas*/) > Selectivity(1 /*sigmas*/));
	ASSERT_TRUE(Selectivity(0 /*sigmas*/) > Selectivity(inf /*sigmas*/));
}

TEST(Intensity, Negation_flips_window)
{
	Intensity i1{ OpenInterval(+2, +4), 0, Selectivity(0) };
	Intensity i2{ OpenInterval(-4, -2), 0, Selectivity(0) };

	ASSERT_TRUE(i1 == -i2);
}

TEST(Intensity, Subtraction_of_depth)
{
	Intensity i1{ OpenInterval(+2, +4), 0, Selectivity(0) };
	Intensity i2{ OpenInterval(+2, +4), 1, Selectivity(0) };

	ASSERT_EQ(i1, i2 - 1);
}
