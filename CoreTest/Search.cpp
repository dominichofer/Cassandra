#include "pch.h"

using namespace Search;

TEST(EvalGameOver, full_of_player)
{
	Position pos =
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"_pos;

	ASSERT_EQ(EvalGameOver(pos), +64);
}

TEST(EvalGameOver, full_of_opponent)
{
	Position pos =
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"_pos;

	ASSERT_EQ(EvalGameOver(pos), -64);
}

TEST(EvalGameOver, half_half)
{
	Position pos =
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"
		"O O O O O O O O"_pos;

	ASSERT_EQ(EvalGameOver(pos), 0);
}

TEST(EvalGameOver, empties_count_toward_player)
{
	Position pos =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - X X - - -"
		"- - - X X - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pos;

	ASSERT_EQ(EvalGameOver(pos), +64);
}

TEST(EvalGameOver, empties_count_toward_opponent)
{
	Position pos =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - O O - - -"
		"- - - O O - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pos;

	ASSERT_EQ(EvalGameOver(pos), -64);
}

TEST(OpenInterval, Equality1)
{
	OpenInterval w1(1, 10);
	OpenInterval w2(2, 10);

	ASSERT_TRUE(w1 == w1);
	ASSERT_FALSE(w1 != w1);
	ASSERT_TRUE(w1 != w2);
	ASSERT_FALSE(w1 == w2);
}

TEST(OpenInterval, Negation_flips_window)
{
	OpenInterval w1(1, 10);
	OpenInterval w2(-10, -1);

	ASSERT_TRUE(-w1 == w2);
}

TEST(OpenInterval, Contains_Score)
{
	OpenInterval w(1, 10);

	ASSERT_FALSE(w.Contains(0));
	ASSERT_FALSE(w.Contains(1));
	ASSERT_TRUE(w.Contains(2));
	ASSERT_TRUE(w.Contains(9));
	ASSERT_FALSE(w.Contains(10));
	ASSERT_FALSE(w.Contains(11));
}

TEST(OpenInterval, Compares_to_Score)
{
	OpenInterval w(1, 10);

	ASSERT_TRUE(w > 0);
	ASSERT_TRUE(w > 1);
	ASSERT_FALSE(w > 2);
	ASSERT_FALSE(w < 9);
	ASSERT_TRUE(w < 10);
	ASSERT_TRUE(w < 11);
}

TEST(OpenInterval, Compares_to_self)
{
	OpenInterval w(1, 10);

	ASSERT_TRUE(w > OpenInterval(-10, 0));
	ASSERT_TRUE(w > OpenInterval(-10, 1));
	ASSERT_FALSE(w > OpenInterval(-10, 20));
	ASSERT_FALSE(w < OpenInterval(9, 20));
	ASSERT_TRUE(w < OpenInterval(10, 20));
	ASSERT_TRUE(w < OpenInterval(11, 20));
}

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
	Intensity i1{ OpenInterval{+1,+2}, 0, Selectivity{0} };
	Intensity i2{ OpenInterval{-2,-1}, 0, Selectivity{0} };

	ASSERT_TRUE(i1 == -i2);
}

TEST(Intensity, Subtraction_of_depth)
{
	Intensity i1{ OpenInterval{+1,+2}, 0, Selectivity{0} };
	Intensity i2{ OpenInterval{+1,+2}, 1, Selectivity{0} };

	ASSERT_TRUE(i1 == i2 - 1);
}
