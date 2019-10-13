#include "pch.h"

namespace IntegrationTests
{
	TEST(CountLastFlip, abstracts_bits_away)
	{
		Position pos =
			"X X X X X X X X"
			"X X X X X X X X"
			"X X X X X X X X"
			"X X X X X X X X"
			"X X X X X X X X"
			"X X X X X X X X"
			"X X X X X X X X"
			"X O O O O O O  "_pos;
		ASSERT_EQ(CountLastFlip(pos, Field::A1), 2 * 6);
	}

	TEST(Play, abstracts_bits_away)
	{
		Position in =
			"               "
			"               "
			"               "
			"      O O      "
			"      O O      "
			"               "
			"               "
			"X O O O O O O  "_pos;
		Position out =
			"               "
			"               "
			"               "
			"      X X      "
			"      X X      "
			"               "
			"               "
			"O O O O O O O O"_pos;

		ASSERT_EQ(Play(in, Field::A1), out);
	}

	TEST(PlayPass, abstracts_bits_away)
	{
		Position in =
			"               "
			"               "
			"               "
			"      X O      "
			"      X O      "
			"               "
			"               "
			"               "_pos;
		Position out =
			"               "
			"               "
			"               "
			"      O X      "
			"      O X      "
			"               "
			"               "
			"               "_pos;

		ASSERT_EQ(PlayPass(in), out);
	}

	TEST(PossibleMoves, abstracts_bits_away)
	{
		Position in =
			"               "
			"               "
			"               "
			"      O O      "
			"      O O      "
			"               "
			"               "
			"X O O O O O O  "_pos;
		Moves out =
			"               "
			"               "
			"               "
			"               "
			"               "
			"               "
			"               "
			"              ."_mov;

		ASSERT_EQ(PossibleMoves(in), out);
	}
}
