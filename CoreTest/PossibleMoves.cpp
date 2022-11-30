#include "pch.h"

TEST(PossibleMoves, start_position)
{
	ASSERT_EQ(PossibleMoves(Position::Start()), Moves{ 0x0000102004080000ULL });
}

TEST(PossibleMoves, random_samples)
{
	RandomPositionGenerator rnd(/*seed*/ 127);

	for (int i = 0; i < 100'000; i++)
	{
		Position pos = rnd();
		Moves potential_moves{ pos.Empties() };
		BitBoard possible_moves;

		for (const auto& move : potential_moves)
		{
			if (Flips(pos, move))
				possible_moves.Set(move);
		}

		ASSERT_EQ(PossibleMoves(pos), possible_moves);
	}
}