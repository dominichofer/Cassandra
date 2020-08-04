#include "pch.h"
#include "Core/Core.h"

TEST(PossibleMoves, start_position)
{
	ASSERT_EQ(PossibleMoves(Position::Start()), Moves{ 0x0000102004080000ULL });
}

TEST(PossibleMoves, start_position_eth)
{
	ASSERT_EQ(PossibleMoves(Position::StartETH()), Moves{ 0x00000000003C0000ULL });
}

TEST(PossibleMoves, random_samples)
{
	const auto seed = 17;
	PosGen::Random rnd(seed);

	for (int i = 0; i < 100'000; i++)
	{
		const Position pos = rnd();
		Moves potential_moves = pos.Empties();
		BitBoard possible_moves;

		for (auto move : potential_moves)
		{
			if (Flips(pos, move))
				possible_moves.Set(move);
		}

		ASSERT_EQ(PossibleMoves(pos), possible_moves);
	}
}