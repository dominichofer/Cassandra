#include "pch.h"

TEST(PossibleMoves, start_position)
{
	ASSERT_EQ(PossibleMoves(0x0000'0008'1000'0000ULL, 0x0000'0010'0800'0000ULL), 0x0000102004080000ULL);
}

TEST(PossibleMoves, start_position_eth)
{
	ASSERT_EQ(PossibleMoves(0x0000'0018'0000'0000ULL, 0x0000'0000'1800'0000ULL), 0x00000000003C0000ULL);
}

TEST(PossibleMoves, random_samples)
{
	const auto seed = 17;
	std::mt19937_64 rnd_engine(seed);
	auto rnd = [&rnd_engine]() { return std::uniform_int_distribution<uint64_t>(0, 0xFFFFFFFFFFFFFFFFULL)(rnd_engine); };

	for (unsigned int i = 0; i < 100'000; i++)
	{
		const uint64_t p = rnd();
		const uint64_t o = rnd();
		const uint64_t P = p & ~o;
		const uint64_t O = o & ~p;

		uint64_t potential_moves = ~(P | O);
		uint64_t possible_moves = 0;

		while (potential_moves)
		{
			const uint64_t move = BitScanLSB(potential_moves);
			RemoveLSB(potential_moves);
			if (Flips(P, O, move))
				possible_moves |=  1ULL << move;
		}

		ASSERT_EQ(PossibleMoves(P, O), possible_moves);
	}
}