#include "pch.h"
#include <cstdint>

#if defined(__AVX512F__)
TEST(PossibleMoves, AVX512_start_position)
{
	ASSERT_EQ(detail::PossibleMoves_AVX512(Position::Start()), Moves{ 0x0000102004080000ULL });
}

TEST(PossibleMoves, AVX512_random_samples)
{
	RandomPositionGenerator rnd(/*seed*/ 127);

	for (int i = 0; i < 100'000; i++)
	{
		Position pos = rnd();
		Moves potential_moves{ pos.Empties() };
		uint64_t possible_moves = 0;

		for (Field move : potential_moves)
			if (Flips(pos, move))
				possible_moves |= 1ULL << static_cast<uint8_t>(move);

		ASSERT_EQ(detail::PossibleMoves_AVX512(pos), possible_moves);
	}
}
#endif

#if defined(__AVX512F__) || defined(__AVX2__)
TEST(PossibleMoves, AVX2_start_position)
{
	ASSERT_EQ(detail::PossibleMoves_AVX2(Position::Start()), Moves{ 0x0000102004080000ULL });
}

TEST(PossibleMoves, AVX2_random_samples)
{
	RandomPositionGenerator rnd(/*seed*/ 127);

	for (int i = 0; i < 100'000; i++)
	{
		Position pos = rnd();
		Moves potential_moves{ pos.Empties() };
		uint64_t possible_moves = 0;

		for (Field move : potential_moves)
			if (Flips(pos, move))
				possible_moves |= 1ULL << static_cast<uint8_t>(move);

		ASSERT_EQ(detail::PossibleMoves_AVX2(pos), possible_moves);
	}
}
#endif

TEST(PossibleMoves, x64_start_position)
{
	ASSERT_EQ(detail::PossibleMoves_x64(Position::Start()), Moves{ 0x0000102004080000ULL });
}

TEST(PossibleMoves, x64_random_samples)
{
	RandomPositionGenerator rnd(/*seed*/ 127);

	for (int i = 0; i < 100'000; i++)
	{
		Position pos = rnd();
		Moves potential_moves{ pos.Empties() };
		uint64_t possible_moves = 0;

		for (Field move : potential_moves)
			if (Flips(pos, move))
				possible_moves |= 1ULL << static_cast<uint8_t>(move);

		ASSERT_EQ(detail::PossibleMoves_x64(pos), possible_moves);
	}
}