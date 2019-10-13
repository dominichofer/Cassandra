#include "pch.h"

namespace Flips_test
{
	uint64_t FlipsInOneDirection(const uint64_t P, const uint64_t O, const uint64_t move, const int dx, const int dy)
	{
		uint64_t flips = 0;
		uint64_t x = (move % 8) + dx;
		uint64_t y = (move / 8) + dy;

		while ((x >= 0) && (x < 8) && (y >= 0) && (y < 8)) // In between boundaries
		{
			const uint64_t bit = 1ui64 << (x + 8 * y);
			if (O & bit) // The field belongs to the opponent
				flips |= bit; // Add to possible flips
			else if (P & bit) // The field belongs to the player
				return flips; // All possible flips become real flips
			else // The field belongs to no player
				return 0; // There are no possible flips
			x += dx;
			y += dy;
		}
		return 0;
	}

	uint64_t Flip_loop(const uint64_t P, const uint64_t O, const uint8_t move)
	{
		return FlipsInOneDirection(P, O, move, -1, -1)
		     | FlipsInOneDirection(P, O, move, -1, +0)
		     | FlipsInOneDirection(P, O, move, -1, +1)
		     | FlipsInOneDirection(P, O, move, +0, -1)
		     | FlipsInOneDirection(P, O, move, +0, +1)
		     | FlipsInOneDirection(P, O, move, +1, -1)
		     | FlipsInOneDirection(P, O, move, +1, +0)
		     | FlipsInOneDirection(P, O, move, +1, +1);
	}

	void TestField(const uint8_t move)
	{
		const auto seed = 14;
		std::mt19937_64 rnd_engine(seed);
		auto rnd = [&rnd_engine]() { return std::uniform_int_distribution<uint64_t>(0, 0xFFFFFFFFFFFFFFFFui64)(rnd_engine); };

		for (unsigned int i = 0; i < 10'000; i++)
		{
			const uint64_t p = rnd();
			const uint64_t o = rnd();
			const uint64_t P = (p & ~o) & ~(1ui64 << move);
			const uint64_t O = (o & ~p) & ~(1ui64 << move);

			ASSERT_EQ(Flips(P, O, move), Flip_loop(P, O, move));
		}
	}

	TEST(Flip, Move_A1) { TestField( 0); }
	TEST(Flip, Move_B1) { TestField( 1); }
	TEST(Flip, Move_C1) { TestField( 2); }
	TEST(Flip, Move_D1) { TestField( 3); }
	TEST(Flip, Move_E1) { TestField( 4); }
	TEST(Flip, Move_F1) { TestField( 5); }
	TEST(Flip, Move_G1) { TestField( 6); }
	TEST(Flip, Move_H1) { TestField( 7); }
	TEST(Flip, Move_A2) { TestField( 8); }
	TEST(Flip, Move_B2) { TestField( 9); }
	TEST(Flip, Move_C2) { TestField(10); }
	TEST(Flip, Move_D2) { TestField(11); }
	TEST(Flip, Move_E2) { TestField(12); }
	TEST(Flip, Move_F2) { TestField(13); }
	TEST(Flip, Move_G2) { TestField(14); }
	TEST(Flip, Move_H2) { TestField(15); }
	TEST(Flip, Move_A3) { TestField(16); }
	TEST(Flip, Move_B3) { TestField(17); }
	TEST(Flip, Move_C3) { TestField(18); }
	TEST(Flip, Move_D3) { TestField(19); }
	TEST(Flip, Move_E3) { TestField(20); }
	TEST(Flip, Move_F3) { TestField(21); }
	TEST(Flip, Move_G3) { TestField(22); }
	TEST(Flip, Move_H3) { TestField(23); }
	TEST(Flip, Move_A4) { TestField(24); }
	TEST(Flip, Move_B4) { TestField(25); }
	TEST(Flip, Move_C4) { TestField(26); }
	TEST(Flip, Move_D4) { TestField(27); }
	TEST(Flip, Move_E4) { TestField(28); }
	TEST(Flip, Move_F4) { TestField(29); }
	TEST(Flip, Move_G4) { TestField(30); }
	TEST(Flip, Move_H4) { TestField(31); }
	TEST(Flip, Move_A5) { TestField(32); }
	TEST(Flip, Move_B5) { TestField(33); }
	TEST(Flip, Move_C5) { TestField(34); }
	TEST(Flip, Move_D5) { TestField(35); }
	TEST(Flip, Move_E5) { TestField(36); }
	TEST(Flip, Move_F5) { TestField(37); }
	TEST(Flip, Move_G5) { TestField(38); }
	TEST(Flip, Move_H5) { TestField(39); }
	TEST(Flip, Move_A6) { TestField(40); }
	TEST(Flip, Move_B6) { TestField(41); }
	TEST(Flip, Move_C6) { TestField(42); }
	TEST(Flip, Move_D6) { TestField(43); }
	TEST(Flip, Move_E6) { TestField(44); }
	TEST(Flip, Move_F6) { TestField(45); }
	TEST(Flip, Move_G6) { TestField(46); }
	TEST(Flip, Move_H6) { TestField(47); }
	TEST(Flip, Move_A7) { TestField(48); }
	TEST(Flip, Move_B7) { TestField(49); }
	TEST(Flip, Move_C7) { TestField(50); }
	TEST(Flip, Move_D7) { TestField(51); }
	TEST(Flip, Move_E7) { TestField(52); }
	TEST(Flip, Move_F7) { TestField(53); }
	TEST(Flip, Move_G7) { TestField(54); }
	TEST(Flip, Move_H7) { TestField(55); }
	TEST(Flip, Move_A8) { TestField(56); }
	TEST(Flip, Move_B8) { TestField(57); }
	TEST(Flip, Move_C8) { TestField(58); }
	TEST(Flip, Move_D8) { TestField(59); }
	TEST(Flip, Move_E8) { TestField(60); }
	TEST(Flip, Move_F8) { TestField(61); }
	TEST(Flip, Move_G8) { TestField(62); }
	TEST(Flip, Move_H8) { TestField(63); }
}
