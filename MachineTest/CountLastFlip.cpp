#include "pch.h"

namespace CountLastFlip_test
{
	void TestField(const uint8_t move)
	{
		const auto seed = 13;
		std::mt19937_64 rnd_engine(seed);
		auto rnd = [&rnd_engine]() { return std::uniform_int_distribution<uint64_t>(0, 0xFFFFFFFFFFFFFFFFui64)(rnd_engine); };

		for (unsigned int i = 0; i < 100'000; i++)
		{
			const auto r = rnd();
			const auto P = r & ~Bit(move);
			const auto O = ~P & ~Bit(move);
			ASSERT_EQ(2 * PopCount(Flips(P, O, move)), CountLastFlip(P, move));
		}
	}

	TEST(CountLastFlip, Move_A1) { TestField( 0); }
	TEST(CountLastFlip, Move_B1) { TestField( 1); }
	TEST(CountLastFlip, Move_C1) { TestField( 2); }
	TEST(CountLastFlip, Move_D1) { TestField( 3); }
	TEST(CountLastFlip, Move_E1) { TestField( 4); }
	TEST(CountLastFlip, Move_F1) { TestField( 5); }
	TEST(CountLastFlip, Move_G1) { TestField( 6); }
	TEST(CountLastFlip, Move_H1) { TestField( 7); }
	TEST(CountLastFlip, Move_A2) { TestField( 8); }
	TEST(CountLastFlip, Move_B2) { TestField( 9); }
	TEST(CountLastFlip, Move_C2) { TestField(10); }
	TEST(CountLastFlip, Move_D2) { TestField(11); }
	TEST(CountLastFlip, Move_E2) { TestField(12); }
	TEST(CountLastFlip, Move_F2) { TestField(13); }
	TEST(CountLastFlip, Move_G2) { TestField(14); }
	TEST(CountLastFlip, Move_H2) { TestField(15); }
	TEST(CountLastFlip, Move_A3) { TestField(16); }
	TEST(CountLastFlip, Move_B3) { TestField(17); }
	TEST(CountLastFlip, Move_C3) { TestField(18); }
	TEST(CountLastFlip, Move_D3) { TestField(19); }
	TEST(CountLastFlip, Move_E3) { TestField(20); }
	TEST(CountLastFlip, Move_F3) { TestField(21); }
	TEST(CountLastFlip, Move_G3) { TestField(22); }
	TEST(CountLastFlip, Move_H3) { TestField(23); }
	TEST(CountLastFlip, Move_A4) { TestField(24); }
	TEST(CountLastFlip, Move_B4) { TestField(25); }
	TEST(CountLastFlip, Move_C4) { TestField(26); }
	TEST(CountLastFlip, Move_D4) { TestField(27); }
	TEST(CountLastFlip, Move_E4) { TestField(28); }
	TEST(CountLastFlip, Move_F4) { TestField(29); }
	TEST(CountLastFlip, Move_G4) { TestField(30); }
	TEST(CountLastFlip, Move_H4) { TestField(31); }
	TEST(CountLastFlip, Move_A5) { TestField(32); }
	TEST(CountLastFlip, Move_B5) { TestField(33); }
	TEST(CountLastFlip, Move_C5) { TestField(34); }
	TEST(CountLastFlip, Move_D5) { TestField(35); }
	TEST(CountLastFlip, Move_E5) { TestField(36); }
	TEST(CountLastFlip, Move_F5) { TestField(37); }
	TEST(CountLastFlip, Move_G5) { TestField(38); }
	TEST(CountLastFlip, Move_H5) { TestField(39); }
	TEST(CountLastFlip, Move_A6) { TestField(40); }
	TEST(CountLastFlip, Move_B6) { TestField(41); }
	TEST(CountLastFlip, Move_C6) { TestField(42); }
	TEST(CountLastFlip, Move_D6) { TestField(43); }
	TEST(CountLastFlip, Move_E6) { TestField(44); }
	TEST(CountLastFlip, Move_F6) { TestField(45); }
	TEST(CountLastFlip, Move_G6) { TestField(46); }
	TEST(CountLastFlip, Move_H6) { TestField(47); }
	TEST(CountLastFlip, Move_A7) { TestField(48); }
	TEST(CountLastFlip, Move_B7) { TestField(49); }
	TEST(CountLastFlip, Move_C7) { TestField(50); }
	TEST(CountLastFlip, Move_D7) { TestField(51); }
	TEST(CountLastFlip, Move_E7) { TestField(52); }
	TEST(CountLastFlip, Move_F7) { TestField(53); }
	TEST(CountLastFlip, Move_G7) { TestField(54); }
	TEST(CountLastFlip, Move_H7) { TestField(55); }
	TEST(CountLastFlip, Move_A8) { TestField(56); }
	TEST(CountLastFlip, Move_B8) { TestField(57); }
	TEST(CountLastFlip, Move_C8) { TestField(58); }
	TEST(CountLastFlip, Move_D8) { TestField(59); }
	TEST(CountLastFlip, Move_E8) { TestField(60); }
	TEST(CountLastFlip, Move_F8) { TestField(61); }
	TEST(CountLastFlip, Move_G8) { TestField(62); }
	TEST(CountLastFlip, Move_H8) { TestField(63); }
}
