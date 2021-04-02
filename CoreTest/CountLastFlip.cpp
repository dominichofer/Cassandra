#include "pch.h"
#include "Core/Core.h"

namespace CountLastFlip_test
{
	void TestField(const Field move)
	{
		const auto seed = 13;
		std::mt19937_64 rnd_engine(seed);
		auto rnd = [&rnd_engine]() { return std::uniform_int_distribution<uint64_t>(0, -1)(rnd_engine); };

		for (int i = 0; i < 100'000; i++)
		{
			const auto r = rnd();
			const auto P = r & ~BitBoard(move);
			const auto O = ~r & ~BitBoard(move);
			Position pos(P, O);
			ASSERT_EQ(popcount(Flips(pos, move)), CountLastFlip(pos, move));
		}
	}

	TEST(CountLastFlip, Move_A1) { TestField(Field::A1); }
	TEST(CountLastFlip, Move_B1) { TestField(Field::B1); }
	TEST(CountLastFlip, Move_C1) { TestField(Field::C1); }
	TEST(CountLastFlip, Move_D1) { TestField(Field::D1); }
	TEST(CountLastFlip, Move_E1) { TestField(Field::E1); }
	TEST(CountLastFlip, Move_F1) { TestField(Field::F1); }
	TEST(CountLastFlip, Move_G1) { TestField(Field::G1); }
	TEST(CountLastFlip, Move_H1) { TestField(Field::H1); }
	TEST(CountLastFlip, Move_A2) { TestField(Field::A2); }
	TEST(CountLastFlip, Move_B2) { TestField(Field::B2); }
	TEST(CountLastFlip, Move_C2) { TestField(Field::C2); }
	TEST(CountLastFlip, Move_D2) { TestField(Field::D2); }
	TEST(CountLastFlip, Move_E2) { TestField(Field::E2); }
	TEST(CountLastFlip, Move_F2) { TestField(Field::F2); }
	TEST(CountLastFlip, Move_G2) { TestField(Field::G2); }
	TEST(CountLastFlip, Move_H2) { TestField(Field::H2); }
	TEST(CountLastFlip, Move_A3) { TestField(Field::A3); }
	TEST(CountLastFlip, Move_B3) { TestField(Field::B3); }
	TEST(CountLastFlip, Move_C3) { TestField(Field::C3); }
	TEST(CountLastFlip, Move_D3) { TestField(Field::D3); }
	TEST(CountLastFlip, Move_E3) { TestField(Field::E3); }
	TEST(CountLastFlip, Move_F3) { TestField(Field::F3); }
	TEST(CountLastFlip, Move_G3) { TestField(Field::G3); }
	TEST(CountLastFlip, Move_H3) { TestField(Field::H3); }
	TEST(CountLastFlip, Move_A4) { TestField(Field::A4); }
	TEST(CountLastFlip, Move_B4) { TestField(Field::B4); }
	TEST(CountLastFlip, Move_C4) { TestField(Field::C4); }
	TEST(CountLastFlip, Move_D4) { TestField(Field::D4); }
	TEST(CountLastFlip, Move_E4) { TestField(Field::E4); }
	TEST(CountLastFlip, Move_F4) { TestField(Field::F4); }
	TEST(CountLastFlip, Move_G4) { TestField(Field::G4); }
	TEST(CountLastFlip, Move_H4) { TestField(Field::H4); }
	TEST(CountLastFlip, Move_A5) { TestField(Field::A5); }
	TEST(CountLastFlip, Move_B5) { TestField(Field::B5); }
	TEST(CountLastFlip, Move_C5) { TestField(Field::C5); }
	TEST(CountLastFlip, Move_D5) { TestField(Field::D5); }
	TEST(CountLastFlip, Move_E5) { TestField(Field::E5); }
	TEST(CountLastFlip, Move_F5) { TestField(Field::F5); }
	TEST(CountLastFlip, Move_G5) { TestField(Field::G5); }
	TEST(CountLastFlip, Move_H5) { TestField(Field::H5); }
	TEST(CountLastFlip, Move_A6) { TestField(Field::A6); }
	TEST(CountLastFlip, Move_B6) { TestField(Field::B6); }
	TEST(CountLastFlip, Move_C6) { TestField(Field::C6); }
	TEST(CountLastFlip, Move_D6) { TestField(Field::D6); }
	TEST(CountLastFlip, Move_E6) { TestField(Field::E6); }
	TEST(CountLastFlip, Move_F6) { TestField(Field::F6); }
	TEST(CountLastFlip, Move_G6) { TestField(Field::G6); }
	TEST(CountLastFlip, Move_H6) { TestField(Field::H6); }
	TEST(CountLastFlip, Move_A7) { TestField(Field::A7); }
	TEST(CountLastFlip, Move_B7) { TestField(Field::B7); }
	TEST(CountLastFlip, Move_C7) { TestField(Field::C7); }
	TEST(CountLastFlip, Move_D7) { TestField(Field::D7); }
	TEST(CountLastFlip, Move_E7) { TestField(Field::E7); }
	TEST(CountLastFlip, Move_F7) { TestField(Field::F7); }
	TEST(CountLastFlip, Move_G7) { TestField(Field::G7); }
	TEST(CountLastFlip, Move_H7) { TestField(Field::H7); }
	TEST(CountLastFlip, Move_A8) { TestField(Field::A8); }
	TEST(CountLastFlip, Move_B8) { TestField(Field::B8); }
	TEST(CountLastFlip, Move_C8) { TestField(Field::C8); }
	TEST(CountLastFlip, Move_D8) { TestField(Field::D8); }
	TEST(CountLastFlip, Move_E8) { TestField(Field::E8); }
	TEST(CountLastFlip, Move_F8) { TestField(Field::F8); }
	TEST(CountLastFlip, Move_G8) { TestField(Field::G8); }
	TEST(CountLastFlip, Move_H8) { TestField(Field::H8); }
}
