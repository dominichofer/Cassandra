#include "pch.h"
#include <cstdint>

namespace Flips_test
{
	uint64_t FlipsInOneDirection(const Position& pos, Field move, int dx, int dy)
	{
		uint64_t flips = 0;
		int x = (std::to_underlying(move) % 8) + dx;
		int y = (std::to_underlying(move) / 8) + dy;

		while ((x >= 0) && (x < 8) && (y >= 0) && (y < 8)) // In between boundaries
		{
			uint64_t bit = 1ULL << (x + 8 * y);
			if (pos.Opponent() & bit) // The field belongs to the opponent
				flips |= bit; // Add to potential flips
			else if (pos.Player() & bit) // The field belongs to the player
				return flips; // All potential flips become real flips
			else // The field belongs to no player
				break; // There are no possible flips
			x += dx;
			y += dy;
		}
		return 0;
	}

	uint64_t Flip_loop(const Position& pos, Field move)
	{
		return FlipsInOneDirection(pos, move, -1, -1)
		     | FlipsInOneDirection(pos, move, -1, +0)
		     | FlipsInOneDirection(pos, move, -1, +1)
		     | FlipsInOneDirection(pos, move, +0, -1)
		     | FlipsInOneDirection(pos, move, +0, +1)
		     | FlipsInOneDirection(pos, move, +1, -1)
		     | FlipsInOneDirection(pos, move, +1, +0)
		     | FlipsInOneDirection(pos, move, +1, +1);
	}

	void TestField_x64(Field move)
	{
		RandomPositionGenerator rnd(/*seed*/ 14);

		for (int i = 0; i < 10'000; i++)
		{
			Position pos = rnd();
			ASSERT_EQ(detail::Flips_x64(pos, move), Flip_loop(pos, move));
		}
	}

	TEST(Flip_x64, Move_A1) { TestField_x64(Field::A1); }
	TEST(Flip_x64, Move_B1) { TestField_x64(Field::B1); }
	TEST(Flip_x64, Move_C1) { TestField_x64(Field::C1); }
	TEST(Flip_x64, Move_D1) { TestField_x64(Field::D1); }
	TEST(Flip_x64, Move_E1) { TestField_x64(Field::E1); }
	TEST(Flip_x64, Move_F1) { TestField_x64(Field::F1); }
	TEST(Flip_x64, Move_G1) { TestField_x64(Field::G1); }
	TEST(Flip_x64, Move_H1) { TestField_x64(Field::H1); }
	TEST(Flip_x64, Move_A2) { TestField_x64(Field::A2); }
	TEST(Flip_x64, Move_B2) { TestField_x64(Field::B2); }
	TEST(Flip_x64, Move_C2) { TestField_x64(Field::C2); }
	TEST(Flip_x64, Move_D2) { TestField_x64(Field::D2); }
	TEST(Flip_x64, Move_E2) { TestField_x64(Field::E2); }
	TEST(Flip_x64, Move_F2) { TestField_x64(Field::F2); }
	TEST(Flip_x64, Move_G2) { TestField_x64(Field::G2); }
	TEST(Flip_x64, Move_H2) { TestField_x64(Field::H2); }
	TEST(Flip_x64, Move_A3) { TestField_x64(Field::A3); }
	TEST(Flip_x64, Move_B3) { TestField_x64(Field::B3); }
	TEST(Flip_x64, Move_C3) { TestField_x64(Field::C3); }
	TEST(Flip_x64, Move_D3) { TestField_x64(Field::D3); }
	TEST(Flip_x64, Move_E3) { TestField_x64(Field::E3); }
	TEST(Flip_x64, Move_F3) { TestField_x64(Field::F3); }
	TEST(Flip_x64, Move_G3) { TestField_x64(Field::G3); }
	TEST(Flip_x64, Move_H3) { TestField_x64(Field::H3); }
	TEST(Flip_x64, Move_A4) { TestField_x64(Field::A4); }
	TEST(Flip_x64, Move_B4) { TestField_x64(Field::B4); }
	TEST(Flip_x64, Move_C4) { TestField_x64(Field::C4); }
	TEST(Flip_x64, Move_D4) { TestField_x64(Field::D4); }
	TEST(Flip_x64, Move_E4) { TestField_x64(Field::E4); }
	TEST(Flip_x64, Move_F4) { TestField_x64(Field::F4); }
	TEST(Flip_x64, Move_G4) { TestField_x64(Field::G4); }
	TEST(Flip_x64, Move_H4) { TestField_x64(Field::H4); }
	TEST(Flip_x64, Move_A5) { TestField_x64(Field::A5); }
	TEST(Flip_x64, Move_B5) { TestField_x64(Field::B5); }
	TEST(Flip_x64, Move_C5) { TestField_x64(Field::C5); }
	TEST(Flip_x64, Move_D5) { TestField_x64(Field::D5); }
	TEST(Flip_x64, Move_E5) { TestField_x64(Field::E5); }
	TEST(Flip_x64, Move_F5) { TestField_x64(Field::F5); }
	TEST(Flip_x64, Move_G5) { TestField_x64(Field::G5); }
	TEST(Flip_x64, Move_H5) { TestField_x64(Field::H5); }
	TEST(Flip_x64, Move_A6) { TestField_x64(Field::A6); }
	TEST(Flip_x64, Move_B6) { TestField_x64(Field::B6); }
	TEST(Flip_x64, Move_C6) { TestField_x64(Field::C6); }
	TEST(Flip_x64, Move_D6) { TestField_x64(Field::D6); }
	TEST(Flip_x64, Move_E6) { TestField_x64(Field::E6); }
	TEST(Flip_x64, Move_F6) { TestField_x64(Field::F6); }
	TEST(Flip_x64, Move_G6) { TestField_x64(Field::G6); }
	TEST(Flip_x64, Move_H6) { TestField_x64(Field::H6); }
	TEST(Flip_x64, Move_A7) { TestField_x64(Field::A7); }
	TEST(Flip_x64, Move_B7) { TestField_x64(Field::B7); }
	TEST(Flip_x64, Move_C7) { TestField_x64(Field::C7); }
	TEST(Flip_x64, Move_D7) { TestField_x64(Field::D7); }
	TEST(Flip_x64, Move_E7) { TestField_x64(Field::E7); }
	TEST(Flip_x64, Move_F7) { TestField_x64(Field::F7); }
	TEST(Flip_x64, Move_G7) { TestField_x64(Field::G7); }
	TEST(Flip_x64, Move_H7) { TestField_x64(Field::H7); }
	TEST(Flip_x64, Move_A8) { TestField_x64(Field::A8); }
	TEST(Flip_x64, Move_B8) { TestField_x64(Field::B8); }
	TEST(Flip_x64, Move_C8) { TestField_x64(Field::C8); }
	TEST(Flip_x64, Move_D8) { TestField_x64(Field::D8); }
	TEST(Flip_x64, Move_E8) { TestField_x64(Field::E8); }
	TEST(Flip_x64, Move_F8) { TestField_x64(Field::F8); }
	TEST(Flip_x64, Move_G8) { TestField_x64(Field::G8); }
	TEST(Flip_x64, Move_H8) { TestField_x64(Field::H8); }


#ifdef __AVX2__
	void TestField_AVX2(Field move)
	{
		RandomPositionGenerator rnd(/*seed*/ 14);

		for (int i = 0; i < 10'000; i++)
		{
			Position pos = rnd();
			ASSERT_EQ(detail::Flips_AVX2(pos, move), Flip_loop(pos, move));
		}
	}
	TEST(Flip_AVX2, Move_A1) { TestField_AVX2(Field::A1); }
	TEST(Flip_AVX2, Move_B1) { TestField_AVX2(Field::B1); }
	TEST(Flip_AVX2, Move_C1) { TestField_AVX2(Field::C1); }
	TEST(Flip_AVX2, Move_D1) { TestField_AVX2(Field::D1); }
	TEST(Flip_AVX2, Move_E1) { TestField_AVX2(Field::E1); }
	TEST(Flip_AVX2, Move_F1) { TestField_AVX2(Field::F1); }
	TEST(Flip_AVX2, Move_G1) { TestField_AVX2(Field::G1); }
	TEST(Flip_AVX2, Move_H1) { TestField_AVX2(Field::H1); }
	TEST(Flip_AVX2, Move_A2) { TestField_AVX2(Field::A2); }
	TEST(Flip_AVX2, Move_B2) { TestField_AVX2(Field::B2); }
	TEST(Flip_AVX2, Move_C2) { TestField_AVX2(Field::C2); }
	TEST(Flip_AVX2, Move_D2) { TestField_AVX2(Field::D2); }
	TEST(Flip_AVX2, Move_E2) { TestField_AVX2(Field::E2); }
	TEST(Flip_AVX2, Move_F2) { TestField_AVX2(Field::F2); }
	TEST(Flip_AVX2, Move_G2) { TestField_AVX2(Field::G2); }
	TEST(Flip_AVX2, Move_H2) { TestField_AVX2(Field::H2); }
	TEST(Flip_AVX2, Move_A3) { TestField_AVX2(Field::A3); }
	TEST(Flip_AVX2, Move_B3) { TestField_AVX2(Field::B3); }
	TEST(Flip_AVX2, Move_C3) { TestField_AVX2(Field::C3); }
	TEST(Flip_AVX2, Move_D3) { TestField_AVX2(Field::D3); }
	TEST(Flip_AVX2, Move_E3) { TestField_AVX2(Field::E3); }
	TEST(Flip_AVX2, Move_F3) { TestField_AVX2(Field::F3); }
	TEST(Flip_AVX2, Move_G3) { TestField_AVX2(Field::G3); }
	TEST(Flip_AVX2, Move_H3) { TestField_AVX2(Field::H3); }
	TEST(Flip_AVX2, Move_A4) { TestField_AVX2(Field::A4); }
	TEST(Flip_AVX2, Move_B4) { TestField_AVX2(Field::B4); }
	TEST(Flip_AVX2, Move_C4) { TestField_AVX2(Field::C4); }
	TEST(Flip_AVX2, Move_D4) { TestField_AVX2(Field::D4); }
	TEST(Flip_AVX2, Move_E4) { TestField_AVX2(Field::E4); }
	TEST(Flip_AVX2, Move_F4) { TestField_AVX2(Field::F4); }
	TEST(Flip_AVX2, Move_G4) { TestField_AVX2(Field::G4); }
	TEST(Flip_AVX2, Move_H4) { TestField_AVX2(Field::H4); }
	TEST(Flip_AVX2, Move_A5) { TestField_AVX2(Field::A5); }
	TEST(Flip_AVX2, Move_B5) { TestField_AVX2(Field::B5); }
	TEST(Flip_AVX2, Move_C5) { TestField_AVX2(Field::C5); }
	TEST(Flip_AVX2, Move_D5) { TestField_AVX2(Field::D5); }
	TEST(Flip_AVX2, Move_E5) { TestField_AVX2(Field::E5); }
	TEST(Flip_AVX2, Move_F5) { TestField_AVX2(Field::F5); }
	TEST(Flip_AVX2, Move_G5) { TestField_AVX2(Field::G5); }
	TEST(Flip_AVX2, Move_H5) { TestField_AVX2(Field::H5); }
	TEST(Flip_AVX2, Move_A6) { TestField_AVX2(Field::A6); }
	TEST(Flip_AVX2, Move_B6) { TestField_AVX2(Field::B6); }
	TEST(Flip_AVX2, Move_C6) { TestField_AVX2(Field::C6); }
	TEST(Flip_AVX2, Move_D6) { TestField_AVX2(Field::D6); }
	TEST(Flip_AVX2, Move_E6) { TestField_AVX2(Field::E6); }
	TEST(Flip_AVX2, Move_F6) { TestField_AVX2(Field::F6); }
	TEST(Flip_AVX2, Move_G6) { TestField_AVX2(Field::G6); }
	TEST(Flip_AVX2, Move_H6) { TestField_AVX2(Field::H6); }
	TEST(Flip_AVX2, Move_A7) { TestField_AVX2(Field::A7); }
	TEST(Flip_AVX2, Move_B7) { TestField_AVX2(Field::B7); }
	TEST(Flip_AVX2, Move_C7) { TestField_AVX2(Field::C7); }
	TEST(Flip_AVX2, Move_D7) { TestField_AVX2(Field::D7); }
	TEST(Flip_AVX2, Move_E7) { TestField_AVX2(Field::E7); }
	TEST(Flip_AVX2, Move_F7) { TestField_AVX2(Field::F7); }
	TEST(Flip_AVX2, Move_G7) { TestField_AVX2(Field::G7); }
	TEST(Flip_AVX2, Move_H7) { TestField_AVX2(Field::H7); }
	TEST(Flip_AVX2, Move_A8) { TestField_AVX2(Field::A8); }
	TEST(Flip_AVX2, Move_B8) { TestField_AVX2(Field::B8); }
	TEST(Flip_AVX2, Move_C8) { TestField_AVX2(Field::C8); }
	TEST(Flip_AVX2, Move_D8) { TestField_AVX2(Field::D8); }
	TEST(Flip_AVX2, Move_E8) { TestField_AVX2(Field::E8); }
	TEST(Flip_AVX2, Move_F8) { TestField_AVX2(Field::F8); }
	TEST(Flip_AVX2, Move_G8) { TestField_AVX2(Field::G8); }
	TEST(Flip_AVX2, Move_H8) { TestField_AVX2(Field::H8); }
#endif
}
