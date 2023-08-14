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

	void TestField(Field move)
	{
		RandomPositionGenerator rnd(/*seed*/ 14);

		for (int i = 0; i < 10'000; i++)
		{
			Position pos = rnd();
			ASSERT_EQ(Flips(pos, move), Flip_loop(pos, move));
		}
	}

	TEST(Flip, Move_A1) { TestField(Field::A1); }
	TEST(Flip, Move_B1) { TestField(Field::B1); }
	TEST(Flip, Move_C1) { TestField(Field::C1); }
	TEST(Flip, Move_D1) { TestField(Field::D1); }
	TEST(Flip, Move_E1) { TestField(Field::E1); }
	TEST(Flip, Move_F1) { TestField(Field::F1); }
	TEST(Flip, Move_G1) { TestField(Field::G1); }
	TEST(Flip, Move_H1) { TestField(Field::H1); }
	TEST(Flip, Move_A2) { TestField(Field::A2); }
	TEST(Flip, Move_B2) { TestField(Field::B2); }
	TEST(Flip, Move_C2) { TestField(Field::C2); }
	TEST(Flip, Move_D2) { TestField(Field::D2); }
	TEST(Flip, Move_E2) { TestField(Field::E2); }
	TEST(Flip, Move_F2) { TestField(Field::F2); }
	TEST(Flip, Move_G2) { TestField(Field::G2); }
	TEST(Flip, Move_H2) { TestField(Field::H2); }
	TEST(Flip, Move_A3) { TestField(Field::A3); }
	TEST(Flip, Move_B3) { TestField(Field::B3); }
	TEST(Flip, Move_C3) { TestField(Field::C3); }
	TEST(Flip, Move_D3) { TestField(Field::D3); }
	TEST(Flip, Move_E3) { TestField(Field::E3); }
	TEST(Flip, Move_F3) { TestField(Field::F3); }
	TEST(Flip, Move_G3) { TestField(Field::G3); }
	TEST(Flip, Move_H3) { TestField(Field::H3); }
	TEST(Flip, Move_A4) { TestField(Field::A4); }
	TEST(Flip, Move_B4) { TestField(Field::B4); }
	TEST(Flip, Move_C4) { TestField(Field::C4); }
	TEST(Flip, Move_D4) { TestField(Field::D4); }
	TEST(Flip, Move_E4) { TestField(Field::E4); }
	TEST(Flip, Move_F4) { TestField(Field::F4); }
	TEST(Flip, Move_G4) { TestField(Field::G4); }
	TEST(Flip, Move_H4) { TestField(Field::H4); }
	TEST(Flip, Move_A5) { TestField(Field::A5); }
	TEST(Flip, Move_B5) { TestField(Field::B5); }
	TEST(Flip, Move_C5) { TestField(Field::C5); }
	TEST(Flip, Move_D5) { TestField(Field::D5); }
	TEST(Flip, Move_E5) { TestField(Field::E5); }
	TEST(Flip, Move_F5) { TestField(Field::F5); }
	TEST(Flip, Move_G5) { TestField(Field::G5); }
	TEST(Flip, Move_H5) { TestField(Field::H5); }
	TEST(Flip, Move_A6) { TestField(Field::A6); }
	TEST(Flip, Move_B6) { TestField(Field::B6); }
	TEST(Flip, Move_C6) { TestField(Field::C6); }
	TEST(Flip, Move_D6) { TestField(Field::D6); }
	TEST(Flip, Move_E6) { TestField(Field::E6); }
	TEST(Flip, Move_F6) { TestField(Field::F6); }
	TEST(Flip, Move_G6) { TestField(Field::G6); }
	TEST(Flip, Move_H6) { TestField(Field::H6); }
	TEST(Flip, Move_A7) { TestField(Field::A7); }
	TEST(Flip, Move_B7) { TestField(Field::B7); }
	TEST(Flip, Move_C7) { TestField(Field::C7); }
	TEST(Flip, Move_D7) { TestField(Field::D7); }
	TEST(Flip, Move_E7) { TestField(Field::E7); }
	TEST(Flip, Move_F7) { TestField(Field::F7); }
	TEST(Flip, Move_G7) { TestField(Field::G7); }
	TEST(Flip, Move_H7) { TestField(Field::H7); }
	TEST(Flip, Move_A8) { TestField(Field::A8); }
	TEST(Flip, Move_B8) { TestField(Field::B8); }
	TEST(Flip, Move_C8) { TestField(Field::C8); }
	TEST(Flip, Move_D8) { TestField(Field::D8); }
	TEST(Flip, Move_E8) { TestField(Field::E8); }
	TEST(Flip, Move_F8) { TestField(Field::F8); }
	TEST(Flip, Move_G8) { TestField(Field::G8); }
	TEST(Flip, Move_H8) { TestField(Field::H8); }
}
