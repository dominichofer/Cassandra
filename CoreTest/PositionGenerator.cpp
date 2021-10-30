#include "pch.h"
#include "Core/Core.h"
#include <numeric>

namespace OEIS_Tests
{
	std::size_t SizeOfAll(int empty_count)
	{
		auto gen = Children(Position::Start(), empty_count);
		std::vector<Position> all(gen.begin(), gen.end());
		std::sort(all.begin(), all.end());
		auto it = std::unique(all.begin(), all.end());
		return std::distance(all.begin(), it);
	}

	TEST(Children, All_with_empty_count_59) { ASSERT_EQ(SizeOfAll(59), 4); }
	TEST(Children, All_with_empty_count_58) { ASSERT_EQ(SizeOfAll(58), 12); }
	TEST(Children, All_with_empty_count_57) { ASSERT_EQ(SizeOfAll(57), 54); }
	TEST(Children, All_with_empty_count_56) { ASSERT_EQ(SizeOfAll(56), 236); }
	TEST(Children, All_with_empty_count_55) { ASSERT_EQ(SizeOfAll(55), 1'288); }
	TEST(Children, All_with_empty_count_54) { ASSERT_EQ(SizeOfAll(54), 7'092); }
	TEST(Children, All_with_empty_count_53) { ASSERT_EQ(SizeOfAll(53), 42'614); }
	TEST(Children, All_with_empty_count_52) { ASSERT_EQ(SizeOfAll(52), 269'352); }

	// Number of different Othello positions at the end of the n-th ply. (https://oeis.org/A124005)
	std::size_t NumberOfDifferentPositions(int plies)
	{
		auto gen = Children(Position::Start(), plies, true);
		std::vector<Position> all(gen.begin(), gen.end());
		std::sort(all.begin(), all.end());
		return std::inner_product(all.begin() + 1, all.end(), all.begin(), 1, std::plus(), std::not_equal_to());
	}
	TEST(Children, different_positions_at_ply_1) { ASSERT_EQ(NumberOfDifferentPositions(1), 4); }
	TEST(Children, different_positions_at_ply_2) { ASSERT_EQ(NumberOfDifferentPositions(2), 12); }
	TEST(Children, different_positions_at_ply_3) { ASSERT_EQ(NumberOfDifferentPositions(3), 54); }
	TEST(Children, different_positions_at_ply_4) { ASSERT_EQ(NumberOfDifferentPositions(4), 236); }
	TEST(Children, different_positions_at_ply_5) { ASSERT_EQ(NumberOfDifferentPositions(5), 1288); }
	TEST(Children, different_positions_at_ply_6) { ASSERT_EQ(NumberOfDifferentPositions(6), 7092); }
	TEST(Children, different_positions_at_ply_7) { ASSERT_EQ(NumberOfDifferentPositions(7), 42614); }
	TEST(Children, different_positions_at_ply_8) { ASSERT_EQ(NumberOfDifferentPositions(8), 269352); }

	// Number of Othello positions with unique realization at the end of the n-th ply. (https://oeis.org/A124006)
	std::size_t NumberOfUniqueRealizations(int plies)
	{
		auto gen = Children(Position::Start(), plies, true);
		std::vector<Position> all(gen.begin(), gen.end());
		std::sort(all.begin(), all.end());

		// Counts Othello positions that occure once and only once in the list.
		const int64_t size = static_cast<int64_t>(all.size());
		if (size < 2)
			return size;

		int64_t sum = 0;
		if (all[0] != all[1])
			sum++;
		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 1; i < size - 1; i++)
			if ((all[i-1] != all[i]) && (all[i] != all[i+1]))
				sum++;
		if (all[size - 2] != all[size - 1])
			sum++;
		return sum;
	}

	TEST(Children, unique_positions_at_ply_1) { ASSERT_EQ(NumberOfUniqueRealizations(1), 4); }
	TEST(Children, unique_positions_at_ply_2) { ASSERT_EQ(NumberOfUniqueRealizations(2), 12); }
	TEST(Children, unique_positions_at_ply_3) { ASSERT_EQ(NumberOfUniqueRealizations(3), 52); }
	TEST(Children, unique_positions_at_ply_4) { ASSERT_EQ(NumberOfUniqueRealizations(4), 228); }
	TEST(Children, unique_positions_at_ply_5) { ASSERT_EQ(NumberOfUniqueRealizations(5), 1192); }
	TEST(Children, unique_positions_at_ply_6) { ASSERT_EQ(NumberOfUniqueRealizations(6), 6160); }
	TEST(Children, unique_positions_at_ply_7) { ASSERT_EQ(NumberOfUniqueRealizations(7), 33344); }
	TEST(Children, unique_positions_at_ply_8) { ASSERT_EQ(NumberOfUniqueRealizations(8), 191380); }

	// Number of possible Reversi games at the end of the n-th ply. (https://oeis.org/A124004)
	std::size_t Number_of_possible_games(int plies)
	{
		auto gen = Children(Position::Start(), plies, true);
		return std::distance(gen.begin(), gen.end());
	}

	TEST(Children, possible_games_at_ply_1) { ASSERT_EQ(Number_of_possible_games(1), 4); }
	TEST(Children, possible_games_at_ply_2) { ASSERT_EQ(Number_of_possible_games(2), 12); }
	TEST(Children, possible_games_at_ply_3) { ASSERT_EQ(Number_of_possible_games(3), 56); }
	TEST(Children, possible_games_at_ply_4) { ASSERT_EQ(Number_of_possible_games(4), 244); }
	TEST(Children, possible_games_at_ply_5) { ASSERT_EQ(Number_of_possible_games(5), 1'396); }
	TEST(Children, possible_games_at_ply_6) { ASSERT_EQ(Number_of_possible_games(6), 8'200); }
	TEST(Children, possible_games_at_ply_7) { ASSERT_EQ(Number_of_possible_games(7), 55'092); }
	TEST(Children, possible_games_at_ply_8) { ASSERT_EQ(Number_of_possible_games(8), 390'216); }
	TEST(Children, possible_games_at_ply_9) { ASSERT_EQ(Number_of_possible_games(9), 3'005'288); }
	TEST(Children, possible_games_at_ply_10) { ASSERT_EQ(Number_of_possible_games(10), 24'571'056); }
}
TEST(PosGen, Random_is_deterministic)
{
	int seed = 42;
	PosGen::Random rnd_1(seed);
	PosGen::Random rnd_2(seed);

	ASSERT_EQ(rnd_1(), rnd_2());
}

TEST(PosGen, RandomWithEmptyCount_is_deterministic)
{
	int seed = 42;
	int empty_count = 15;
	PosGen::RandomWithEmptyCount rnd_1(empty_count, seed);
	PosGen::RandomWithEmptyCount rnd_2(empty_count, seed);

	ASSERT_EQ(rnd_1(), rnd_2());
}

TEST(PosGen, RandomWithEmptyCount_returns_empty_count)
{
	for (int empty_count = 0; empty_count <= 60; empty_count++)
	{
		PosGen::RandomWithEmptyCount rnd(empty_count);
		ASSERT_EQ(rnd().EmptyCount(), empty_count);
	}
}

class MockPlayer : public Player
{
	Position Play(const Position& in) override
	{
		return Position{ in.Opponent(), in.Player() | GetLSB(in.Empties()) };
	}
};

TEST(PosGen, Played_returns_empty_count)
{
	MockPlayer player1, player2;

	for (int empty_count = 0; empty_count <= 60; empty_count++)
	{
		PosGen::Played gen(player1, player2, empty_count);
		ASSERT_EQ(gen().EmptyCount(), empty_count);
	}
}

TEST(Children, zero_plies_is_self)
{
	auto gen1 = Children(Position::Start(), 0, true);
	ASSERT_EQ(std::distance(gen1.begin(), gen1.end()), 1);
	ASSERT_EQ(*gen1.begin(), Position::Start());

	auto gen2 = Children(Position::Start(), 0, false);
	ASSERT_EQ(std::distance(gen2.begin(), gen2.end()), 1);
	ASSERT_EQ(*gen2.begin(), Position::Start());
}

TEST(Children, zero_plies_of_unplayable_is_self)
{
	Position unplayable{~BitBoard{}, BitBoard{}};

	auto gen1 = Children(unplayable, 0, true);
	ASSERT_EQ(std::distance(gen1.begin(), gen1.end()), 1);
	ASSERT_EQ(*gen1.begin(), unplayable);

	auto gen2 = Children(unplayable, 0, false);
	ASSERT_EQ(std::distance(gen2.begin(), gen2.end()), 1);
	ASSERT_EQ(*gen2.begin(), unplayable);
}


TEST(pass_is_a_ply, passable_position_has_a_child)
{
	auto pos = 
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - X X - - -"
		"O O O O O O O O"_pos;
	auto gen = Children(pos, 1, true);
	ASSERT_EQ(std::distance(gen.begin(), gen.end()), 1);
}
TEST(pass_is_no_ply, passable_position_has_children)
{
	auto pos = 
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - X X - - -"
		"O O O O O O O O"_pos;
	auto gen = Children(pos, 1, false);
	ASSERT_EQ(std::distance(gen.begin(), gen.end()), PossibleMoves(PlayPass(pos)).size());
}
TEST(pass_is_a_ply, end_position_has_no_children)
{
	auto pos = 
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pos;
	auto gen = Children(pos, 1, true);
	ASSERT_EQ(std::distance(gen.begin(), gen.end()), 0);
}
TEST(pass_is_no_ply, end_position_has_no_children)
{
	auto pos = 
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pos;
	auto gen = Children(pos, 1, false);
	ASSERT_EQ(std::distance(gen.begin(), gen.end()), 0);
}