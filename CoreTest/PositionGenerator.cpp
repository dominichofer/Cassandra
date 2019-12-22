#include "pch.h"

TEST(PosGen, Random_is_deterministic)
{
	PositionGenerator pg_1(42);
	PositionGenerator pg_2(42);

	ASSERT_EQ(pg_1.Random(), pg_2.Random());
}

TEST(PosGen, Random_with_empty_count_is_deterministic)
{
	PositionGenerator pg_1(42);
	PositionGenerator pg_2(42);

	ASSERT_EQ(pg_1.Random(8), pg_2.Random(8));
}

TEST(PosGen, Random_with_empty_count_returns_empty_count)
{
	PositionGenerator pg(42);

	for (std::size_t empty_count = 0; empty_count < 60; empty_count++)
		ASSERT_EQ(pg.Random(empty_count).EmptyCount(), empty_count);
}

class MockPlayer : public Player
{
	Position Play(Position in) noexcept(false) final
	{
		auto P = in.GetP();
		P[in.Empties().FirstField()] = true;
		return { in.GetO(), P };
	}
};
TEST(PosGen, Played_returns_empty_count)
{
	PositionGenerator pg(42);
	MockPlayer mock;

	for (std::size_t empty_count = 0; empty_count <= 60; empty_count++)
		ASSERT_EQ(pg.Played(mock, empty_count).EmptyCount(), empty_count);
}

std::size_t SizeOfAll(std::size_t empty_count)
{
	std::vector<Position> all;
	PositionGenerator::All(std::back_inserter(all), empty_count);
	std::sort(all.begin(), all.end(), 
		[](Position l, Position r) { return (l.GetP() == r.GetP()) ? (l.GetO() < r.GetO()) : (l.GetP() < r.GetP()); });
	auto it = std::unique(all.begin(), all.end());
	return std::distance(all.begin(), it);
}

TEST(PosGen, All_with_empty_count_60) { ASSERT_EQ(SizeOfAll(60), 1); }
TEST(PosGen, All_with_empty_count_59) { ASSERT_EQ(SizeOfAll(59), 4); }
TEST(PosGen, All_with_empty_count_58) { ASSERT_EQ(SizeOfAll(58), 12); }
TEST(PosGen, All_with_empty_count_57) { ASSERT_EQ(SizeOfAll(57), 54); }
TEST(PosGen, All_with_empty_count_56) { ASSERT_EQ(SizeOfAll(56), 236); }
TEST(PosGen, All_with_empty_count_55) { ASSERT_EQ(SizeOfAll(55), 1'288); }
TEST(PosGen, All_with_empty_count_54) { ASSERT_EQ(SizeOfAll(54), 7'092); }
TEST(PosGen, All_with_empty_count_53) { ASSERT_EQ(SizeOfAll(53), 42'614); }
TEST(PosGen, All_with_empty_count_52) { ASSERT_EQ(SizeOfAll(52), 269'352); }

// Number of different Othello positions at the end of the n-th ply. (https://oeis.org/A124005)
std::size_t Number_of_different_positions(std::size_t plies)
{
	std::vector<Position> all;
	PositionGenerator::All(std::back_inserter(all), plies, 1);
	std::sort(all.begin(), all.end(),
		[](Position l, Position r) { return (l.GetP() == r.GetP()) ? (l.GetO() < r.GetO()) : (l.GetP() < r.GetP()); });
	auto it = std::unique(all.begin(), all.end());
	return std::distance(all.begin(), it);
}

TEST(PosGen, different_positions_ply_0) { ASSERT_EQ(Number_of_different_positions(0), 1); }
TEST(PosGen, different_positions_ply_1) { ASSERT_EQ(Number_of_different_positions(1), 4); }
TEST(PosGen, different_positions_ply_2) { ASSERT_EQ(Number_of_different_positions(2), 12); }
TEST(PosGen, different_positions_ply_3) { ASSERT_EQ(Number_of_different_positions(3), 54); }
TEST(PosGen, different_positions_ply_4) { ASSERT_EQ(Number_of_different_positions(4), 236); }
TEST(PosGen, different_positions_ply_5) { ASSERT_EQ(Number_of_different_positions(5), 1288); }
TEST(PosGen, different_positions_ply_6) { ASSERT_EQ(Number_of_different_positions(6), 7092); }
TEST(PosGen, different_positions_ply_7) { ASSERT_EQ(Number_of_different_positions(7), 42614); }
TEST(PosGen, different_positions_ply_8) { ASSERT_EQ(Number_of_different_positions(8), 269352); }

// Number of Othello positions with unique realization at the end of the n-th ply. (https://oeis.org/A124006)
std::size_t Number_of_unique_realization(std::size_t plies)
{
	std::vector<Position> all;
	PositionGenerator::All(std::back_inserter(all), plies, 1);
	std::sort(all.begin(), all.end(),
		[](Position l, Position r) { return (l.GetP() == r.GetP()) ? (l.GetO() < r.GetO()) : (l.GetP() < r.GetP()); });

	std::size_t sum = 0;
	Position testee = all.front();
	bool unique = true;
	for (auto it = all.cbegin() + 1; it != all.cend(); ++it)
	{
		Position pos = *it;
		if (testee == pos)
			unique = false;
		else
		{
			if (unique)
				sum++;
			testee = pos;
			unique = true;
		}
	}
	if (unique)
		sum++;
	return sum;
}

TEST(PosGen, unique_positions_ply_0) { ASSERT_EQ(Number_of_unique_realization(0), 1); }
TEST(PosGen, unique_positions_ply_1) { ASSERT_EQ(Number_of_unique_realization(1), 4); }
TEST(PosGen, unique_positions_ply_2) { ASSERT_EQ(Number_of_unique_realization(2), 12); }
TEST(PosGen, unique_positions_ply_3) { ASSERT_EQ(Number_of_unique_realization(3), 52); }
TEST(PosGen, unique_positions_ply_4) { ASSERT_EQ(Number_of_unique_realization(4), 228); }
TEST(PosGen, unique_positions_ply_5) { ASSERT_EQ(Number_of_unique_realization(5), 1192); }
TEST(PosGen, unique_positions_ply_6) { ASSERT_EQ(Number_of_unique_realization(6), 6160); }
TEST(PosGen, unique_positions_ply_7) { ASSERT_EQ(Number_of_unique_realization(7), 33344); }
TEST(PosGen, unique_positions_ply_8) { ASSERT_EQ(Number_of_unique_realization(8), 191380); }

// Number of possible Reversi games at the end of the n-th ply. (https://oeis.org/A124004)
std::size_t Number_of_possible_games(std::size_t plies)
{
	std::vector<Position> all;
	PositionGenerator::All(std::back_inserter(all), plies, 1);
	return all.size();
}

TEST(PosGen, possible_games_ply_0) { ASSERT_EQ(Number_of_possible_games(0), 1); }
TEST(PosGen, possible_games_ply_1) { ASSERT_EQ(Number_of_possible_games(1), 4); }
TEST(PosGen, possible_games_ply_2) { ASSERT_EQ(Number_of_possible_games(2), 12); }
TEST(PosGen, possible_games_ply_3) { ASSERT_EQ(Number_of_possible_games(3), 56); }
TEST(PosGen, possible_games_ply_4) { ASSERT_EQ(Number_of_possible_games(4), 244); }
TEST(PosGen, possible_games_ply_5) { ASSERT_EQ(Number_of_possible_games(5), 1396); }
TEST(PosGen, possible_games_ply_6) { ASSERT_EQ(Number_of_possible_games(6), 8200); }
TEST(PosGen, possible_games_ply_7) { ASSERT_EQ(Number_of_possible_games(7), 55092); }
TEST(PosGen, possible_games_ply_8) { ASSERT_EQ(Number_of_possible_games(8), 390216); }
