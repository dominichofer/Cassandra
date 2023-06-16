#include "pch.h"
#include <ranges>
#include <numeric>

std::size_t NumberOfChildren(int empty_count)
{
	return std::ranges::distance(Children(Position::Start(), empty_count));
}

TEST(Children, NumberOfChildren_e59) { ASSERT_EQ(NumberOfChildren(59), 4); }
TEST(Children, NumberOfChildren_e58) { ASSERT_EQ(NumberOfChildren(58), 12); }
TEST(Children, NumberOfChildren_e57) { ASSERT_EQ(NumberOfChildren(57), 56); }
TEST(Children, NumberOfChildren_e56) { ASSERT_EQ(NumberOfChildren(56), 244); }
TEST(Children, NumberOfChildren_e55) { ASSERT_EQ(NumberOfChildren(55), 1'396); }
TEST(Children, NumberOfChildren_e54) { ASSERT_EQ(NumberOfChildren(54), 8'200); }
TEST(Children, NumberOfChildren_e53) { ASSERT_EQ(NumberOfChildren(53), 55'092); }
TEST(Children, NumberOfChildren_e52) { ASSERT_EQ(NumberOfChildren(52), 390'216); }
TEST(Children, NumberOfChildren_e51) { ASSERT_EQ(NumberOfChildren(51), 3'005'320); }

std::size_t NumberOfUniqueChildren(int empty_count)
{
	return std::size(UniqueChildren(Position::Start(), empty_count));
}

TEST(Children, unique_positions_at_ply_1) { ASSERT_EQ(NumberOfUniqueChildren(59), 1); }
TEST(Children, unique_positions_at_ply_2) { ASSERT_EQ(NumberOfUniqueChildren(58), 3); }
TEST(Children, unique_positions_at_ply_3) { ASSERT_EQ(NumberOfUniqueChildren(57), 14); }
TEST(Children, unique_positions_at_ply_4) { ASSERT_EQ(NumberOfUniqueChildren(56), 60); }
TEST(Children, unique_positions_at_ply_5) { ASSERT_EQ(NumberOfUniqueChildren(55), 322); }
TEST(Children, unique_positions_at_ply_6) { ASSERT_EQ(NumberOfUniqueChildren(54), 1'773); }
TEST(Children, unique_positions_at_ply_7) { ASSERT_EQ(NumberOfUniqueChildren(53), 10'649); }
TEST(Children, unique_positions_at_ply_8) { ASSERT_EQ(NumberOfUniqueChildren(52), 67'245); }

// Number of possible Reversi games at the end of the n-th ply. (https://oeis.org/A124004)
std::size_t NumberOfPossibleGames(int plies)
{
	return std::ranges::distance(Children(Position::Start(), plies, true));
}

TEST(Children, possible_games_at_ply_1) { ASSERT_EQ(NumberOfPossibleGames(1), 4); }
TEST(Children, possible_games_at_ply_2) { ASSERT_EQ(NumberOfPossibleGames(2), 12); }
TEST(Children, possible_games_at_ply_3) { ASSERT_EQ(NumberOfPossibleGames(3), 56); }
TEST(Children, possible_games_at_ply_4) { ASSERT_EQ(NumberOfPossibleGames(4), 244); }
TEST(Children, possible_games_at_ply_5) { ASSERT_EQ(NumberOfPossibleGames(5), 1'396); }
TEST(Children, possible_games_at_ply_6) { ASSERT_EQ(NumberOfPossibleGames(6), 8'200); }
TEST(Children, possible_games_at_ply_7) { ASSERT_EQ(NumberOfPossibleGames(7), 55'092); }
TEST(Children, possible_games_at_ply_8) { ASSERT_EQ(NumberOfPossibleGames(8), 390'216); }
TEST(Children, possible_games_at_ply_9) { ASSERT_EQ(NumberOfPossibleGames(9), 3'005'288); }


// Number of different Othello positions at the end of the n-th ply. (https://oeis.org/A124005)
std::size_t NumberOfDifferentPositions(int plies)
{
	std::vector<Position> all;
	for (Position child : Children(Position::Start(), plies, true))
		all.push_back(child);
	std::sort(all.begin(), all.end());
	return std::inner_product(all.begin() + 1, all.end(), all.begin(), 1, std::plus(), std::not_equal_to());
}
TEST(Children, different_positions_at_ply_1) { ASSERT_EQ(NumberOfDifferentPositions(1), 4); }
TEST(Children, different_positions_at_ply_2) { ASSERT_EQ(NumberOfDifferentPositions(2), 12); }
TEST(Children, different_positions_at_ply_3) { ASSERT_EQ(NumberOfDifferentPositions(3), 54); }
TEST(Children, different_positions_at_ply_4) { ASSERT_EQ(NumberOfDifferentPositions(4), 236); }
TEST(Children, different_positions_at_ply_5) { ASSERT_EQ(NumberOfDifferentPositions(5), 1'288); }
TEST(Children, different_positions_at_ply_6) { ASSERT_EQ(NumberOfDifferentPositions(6), 7'092); }
TEST(Children, different_positions_at_ply_7) { ASSERT_EQ(NumberOfDifferentPositions(7), 42'614); }
TEST(Children, different_positions_at_ply_8) { ASSERT_EQ(NumberOfDifferentPositions(8), 269'352); }

// Number of Othello positions with unique realization at the end of the n-th ply. (https://oeis.org/A124006)
std::size_t NumberOfUniqueRealizations(int plies)
{
	std::vector<Position> all;
	for (Position child : Children(Position::Start(), plies, true))
		all.push_back(child);
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
		if ((all[i - 1] != all[i]) && (all[i] != all[i + 1]))
			sum++;
	if (all[size - 2] != all[size - 1])
		sum++;
	return sum;
}

TEST(Children, unique_realizations_at_ply_1) { ASSERT_EQ(NumberOfUniqueRealizations(1), 4); }
TEST(Children, unique_realizations_at_ply_2) { ASSERT_EQ(NumberOfUniqueRealizations(2), 12); }
TEST(Children, unique_realizations_at_ply_3) { ASSERT_EQ(NumberOfUniqueRealizations(3), 52); }
TEST(Children, unique_realizations_at_ply_4) { ASSERT_EQ(NumberOfUniqueRealizations(4), 228); }
TEST(Children, unique_realizations_at_ply_5) { ASSERT_EQ(NumberOfUniqueRealizations(5), 1192); }
TEST(Children, unique_realizations_at_ply_6) { ASSERT_EQ(NumberOfUniqueRealizations(6), 6160); }
TEST(Children, unique_realizations_at_ply_7) { ASSERT_EQ(NumberOfUniqueRealizations(7), 33344); }
TEST(Children, unique_realizations_at_ply_8) { ASSERT_EQ(NumberOfUniqueRealizations(8), 191380); }


TEST(Children, zero_plies_is_self)
{
	auto gen1 = Children(Position::Start(), 0, true);
	ASSERT_EQ(std::ranges::distance(gen1), 1);
	ASSERT_EQ(*gen1.begin(), Position::Start());

	auto gen2 = Children(Position::Start(), 0, false);
	ASSERT_EQ(std::ranges::distance(gen2), 1);
	ASSERT_EQ(*gen2.begin(), Position::Start());
}

TEST(Children, zero_plies_of_unplayable_is_self)
{
	Position unplayable{ 0xFFFFFFFFFFFFFFFFULL, 0 };

	auto gen1 = Children(unplayable, 0, true);
	ASSERT_EQ(std::ranges::distance(gen1), 1);
	ASSERT_EQ(*gen1.begin(), unplayable);

	auto gen2 = Children(unplayable, 0, false);
	ASSERT_EQ(std::ranges::distance(gen2), 1);
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
	ASSERT_TRUE(std::ranges::distance(gen) == 1);
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
	ASSERT_EQ(std::ranges::distance(gen), PossibleMoves(PlayPass(pos)).size());
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
	ASSERT_EQ(std::ranges::distance(gen), 0);
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
	ASSERT_EQ(std::ranges::distance(gen), 0);
}