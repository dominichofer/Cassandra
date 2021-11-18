#include "pch.h"

struct PosScore
{
	Position pos;
	int8_t score;
};

const PosScore Zero_empty_0 = {
	"X X X X X X X X"
	"O O O O O O O O"
	"X X X X X X X X"
	"O O O O O O O O"
	"X X X X X X X X"
	"O O O O O O O O"
	"X X X X X X X X"
	"O O O O O O O O"_pos, 0 / 2  };

const PosScore Zero_empty_1 = {
	"X X X X X X X X"
	"O O O O O O O O"
	"X X X X X X X X"
	"O O O O O O O O"
	"X X X X X X X X"
	"O O O O O O O O"
	"X X X X X X X X"
	"X X X X X X X X"_pos, +16 / 2  };

const PosScore Zero_empty_2 = {
	"X X X X X X X X"
	"O O O O O O O O"
	"X X X X X X X X"
	"O O O O O O O O"
	"X X X X X X X X"
	"O O O O O O O O"
	"O O O O O O O O"
	"O O O O O O O O"_pos, -16 / 2  };

const PosScore One_empty_0 = {
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X -"_pos, +64 / 2  };

const PosScore One_empty_1 = {
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X O O"
	"X X X X X X O -"_pos, +64 / 2  };

const PosScore One_empty_2 = {
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"O X X X X X X -"_pos, +48 / 2  };

const PosScore One_empty_3 = {
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X O"
	"O X X X X X X -"_pos, +62 / 2  };

const PosScore Two_empty_0 = {
	"X X X X X X X -"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X -"_pos, +64 / 2  };

const PosScore Two_empty_1 = {
	"X X X X X X O -"
	"X X X X X X O O"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X O O"
	"X X X X X X O -"_pos, +64 / 2  };

const PosScore Two_empty_2 = {
	"X X X X X X X  "
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"O X X X X X X -"_pos, +22 / 2  };

const PosScore Two_empty_3 = {
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X O X X"
	"X X X X X X X O"
	"X X X X X X O -"
	"O X X X X X X -"_pos, +54 / 2  };

const PosScore Three_empty_0 = {
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X - - -"_pos, +64 / 2  };

const PosScore Three_empty_1 = {
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X O X"
	"X X X X X - - -"_pos, +64 / 2  };

const PosScore Three_empty_2 = {
	"X X X X X O O O"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X - - -"_pos, +16 / 2  };

const PosScore Three_empty_3 = {
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X O"
	"X X X X X O X X"
	"X X X X X X X -"
	"X X X X X X O -"
	"O X X X X X X -"_pos, +58 / 2  };

const PosScore Four_empty_0 = {
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X - - - -"_pos, +64 / 2  };

const PosScore Four_empty_1 = {
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X O O O O O"
	"X X X O - - - -"_pos, +64 / 2  };

const PosScore Four_empty_2 = {
	"X X X X O O O O"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X - - - -"_pos, +0 / 2  };

const PosScore Four_empty_3 = {
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X O O X X X"
	"X X O X X O X X"
	"X X - - - - X X"_pos, +56 / 2  };


const PosScore Five_empty = {
	"X X X X X X X -"
	"X X X X X X X O"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X X X X X X X X"
	"X O X O X O X O"
	"X - X - X - X -"_pos, +64 / 2  };


class NegaMaxTest : public ::testing::Test
{
public:
	void Test(PosScore pos_score)
	{
		const Position pos = pos_score.pos;
		const int correct = pos_score.score;

		const auto result = NegaMax{}.Eval(pos);
				
		ASSERT_EQ(result, correct);
	}
};

TEST_F(NegaMaxTest, Zero_empty_0) { Test(Zero_empty_0); }
TEST_F(NegaMaxTest, Zero_empty_1) { Test(Zero_empty_1); }
TEST_F(NegaMaxTest, Zero_empty_2) { Test(Zero_empty_2); }
TEST_F(NegaMaxTest, One_empty_0) { Test(One_empty_0); }
TEST_F(NegaMaxTest, One_empty_1) { Test(One_empty_1); }
TEST_F(NegaMaxTest, One_empty_2) { Test(One_empty_2); }
TEST_F(NegaMaxTest, One_empty_3) { Test(One_empty_3); }
TEST_F(NegaMaxTest, Two_empty_0) { Test(Two_empty_0); }
TEST_F(NegaMaxTest, Two_empty_1) { Test(Two_empty_1); }
TEST_F(NegaMaxTest, Two_empty_2) { Test(Two_empty_2); }
TEST_F(NegaMaxTest, Two_empty_3) { Test(Two_empty_3); }
TEST_F(NegaMaxTest, Three_empty_0) { Test(Three_empty_0); }
TEST_F(NegaMaxTest, Three_empty_1) { Test(Three_empty_1); }
TEST_F(NegaMaxTest, Three_empty_2) { Test(Three_empty_2); }
TEST_F(NegaMaxTest, Three_empty_3) { Test(Three_empty_3); }
TEST_F(NegaMaxTest, Four_empty_0) { Test(Four_empty_0); }
TEST_F(NegaMaxTest, Four_empty_1) { Test(Four_empty_1); }
TEST_F(NegaMaxTest, Four_empty_2) { Test(Four_empty_2); }
TEST_F(NegaMaxTest, Four_empty_3) { Test(Four_empty_3); }
TEST_F(NegaMaxTest, Five_empty) { Test(Five_empty); }


class AlphaBetaFailHardTest : public ::testing::Test
{
public:
	void Test(const Position& pos, int correct, OpenInterval w)
	{
		const int result = AlphaBetaFailHard{}.Eval(pos, w);

		if (correct < w) // fail low
			ASSERT_EQ(result, w.lower());
		else if (correct > w) // fail high
			ASSERT_EQ(result, w.upper());
		else // score found
			ASSERT_EQ(result, correct);
	}

	void Test(const Puzzle& puzzle)
	{
		Test(puzzle.pos, puzzle.tasks.back().Score(), OpenInterval::Whole());
	}

	void Test_all_windows(const Position& pos,  int correct)
	{
		for (int lower : std::views::iota(min_score, max_score + 1))
			for (int upper : std::views::iota(lower + 1, max_score + 1))
				Test(pos, correct, { lower, upper });
	}

	void Test_all_windows(const PosScore& pos)
	{
		Test_all_windows(pos.pos, pos.score);
	}
};

TEST_F(AlphaBetaFailHardTest, Zero_empty_0) { Test_all_windows(Zero_empty_0); }
TEST_F(AlphaBetaFailHardTest, Zero_empty_1) { Test_all_windows(Zero_empty_1); }
TEST_F(AlphaBetaFailHardTest, Zero_empty_2) { Test_all_windows(Zero_empty_2); }
TEST_F(AlphaBetaFailHardTest, One_empty_0) { Test_all_windows(One_empty_0); }
TEST_F(AlphaBetaFailHardTest, One_empty_1) { Test_all_windows(One_empty_1); }
TEST_F(AlphaBetaFailHardTest, One_empty_2) { Test_all_windows(One_empty_2); }
TEST_F(AlphaBetaFailHardTest, One_empty_3) { Test_all_windows(One_empty_3); }
TEST_F(AlphaBetaFailHardTest, Two_empty_0) { Test_all_windows(Two_empty_0); }
TEST_F(AlphaBetaFailHardTest, Two_empty_1) { Test_all_windows(Two_empty_1); }
TEST_F(AlphaBetaFailHardTest, Two_empty_2) { Test_all_windows(Two_empty_2); }
TEST_F(AlphaBetaFailHardTest, Two_empty_3) { Test_all_windows(Two_empty_3); }
TEST_F(AlphaBetaFailHardTest, Three_empty_0) { Test_all_windows(Three_empty_0); }
TEST_F(AlphaBetaFailHardTest, Three_empty_1) { Test_all_windows(Three_empty_1); }
TEST_F(AlphaBetaFailHardTest, Three_empty_2) { Test_all_windows(Three_empty_2); }
TEST_F(AlphaBetaFailHardTest, Three_empty_3) { Test_all_windows(Three_empty_3); }
TEST_F(AlphaBetaFailHardTest, Four_empty_0) { Test_all_windows(Four_empty_0); }
TEST_F(AlphaBetaFailHardTest, Four_empty_1) { Test_all_windows(Four_empty_1); }
TEST_F(AlphaBetaFailHardTest, Four_empty_2) { Test_all_windows(Four_empty_2); }
TEST_F(AlphaBetaFailHardTest, Four_empty_3) { Test_all_windows(Four_empty_3); }
TEST_F(AlphaBetaFailHardTest, Five_empty) { Test_all_windows(Five_empty); }
// TEST_F(AlphaBetaFailHardTest, FForum_1) { Test(FForum[1]); }
// TEST_F(AlphaBetaFailHardTest, FForum_2) { Test(FForum[2]); }
// TEST_F(AlphaBetaFailHardTest, FForum_3) { Test(FForum[3]); }
// TEST_F(AlphaBetaFailHardTest, FForum_4) { Test(FForum[4]); }
// TEST_F(AlphaBetaFailHardTest, FForum_5) { Test(FForum[5]); }
// TEST_F(AlphaBetaFailHardTest, FForum_6) { Test(FForum[6]); }
// TEST_F(AlphaBetaFailHardTest, FForum_7) { Test(FForum[7]); }
// TEST_F(AlphaBetaFailHardTest, FForum_8) { Test(FForum[8]); }
// TEST_F(AlphaBetaFailHardTest, FForum_9) { Test(FForum[9]); }
// TEST_F(AlphaBetaFailHardTest, FForum_10) { Test(FForum[10]); }


class AlphaBetaFailSoftTest : public ::testing::Test
{
public:
	void Test(const Position& pos, const int& correct, const OpenInterval& w)
	{
		const auto result = AlphaBetaFailSoft{}.Eval(pos, w);

		if (correct < w) // fail low
			ASSERT_LE(result, w.lower());
		else if (correct > w) // fail high
			ASSERT_GE(result, w.upper());
		else // score found
			ASSERT_EQ(result, correct);
	}

	void Test(const Puzzle& puzzle)
	{
		Test(puzzle.pos, puzzle.tasks.back().Score(), OpenInterval::Whole());
	}

	void Test_all_windows(const Position& pos, const int& correct)
	{
		for (int lower : std::views::iota(min_score, max_score + 1))
			for (int upper : std::views::iota(lower + 1, max_score + 1))
				Test(pos, correct, { lower, upper });
	}

	void Test_all_windows(const PosScore& pos)
	{
		Test_all_windows(pos.pos, pos.score);
	}
};

TEST_F(AlphaBetaFailSoftTest, Zero_empty_0) { Test_all_windows(Zero_empty_0); }
TEST_F(AlphaBetaFailSoftTest, Zero_empty_1) { Test_all_windows(Zero_empty_1); }
TEST_F(AlphaBetaFailSoftTest, Zero_empty_2) { Test_all_windows(Zero_empty_2); }
TEST_F(AlphaBetaFailSoftTest, One_empty_0) { Test_all_windows(One_empty_0); }
TEST_F(AlphaBetaFailSoftTest, One_empty_1) { Test_all_windows(One_empty_1); }
TEST_F(AlphaBetaFailSoftTest, One_empty_2) { Test_all_windows(One_empty_2); }
TEST_F(AlphaBetaFailSoftTest, One_empty_3) { Test_all_windows(One_empty_3); }
TEST_F(AlphaBetaFailSoftTest, Two_empty_0) { Test_all_windows(Two_empty_0); }
TEST_F(AlphaBetaFailSoftTest, Two_empty_1) { Test_all_windows(Two_empty_1); }
TEST_F(AlphaBetaFailSoftTest, Two_empty_2) { Test_all_windows(Two_empty_2); }
TEST_F(AlphaBetaFailSoftTest, Two_empty_3) { Test_all_windows(Two_empty_3); }
TEST_F(AlphaBetaFailSoftTest, Three_empty_0) { Test_all_windows(Three_empty_0); }
TEST_F(AlphaBetaFailSoftTest, Three_empty_1) { Test_all_windows(Three_empty_1); }
TEST_F(AlphaBetaFailSoftTest, Three_empty_2) { Test_all_windows(Three_empty_2); }
TEST_F(AlphaBetaFailSoftTest, Three_empty_3) { Test_all_windows(Three_empty_3); }
TEST_F(AlphaBetaFailSoftTest, Four_empty_0) { Test_all_windows(Four_empty_0); }
TEST_F(AlphaBetaFailSoftTest, Four_empty_1) { Test_all_windows(Four_empty_1); }
TEST_F(AlphaBetaFailSoftTest, Four_empty_2) { Test_all_windows(Four_empty_2); }
TEST_F(AlphaBetaFailSoftTest, Four_empty_3) { Test_all_windows(Four_empty_3); }
TEST_F(AlphaBetaFailSoftTest, Five_empty) { Test_all_windows(Five_empty); }
// TEST_F(AlphaBetaFailSoftTest, FForum_1) { Test(FForum[1]); }
// TEST_F(AlphaBetaFailSoftTest, FForum_2) { Test(FForum[2]); }
// TEST_F(AlphaBetaFailSoftTest, FForum_3) { Test(FForum[3]); }
// TEST_F(AlphaBetaFailSoftTest, FForum_4) { Test(FForum[4]); }
// TEST_F(AlphaBetaFailSoftTest, FForum_5) { Test(FForum[5]); }
// TEST_F(AlphaBetaFailSoftTest, FForum_6) { Test(FForum[6]); }
// TEST_F(AlphaBetaFailSoftTest, FForum_7) { Test(FForum[7]); }
// TEST_F(AlphaBetaFailSoftTest, FForum_8) { Test(FForum[8]); }
// TEST_F(AlphaBetaFailSoftTest, FForum_9) { Test(FForum[9]); }
// TEST_F(AlphaBetaFailSoftTest, FForum_10) { Test(FForum[10]); }

class PVSTest : public ::testing::Test
{
public:
	HashTablePVS tt{ 1 };
	AAGLEM pe;

	void Test(const Position& pos, const int& correct, const OpenInterval& w)
	{
		int result = PVS{ tt, pe }.Eval(pos, w);

		if (correct < w) // fail low
			ASSERT_LE(result, w.lower());
		else if (correct > w) // fail high
			ASSERT_GE(result, w.upper());
		else // score found
			ASSERT_EQ(result, correct);
	}

	void Test(const Puzzle& puzzle)
	{
		Test(puzzle.pos, puzzle.tasks.back().Score(), OpenInterval::Whole());
	}

	void Test_all_windows(const Position& pos, const int& correct)
	{
		for (int lower : std::views::iota(min_score, max_score + 1))
			for (int upper : std::views::iota(lower + 1, max_score + 1))
				Test(pos, correct, { lower, upper });
	}

	void Test_all_windows(const PosScore& pos)
	{
		Test_all_windows(pos.pos, pos.score);
	}
};

TEST_F(PVSTest, Zero_empty_0) { Test_all_windows(Zero_empty_0); }
TEST_F(PVSTest, Zero_empty_1) { Test_all_windows(Zero_empty_1); }
TEST_F(PVSTest, Zero_empty_2) { Test_all_windows(Zero_empty_2); }
TEST_F(PVSTest, One_empty_0) { Test_all_windows(One_empty_0); }
TEST_F(PVSTest, One_empty_1) { Test_all_windows(One_empty_1); }
TEST_F(PVSTest, One_empty_2) { Test_all_windows(One_empty_2); }
TEST_F(PVSTest, One_empty_3) { Test_all_windows(One_empty_3); }
TEST_F(PVSTest, Two_empty_0) { Test_all_windows(Two_empty_0); }
TEST_F(PVSTest, Two_empty_1) { Test_all_windows(Two_empty_1); }
TEST_F(PVSTest, Two_empty_2) { Test_all_windows(Two_empty_2); }
TEST_F(PVSTest, Two_empty_3) { Test_all_windows(Two_empty_3); }
TEST_F(PVSTest, Three_empty_0) { Test_all_windows(Three_empty_0); }
TEST_F(PVSTest, Three_empty_1) { Test_all_windows(Three_empty_1); }
TEST_F(PVSTest, Three_empty_2) { Test_all_windows(Three_empty_2); }
TEST_F(PVSTest, Three_empty_3) { Test_all_windows(Three_empty_3); }
TEST_F(PVSTest, Four_empty_0) { Test_all_windows(Four_empty_0); }
TEST_F(PVSTest, Four_empty_1) { Test_all_windows(Four_empty_1); }
TEST_F(PVSTest, Four_empty_2) { Test_all_windows(Four_empty_2); }
TEST_F(PVSTest, Four_empty_3) { Test_all_windows(Four_empty_3); }
TEST_F(PVSTest, Five_empty) { Test_all_windows(Five_empty); }
// TEST_F(PVSTest, FForum_1) { Test(FForum[1]); }
// TEST_F(PVSTest, FForum_2) { Test(FForum[2]); }
// TEST_F(PVSTest, FForum_3) { Test(FForum[3]); }
// TEST_F(PVSTest, FForum_4) { Test(FForum[4]); }
// TEST_F(PVSTest, FForum_5) { Test(FForum[5]); }
// TEST_F(PVSTest, FForum_6) { Test(FForum[6]); }
// TEST_F(PVSTest, FForum_7) { Test(FForum[7]); }
// TEST_F(PVSTest, FForum_8) { Test(FForum[8]); }
// TEST_F(PVSTest, FForum_9) { Test(FForum[9]); }
// TEST_F(PVSTest, FForum_10) { Test(FForum[10]); }

class PVS_TT : public ::testing::Test
{
public:
	HashTablePVS tt{ 1'000 };
	AAGLEM pe;
	
	void Test(const Position& pos, const int& correct, const OpenInterval& w)
	{
		int result = PVS{ tt, pe }.Eval(pos, w);

		if (correct < w) // fail low
			ASSERT_LE(result, w.lower());
		else if (correct > w) // fail high
			ASSERT_GE(result, w.upper());
		else // score found
			ASSERT_EQ(result, correct);
	}

	void Test(const Puzzle& puzzle)
	{
		Test(puzzle.pos, puzzle.tasks.back().Score(), OpenInterval::Whole());
	}

	void Test_all_windows(const Position& pos, const int& correct)
	{
		for (int lower : std::views::iota(min_score, max_score + 1))
			for (int upper : std::views::iota(lower + 1, max_score + 1))
				Test(pos, correct, { lower, upper });
	}

	void Test_all_windows(const PosScore& pos)
	{
		Test_all_windows(pos.pos, pos.score);
	}
};

// TEST_F(PVS_TT, FForum_1) { Test(FForum[1]); }
// TEST_F(PVS_TT, FForum_2) { Test(FForum[2]); }
// TEST_F(PVS_TT, FForum_3) { Test(FForum[3]); }
// TEST_F(PVS_TT, FForum_4) { Test(FForum[4]); }
// TEST_F(PVS_TT, FForum_5) { Test(FForum[5]); }
// TEST_F(PVS_TT, FForum_6) { Test(FForum[6]); }
// TEST_F(PVS_TT, FForum_7) { Test(FForum[7]); }
// TEST_F(PVS_TT, FForum_8) { Test(FForum[8]); }
// TEST_F(PVS_TT, FForum_9) { Test(FForum[9]); }
// TEST_F(PVS_TT, FForum_10) { Test(FForum[10]); }
