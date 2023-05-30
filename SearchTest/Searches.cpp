#include "pch.h"

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
	"X X X X X X X -"
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

enum class Fail { hard, soft };

void Test(Algorithm& alg, PosScore pos_score, Fail fail = Fail::soft, OpenInterval w = { -inf_score, +inf_score })
{
	auto pos = pos_score.pos;
	auto correct = pos_score.score;

	int result = alg.Eval(pos, w);

	if (correct < w) // fail low
		if (fail == Fail::hard)
			ASSERT_EQ(result, w.Lower());
		else
			ASSERT_LE(result, w.Lower());
	else if (correct > w) // fail high
		if (fail == Fail::hard)
			ASSERT_EQ(result, w.Upper());
		else
			ASSERT_GE(result, w.Upper());
	else // score found
		ASSERT_EQ(result, correct);
}
void Test(Algorithm&& alg, PosScore pos_score, Fail fail = Fail::soft, OpenInterval w = { -inf_score, +inf_score })
{
	Test(alg, pos_score, fail, w);
}

void TestAllWindows(Algorithm& alg, PosScore pos_score, Fail fail = Fail::soft)
{
	for (int lower : std::views::iota(min_score, max_score + 1))
		for (int upper : std::views::iota(lower + 1, max_score + 1))
			Test(alg, pos_score, fail, { lower, upper });
}
void TestAllWindows(Algorithm&& alg, PosScore pos_score, Fail fail = Fail::soft)
{
	TestAllWindows(alg, pos_score, fail);
}

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

TEST(NegaMax, Zero_empty_0) { ::Test(NegaMax{}, Zero_empty_0); }
TEST(NegaMax, Zero_empty_1) { ::Test(NegaMax{}, Zero_empty_1); }
TEST(NegaMax, Zero_empty_2) { ::Test(NegaMax{}, Zero_empty_2); }
TEST(NegaMax, One_empty_0) { ::Test(NegaMax{}, One_empty_0); }
TEST(NegaMax, One_empty_1) { ::Test(NegaMax{}, One_empty_1); }
TEST(NegaMax, One_empty_2) { ::Test(NegaMax{}, One_empty_2); }
TEST(NegaMax, One_empty_3) { ::Test(NegaMax{}, One_empty_3); }
TEST(NegaMax, Two_empty_0) { ::Test(NegaMax{}, Two_empty_0); }
TEST(NegaMax, Two_empty_1) { ::Test(NegaMax{}, Two_empty_1); }
TEST(NegaMax, Two_empty_2) { ::Test(NegaMax{}, Two_empty_2); }
TEST(NegaMax, Two_empty_3) { ::Test(NegaMax{}, Two_empty_3); }
TEST(NegaMax, Three_empty_0) { ::Test(NegaMax{}, Three_empty_0); }
TEST(NegaMax, Three_empty_1) { ::Test(NegaMax{}, Three_empty_1); }
TEST(NegaMax, Three_empty_2) { ::Test(NegaMax{}, Three_empty_2); }
TEST(NegaMax, Three_empty_3) { ::Test(NegaMax{}, Three_empty_3); }
TEST(NegaMax, Four_empty_0) { ::Test(NegaMax{}, Four_empty_0); }
TEST(NegaMax, Four_empty_1) { ::Test(NegaMax{}, Four_empty_1); }
TEST(NegaMax, Four_empty_2) { ::Test(NegaMax{}, Four_empty_2); }
TEST(NegaMax, Four_empty_3) { ::Test(NegaMax{}, Four_empty_3); }
TEST(NegaMax, Five_empty) { ::Test(NegaMax{}, Five_empty); }

TEST(AlphaBetaFailHard, Zero_empty_0) { TestAllWindows(AlphaBetaFailHard{}, Zero_empty_0, Fail::hard); }
TEST(AlphaBetaFailHard, Zero_empty_1) { TestAllWindows(AlphaBetaFailHard{}, Zero_empty_1, Fail::hard); }
TEST(AlphaBetaFailHard, Zero_empty_2) { TestAllWindows(AlphaBetaFailHard{}, Zero_empty_2, Fail::hard); }
TEST(AlphaBetaFailHard, One_empty_0) { TestAllWindows(AlphaBetaFailHard{}, One_empty_0, Fail::hard); }
TEST(AlphaBetaFailHard, One_empty_1) { TestAllWindows(AlphaBetaFailHard{}, One_empty_1, Fail::hard); }
TEST(AlphaBetaFailHard, One_empty_2) { TestAllWindows(AlphaBetaFailHard{}, One_empty_2, Fail::hard); }
TEST(AlphaBetaFailHard, One_empty_3) { TestAllWindows(AlphaBetaFailHard{}, One_empty_3, Fail::hard); }
TEST(AlphaBetaFailHard, Two_empty_0) { TestAllWindows(AlphaBetaFailHard{}, Two_empty_0, Fail::hard); }
TEST(AlphaBetaFailHard, Two_empty_1) { TestAllWindows(AlphaBetaFailHard{}, Two_empty_1, Fail::hard); }
TEST(AlphaBetaFailHard, Two_empty_2) { TestAllWindows(AlphaBetaFailHard{}, Two_empty_2, Fail::hard); }
TEST(AlphaBetaFailHard, Two_empty_3) { TestAllWindows(AlphaBetaFailHard{}, Two_empty_3, Fail::hard); }
TEST(AlphaBetaFailHard, Three_empty_0) { TestAllWindows(AlphaBetaFailHard{}, Three_empty_0, Fail::hard); }
TEST(AlphaBetaFailHard, Three_empty_1) { TestAllWindows(AlphaBetaFailHard{}, Three_empty_1, Fail::hard); }
TEST(AlphaBetaFailHard, Three_empty_2) { TestAllWindows(AlphaBetaFailHard{}, Three_empty_2, Fail::hard); }
TEST(AlphaBetaFailHard, Three_empty_3) { TestAllWindows(AlphaBetaFailHard{}, Three_empty_3, Fail::hard); }
TEST(AlphaBetaFailHard, Four_empty_0) { TestAllWindows(AlphaBetaFailHard{}, Four_empty_0, Fail::hard); }
TEST(AlphaBetaFailHard, Four_empty_1) { TestAllWindows(AlphaBetaFailHard{}, Four_empty_1, Fail::hard); }
TEST(AlphaBetaFailHard, Four_empty_2) { TestAllWindows(AlphaBetaFailHard{}, Four_empty_2, Fail::hard); }
TEST(AlphaBetaFailHard, Four_empty_3) { TestAllWindows(AlphaBetaFailHard{}, Four_empty_3, Fail::hard); }
TEST(AlphaBetaFailHard, Five_empty) { TestAllWindows(AlphaBetaFailHard{}, Five_empty, Fail::hard); }
TEST(AlphaBetaFailHard, FForum_1) { ::Test(AlphaBetaFailHard{}, FForum[1]); }
TEST(AlphaBetaFailHard, FForum_2) { ::Test(AlphaBetaFailHard{}, FForum[2]); }
TEST(AlphaBetaFailHard, FForum_3) { ::Test(AlphaBetaFailHard{}, FForum[3]); }
TEST(AlphaBetaFailHard, FForum_4) { ::Test(AlphaBetaFailHard{}, FForum[4]); }
TEST(AlphaBetaFailHard, FForum_5) { ::Test(AlphaBetaFailHard{}, FForum[5]); }
TEST(AlphaBetaFailHard, FForum_6) { ::Test(AlphaBetaFailHard{}, FForum[6]); }
TEST(AlphaBetaFailHard, FForum_7) { ::Test(AlphaBetaFailHard{}, FForum[7]); }
TEST(AlphaBetaFailHard, FForum_8) { ::Test(AlphaBetaFailHard{}, FForum[8]); }
TEST(AlphaBetaFailHard, FForum_9) { ::Test(AlphaBetaFailHard{}, FForum[9]); }
TEST(AlphaBetaFailHard, FForum_10) { ::Test(AlphaBetaFailHard{}, FForum[10]); }

TEST(AlphaBetaFailSoft, Zero_empty_0) { TestAllWindows(AlphaBetaFailSoft{}, Zero_empty_0); }
TEST(AlphaBetaFailSoft, Zero_empty_1) { TestAllWindows(AlphaBetaFailSoft{}, Zero_empty_1); }
TEST(AlphaBetaFailSoft, Zero_empty_2) { TestAllWindows(AlphaBetaFailSoft{}, Zero_empty_2); }
TEST(AlphaBetaFailSoft, One_empty_0) { TestAllWindows(AlphaBetaFailSoft{}, One_empty_0); }
TEST(AlphaBetaFailSoft, One_empty_1) { TestAllWindows(AlphaBetaFailSoft{}, One_empty_1); }
TEST(AlphaBetaFailSoft, One_empty_2) { TestAllWindows(AlphaBetaFailSoft{}, One_empty_2); }
TEST(AlphaBetaFailSoft, One_empty_3) { TestAllWindows(AlphaBetaFailSoft{}, One_empty_3); }
TEST(AlphaBetaFailSoft, Two_empty_0) { TestAllWindows(AlphaBetaFailSoft{}, Two_empty_0); }
TEST(AlphaBetaFailSoft, Two_empty_1) { TestAllWindows(AlphaBetaFailSoft{}, Two_empty_1); }
TEST(AlphaBetaFailSoft, Two_empty_2) { TestAllWindows(AlphaBetaFailSoft{}, Two_empty_2); }
TEST(AlphaBetaFailSoft, Two_empty_3) { TestAllWindows(AlphaBetaFailSoft{}, Two_empty_3); }
TEST(AlphaBetaFailSoft, Three_empty_0) { TestAllWindows(AlphaBetaFailSoft{}, Three_empty_0); }
TEST(AlphaBetaFailSoft, Three_empty_1) { TestAllWindows(AlphaBetaFailSoft{}, Three_empty_1); }
TEST(AlphaBetaFailSoft, Three_empty_2) { TestAllWindows(AlphaBetaFailSoft{}, Three_empty_2); }
TEST(AlphaBetaFailSoft, Three_empty_3) { TestAllWindows(AlphaBetaFailSoft{}, Three_empty_3); }
TEST(AlphaBetaFailSoft, Four_empty_0) { TestAllWindows(AlphaBetaFailSoft{}, Four_empty_0); }
TEST(AlphaBetaFailSoft, Four_empty_1) { TestAllWindows(AlphaBetaFailSoft{}, Four_empty_1); }
TEST(AlphaBetaFailSoft, Four_empty_2) { TestAllWindows(AlphaBetaFailSoft{}, Four_empty_2); }
TEST(AlphaBetaFailSoft, Four_empty_3) { TestAllWindows(AlphaBetaFailSoft{}, Four_empty_3); }
TEST(AlphaBetaFailSoft, Five_empty) { TestAllWindows(AlphaBetaFailSoft{}, Five_empty); }
TEST(AlphaBetaFailSoft, FForum_1) { ::Test(AlphaBetaFailSoft{}, FForum[1]); }
TEST(AlphaBetaFailSoft, FForum_2) { ::Test(AlphaBetaFailSoft{}, FForum[2]); }
TEST(AlphaBetaFailSoft, FForum_3) { ::Test(AlphaBetaFailSoft{}, FForum[3]); }
TEST(AlphaBetaFailSoft, FForum_4) { ::Test(AlphaBetaFailSoft{}, FForum[4]); }
TEST(AlphaBetaFailSoft, FForum_5) { ::Test(AlphaBetaFailSoft{}, FForum[5]); }
TEST(AlphaBetaFailSoft, FForum_6) { ::Test(AlphaBetaFailSoft{}, FForum[6]); }
TEST(AlphaBetaFailSoft, FForum_7) { ::Test(AlphaBetaFailSoft{}, FForum[7]); }
TEST(AlphaBetaFailSoft, FForum_8) { ::Test(AlphaBetaFailSoft{}, FForum[8]); }
TEST(AlphaBetaFailSoft, FForum_9) { ::Test(AlphaBetaFailSoft{}, FForum[9]); }
TEST(AlphaBetaFailSoft, FForum_10) { ::Test(AlphaBetaFailSoft{}, FForum[10]); }

TEST(AlphaBetaFailSuperSoft, Zero_empty_0) { TestAllWindows(AlphaBetaFailSuperSoft{}, Zero_empty_0); }
TEST(AlphaBetaFailSuperSoft, Zero_empty_1) { TestAllWindows(AlphaBetaFailSuperSoft{}, Zero_empty_1); }
TEST(AlphaBetaFailSuperSoft, Zero_empty_2) { TestAllWindows(AlphaBetaFailSuperSoft{}, Zero_empty_2); }
TEST(AlphaBetaFailSuperSoft, One_empty_0) { TestAllWindows(AlphaBetaFailSuperSoft{}, One_empty_0); }
TEST(AlphaBetaFailSuperSoft, One_empty_1) { TestAllWindows(AlphaBetaFailSuperSoft{}, One_empty_1); }
TEST(AlphaBetaFailSuperSoft, One_empty_2) { TestAllWindows(AlphaBetaFailSuperSoft{}, One_empty_2); }
TEST(AlphaBetaFailSuperSoft, One_empty_3) { TestAllWindows(AlphaBetaFailSuperSoft{}, One_empty_3); }
TEST(AlphaBetaFailSuperSoft, Two_empty_0) { TestAllWindows(AlphaBetaFailSuperSoft{}, Two_empty_0); }
TEST(AlphaBetaFailSuperSoft, Two_empty_1) { TestAllWindows(AlphaBetaFailSuperSoft{}, Two_empty_1); }
TEST(AlphaBetaFailSuperSoft, Two_empty_2) { TestAllWindows(AlphaBetaFailSuperSoft{}, Two_empty_2); }
TEST(AlphaBetaFailSuperSoft, Two_empty_3) { TestAllWindows(AlphaBetaFailSuperSoft{}, Two_empty_3); }
TEST(AlphaBetaFailSuperSoft, Three_empty_0) { TestAllWindows(AlphaBetaFailSuperSoft{}, Three_empty_0); }
TEST(AlphaBetaFailSuperSoft, Three_empty_1) { TestAllWindows(AlphaBetaFailSuperSoft{}, Three_empty_1); }
TEST(AlphaBetaFailSuperSoft, Three_empty_2) { TestAllWindows(AlphaBetaFailSuperSoft{}, Three_empty_2); }
TEST(AlphaBetaFailSuperSoft, Three_empty_3) { TestAllWindows(AlphaBetaFailSuperSoft{}, Three_empty_3); }
TEST(AlphaBetaFailSuperSoft, Four_empty_0) { TestAllWindows(AlphaBetaFailSuperSoft{}, Four_empty_0); }
TEST(AlphaBetaFailSuperSoft, Four_empty_1) { TestAllWindows(AlphaBetaFailSuperSoft{}, Four_empty_1); }
TEST(AlphaBetaFailSuperSoft, Four_empty_2) { TestAllWindows(AlphaBetaFailSuperSoft{}, Four_empty_2); }
TEST(AlphaBetaFailSuperSoft, Four_empty_3) { TestAllWindows(AlphaBetaFailSuperSoft{}, Four_empty_3); }
TEST(AlphaBetaFailSuperSoft, Five_empty) { TestAllWindows(AlphaBetaFailSuperSoft{}, Five_empty); }
TEST(AlphaBetaFailSuperSoft, FForum_1) { ::Test(AlphaBetaFailSuperSoft{}, FForum[1]); }
TEST(AlphaBetaFailSuperSoft, FForum_2) { ::Test(AlphaBetaFailSuperSoft{}, FForum[2]); }
TEST(AlphaBetaFailSuperSoft, FForum_3) { ::Test(AlphaBetaFailSuperSoft{}, FForum[3]); }
TEST(AlphaBetaFailSuperSoft, FForum_4) { ::Test(AlphaBetaFailSuperSoft{}, FForum[4]); }
TEST(AlphaBetaFailSuperSoft, FForum_5) { ::Test(AlphaBetaFailSuperSoft{}, FForum[5]); }
TEST(AlphaBetaFailSuperSoft, FForum_6) { ::Test(AlphaBetaFailSuperSoft{}, FForum[6]); }
TEST(AlphaBetaFailSuperSoft, FForum_7) { ::Test(AlphaBetaFailSuperSoft{}, FForum[7]); }
TEST(AlphaBetaFailSuperSoft, FForum_8) { ::Test(AlphaBetaFailSuperSoft{}, FForum[8]); }
TEST(AlphaBetaFailSuperSoft, FForum_9) { ::Test(AlphaBetaFailSuperSoft{}, FForum[9]); }
TEST(AlphaBetaFailSuperSoft, FForum_10) { ::Test(AlphaBetaFailSuperSoft{}, FForum[10]); }

HT ht_0{ 1 };
PVS pvs{ ht_0, AAGLEM{} };

TEST(PVS, Zero_empty_0) { TestAllWindows(pvs, Zero_empty_0); }
TEST(PVS, Zero_empty_1) { TestAllWindows(pvs, Zero_empty_1); }
TEST(PVS, Zero_empty_2) { TestAllWindows(pvs, Zero_empty_2); }
TEST(PVS, One_empty_0) { TestAllWindows(pvs, One_empty_0); }
TEST(PVS, One_empty_1) { TestAllWindows(pvs, One_empty_1); }
TEST(PVS, One_empty_2) { TestAllWindows(pvs, One_empty_2); }
TEST(PVS, One_empty_3) { TestAllWindows(pvs, One_empty_3); }
TEST(PVS, Two_empty_0) { TestAllWindows(pvs, Two_empty_0); }
TEST(PVS, Two_empty_1) { TestAllWindows(pvs, Two_empty_1); }
TEST(PVS, Two_empty_2) { TestAllWindows(pvs, Two_empty_2); }
TEST(PVS, Two_empty_3) { TestAllWindows(pvs, Two_empty_3); }
TEST(PVS, Three_empty_0) { TestAllWindows(pvs, Three_empty_0); }
TEST(PVS, Three_empty_1) { TestAllWindows(pvs, Three_empty_1); }
TEST(PVS, Three_empty_2) { TestAllWindows(pvs, Three_empty_2); }
TEST(PVS, Three_empty_3) { TestAllWindows(pvs, Three_empty_3); }
TEST(PVS, Four_empty_0) { TestAllWindows(pvs, Four_empty_0); }
TEST(PVS, Four_empty_1) { TestAllWindows(pvs, Four_empty_1); }
TEST(PVS, Four_empty_2) { TestAllWindows(pvs, Four_empty_2); }
TEST(PVS, Four_empty_3) { TestAllWindows(pvs, Four_empty_3); }
TEST(PVS, Five_empty) { TestAllWindows(pvs, Five_empty); }
TEST(PVS, FForum_1) { ::Test(pvs, FForum[1]); }
TEST(PVS, FForum_2) { ::Test(pvs, FForum[2]); }
TEST(PVS, FForum_3) { ::Test(pvs, FForum[3]); }
TEST(PVS, FForum_4) { ::Test(pvs, FForum[4]); }
TEST(PVS, FForum_5) { ::Test(pvs, FForum[5]); }
TEST(PVS, FForum_6) { ::Test(pvs, FForum[6]); }
TEST(PVS, FForum_7) { ::Test(pvs, FForum[7]); }
TEST(PVS, FForum_8) { ::Test(pvs, FForum[8]); }
TEST(PVS, FForum_9) { ::Test(pvs, FForum[9]); }
TEST(PVS, FForum_10) { ::Test(pvs, FForum[10]); }

HT ht{ 1 };
PVS pvs_tt{ ht, AAGLEM{} };

TEST(PVS_TT, Zero_empty_0) { TestAllWindows(pvs_tt, Zero_empty_0); }
TEST(PVS_TT, Zero_empty_1) { TestAllWindows(pvs_tt, Zero_empty_1); }
TEST(PVS_TT, Zero_empty_2) { TestAllWindows(pvs_tt, Zero_empty_2); }
TEST(PVS_TT, One_empty_0) { TestAllWindows(pvs_tt, One_empty_0); }
TEST(PVS_TT, One_empty_1) { TestAllWindows(pvs_tt, One_empty_1); }
TEST(PVS_TT, One_empty_2) { TestAllWindows(pvs_tt, One_empty_2); }
TEST(PVS_TT, One_empty_3) { TestAllWindows(pvs_tt, One_empty_3); }
TEST(PVS_TT, Two_empty_0) { TestAllWindows(pvs_tt, Two_empty_0); }
TEST(PVS_TT, Two_empty_1) { TestAllWindows(pvs_tt, Two_empty_1); }
TEST(PVS_TT, Two_empty_2) { TestAllWindows(pvs_tt, Two_empty_2); }
TEST(PVS_TT, Two_empty_3) { TestAllWindows(pvs_tt, Two_empty_3); }
TEST(PVS_TT, Three_empty_0) { TestAllWindows(pvs_tt, Three_empty_0); }
TEST(PVS_TT, Three_empty_1) { TestAllWindows(pvs_tt, Three_empty_1); }
TEST(PVS_TT, Three_empty_2) { TestAllWindows(pvs_tt, Three_empty_2); }
TEST(PVS_TT, Three_empty_3) { TestAllWindows(pvs_tt, Three_empty_3); }
TEST(PVS_TT, Four_empty_0) { TestAllWindows(pvs_tt, Four_empty_0); }
TEST(PVS_TT, Four_empty_1) { TestAllWindows(pvs_tt, Four_empty_1); }
TEST(PVS_TT, Four_empty_2) { TestAllWindows(pvs_tt, Four_empty_2); }
TEST(PVS_TT, Four_empty_3) { TestAllWindows(pvs_tt, Four_empty_3); }
TEST(PVS_TT, Five_empty) { TestAllWindows(pvs_tt, Five_empty); }
TEST(PVS_TT, FForum_1) { ::Test(pvs_tt, FForum[1]); }
TEST(PVS_TT, FForum_2) { ::Test(pvs_tt, FForum[2]); }
TEST(PVS_TT, FForum_3) { ::Test(pvs_tt, FForum[3]); }
TEST(PVS_TT, FForum_4) { ::Test(pvs_tt, FForum[4]); }
TEST(PVS_TT, FForum_5) { ::Test(pvs_tt, FForum[5]); }
TEST(PVS_TT, FForum_6) { ::Test(pvs_tt, FForum[6]); }
TEST(PVS_TT, FForum_7) { ::Test(pvs_tt, FForum[7]); }
TEST(PVS_TT, FForum_8) { ::Test(pvs_tt, FForum[8]); }
TEST(PVS_TT, FForum_9) { ::Test(pvs_tt, FForum[9]); }
TEST(PVS_TT, FForum_10) { ::Test(pvs_tt, FForum[10]); }
