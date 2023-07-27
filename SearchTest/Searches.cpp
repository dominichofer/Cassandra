#include "pch.h"

//enum class Fail { hard, soft };
//
//void Test(Context& alg, PosScore pos_score, Fail fail = Fail::soft, OpenInterval w = { -inf_score, +inf_score })
//{
//	auto pos = pos_score.pos;
//	auto correct = pos_score.score;
//
//	int result = alg.Eval(pos, w);
//
//	if (correct < w) // fail low
//		if (fail == Fail::hard)
//			ASSERT_EQ(result, w.Lower());
//		else
//			ASSERT_LE(result, w.Lower());
//	else if (correct > w) // fail high
//		if (fail == Fail::hard)
//			ASSERT_EQ(result, w.Upper());
//		else
//			ASSERT_GE(result, w.Upper());
//	else // score found
//		ASSERT_EQ(result, correct);
//}
//void Test(Context&& alg, PosScore pos_score, Fail fail = Fail::soft, OpenInterval w = { -inf_score, +inf_score })
//{
//	Test(alg, pos_score, fail, w);
//}
//
//void TestAllWindows(Algorithm& alg, PosScore pos_score, Fail fail = Fail::soft)
//{
//	for (int lower : std::views::iota(min_score, max_score + 1))
//		for (int upper : std::views::iota(lower + 1, max_score + 1))
//			Test(alg, pos_score, fail, { lower, upper });
//}
//void TestAllWindows(Algorithm&& alg, PosScore pos_score, Fail fail = Fail::soft)
//{
//	TestAllWindows(alg, pos_score, fail);
//}
//

TEST(NegaMax, Endgame)
{
	std::vector<PosScore> data = LoadPosScoreFile("..\\data\\endgame.ps");
	for (const PosScore& ps : data)
	{
		int score = NegaMax{}.Eval(ps.pos).score;
		EXPECT_EQ(score, ps.score);
	}
}

TEST(AlphaBeta, Endgame)
{
	std::vector<PosScore> data = LoadPosScoreFile("..\\data\\endgame.ps");
	for (const PosScore& ps : data)
	{
		int score = AlphaBeta{}.Eval(ps.pos).score;
		EXPECT_EQ(score, ps.score);
	}
}

TEST(PVS, Endgame)
{
	std::vector<PosScore> data = LoadPosScoreFile("..\\data\\endgame.ps");
	for (const PosScore& ps : data)
	{
		HT tt{ 1'000'000 };
		Result result = PVS{ tt, EstimatorStub{} }.Eval(ps.pos);
		EXPECT_EQ(result.score, ps.score);
	}
}

TEST(PVS, FForum1)
{
	std::vector<PosScore> data = LoadPosScoreFile("..\\data\\fforum-1-19.ps");
	for (const PosScore& ps : data)
	{
		HT tt{ 1'000'000 };
		Result result = PVS{ tt, EstimatorStub{} }.Eval(ps.pos);
		EXPECT_EQ(result.score, ps.score);
	}
}

//TEST(NegaMax, Zero_empty_0) { ::Test(NegaMax{}, Zero_empty_0); }
//TEST(NegaMax, Zero_empty_1) { ::Test(NegaMax{}, Zero_empty_1); }
//TEST(NegaMax, Zero_empty_2) { ::Test(NegaMax{}, Zero_empty_2); }
//TEST(NegaMax, One_empty_0) { ::Test(NegaMax{}, One_empty_0); }
//TEST(NegaMax, One_empty_1) { ::Test(NegaMax{}, One_empty_1); }
//TEST(NegaMax, One_empty_2) { ::Test(NegaMax{}, One_empty_2); }
//TEST(NegaMax, One_empty_3) { ::Test(NegaMax{}, One_empty_3); }
//TEST(NegaMax, Two_empty_0) { ::Test(NegaMax{}, Two_empty_0); }
//TEST(NegaMax, Two_empty_1) { ::Test(NegaMax{}, Two_empty_1); }
//TEST(NegaMax, Two_empty_2) { ::Test(NegaMax{}, Two_empty_2); }
//TEST(NegaMax, Two_empty_3) { ::Test(NegaMax{}, Two_empty_3); }
//TEST(NegaMax, Three_empty_0) { ::Test(NegaMax{}, Three_empty_0); }
//TEST(NegaMax, Three_empty_1) { ::Test(NegaMax{}, Three_empty_1); }
//TEST(NegaMax, Three_empty_2) { ::Test(NegaMax{}, Three_empty_2); }
//TEST(NegaMax, Three_empty_3) { ::Test(NegaMax{}, Three_empty_3); }
//TEST(NegaMax, Four_empty_0) { ::Test(NegaMax{}, Four_empty_0); }
//TEST(NegaMax, Four_empty_1) { ::Test(NegaMax{}, Four_empty_1); }
//TEST(NegaMax, Four_empty_2) { ::Test(NegaMax{}, Four_empty_2); }
//TEST(NegaMax, Four_empty_3) { ::Test(NegaMax{}, Four_empty_3); }
//TEST(NegaMax, Five_empty) { ::Test(NegaMax{}, Five_empty); }

//TEST(AlphaBeta, Zero_empty_0) { TestAllWindows(AlphaBeta{}, Zero_empty_0, Fail::hard); }
//TEST(AlphaBeta, Zero_empty_1) { TestAllWindows(AlphaBeta{}, Zero_empty_1, Fail::hard); }
//TEST(AlphaBeta, Zero_empty_2) { TestAllWindows(AlphaBeta{}, Zero_empty_2, Fail::hard); }
//TEST(AlphaBeta, One_empty_0) { TestAllWindows(AlphaBeta{}, One_empty_0, Fail::hard); }
//TEST(AlphaBeta, One_empty_1) { TestAllWindows(AlphaBeta{}, One_empty_1, Fail::hard); }
//TEST(AlphaBeta, One_empty_2) { TestAllWindows(AlphaBeta{}, One_empty_2, Fail::hard); }
//TEST(AlphaBeta, One_empty_3) { TestAllWindows(AlphaBeta{}, One_empty_3, Fail::hard); }
//TEST(AlphaBeta, Two_empty_0) { TestAllWindows(AlphaBeta{}, Two_empty_0, Fail::hard); }
//TEST(AlphaBeta, Two_empty_1) { TestAllWindows(AlphaBeta{}, Two_empty_1, Fail::hard); }
//TEST(AlphaBeta, Two_empty_2) { TestAllWindows(AlphaBeta{}, Two_empty_2, Fail::hard); }
//TEST(AlphaBeta, Two_empty_3) { TestAllWindows(AlphaBeta{}, Two_empty_3, Fail::hard); }
//TEST(AlphaBeta, Three_empty_0) { TestAllWindows(AlphaBeta{}, Three_empty_0, Fail::hard); }
//TEST(AlphaBeta, Three_empty_1) { TestAllWindows(AlphaBeta{}, Three_empty_1, Fail::hard); }
//TEST(AlphaBeta, Three_empty_2) { TestAllWindows(AlphaBeta{}, Three_empty_2, Fail::hard); }
//TEST(AlphaBeta, Three_empty_3) { TestAllWindows(AlphaBeta{}, Three_empty_3, Fail::hard); }
//TEST(AlphaBeta, Four_empty_0) { TestAllWindows(AlphaBeta{}, Four_empty_0, Fail::hard); }
//TEST(AlphaBeta, Four_empty_1) { TestAllWindows(AlphaBeta{}, Four_empty_1, Fail::hard); }
//TEST(AlphaBeta, Four_empty_2) { TestAllWindows(AlphaBeta{}, Four_empty_2, Fail::hard); }
//TEST(AlphaBeta, Four_empty_3) { TestAllWindows(AlphaBeta{}, Four_empty_3, Fail::hard); }
//TEST(AlphaBeta, Five_empty) { TestAllWindows(AlphaBeta{}, Five_empty, Fail::hard); }
//TEST(AlphaBeta, FForum_1) { ::Test(AlphaBeta{}, FForum[1]); }
//TEST(AlphaBeta, FForum_2) { ::Test(AlphaBeta{}, FForum[2]); }
//TEST(AlphaBeta, FForum_3) { ::Test(AlphaBeta{}, FForum[3]); }
//TEST(AlphaBeta, FForum_4) { ::Test(AlphaBeta{}, FForum[4]); }
//TEST(AlphaBeta, FForum_5) { ::Test(AlphaBeta{}, FForum[5]); }
//TEST(AlphaBeta, FForum_6) { ::Test(AlphaBeta{}, FForum[6]); }
//TEST(AlphaBeta, FForum_7) { ::Test(AlphaBeta{}, FForum[7]); }
//TEST(AlphaBeta, FForum_8) { ::Test(AlphaBeta{}, FForum[8]); }
//TEST(AlphaBeta, FForum_9) { ::Test(AlphaBeta{}, FForum[9]); }
//TEST(AlphaBeta, FForum_10) { ::Test(AlphaBeta{}, FForum[10]); }
//
//HT ht_0{ 1 };
//PVS pvs{ ht_0, AAGLEM{} };
//
//TEST(PVS, Zero_empty_0) { TestAllWindows(pvs, Zero_empty_0); }
//TEST(PVS, Zero_empty_1) { TestAllWindows(pvs, Zero_empty_1); }
//TEST(PVS, Zero_empty_2) { TestAllWindows(pvs, Zero_empty_2); }
//TEST(PVS, One_empty_0) { TestAllWindows(pvs, One_empty_0); }
//TEST(PVS, One_empty_1) { TestAllWindows(pvs, One_empty_1); }
//TEST(PVS, One_empty_2) { TestAllWindows(pvs, One_empty_2); }
//TEST(PVS, One_empty_3) { TestAllWindows(pvs, One_empty_3); }
//TEST(PVS, Two_empty_0) { TestAllWindows(pvs, Two_empty_0); }
//TEST(PVS, Two_empty_1) { TestAllWindows(pvs, Two_empty_1); }
//TEST(PVS, Two_empty_2) { TestAllWindows(pvs, Two_empty_2); }
//TEST(PVS, Two_empty_3) { TestAllWindows(pvs, Two_empty_3); }
//TEST(PVS, Three_empty_0) { TestAllWindows(pvs, Three_empty_0); }
//TEST(PVS, Three_empty_1) { TestAllWindows(pvs, Three_empty_1); }
//TEST(PVS, Three_empty_2) { TestAllWindows(pvs, Three_empty_2); }
//TEST(PVS, Three_empty_3) { TestAllWindows(pvs, Three_empty_3); }
//TEST(PVS, Four_empty_0) { TestAllWindows(pvs, Four_empty_0); }
//TEST(PVS, Four_empty_1) { TestAllWindows(pvs, Four_empty_1); }
//TEST(PVS, Four_empty_2) { TestAllWindows(pvs, Four_empty_2); }
//TEST(PVS, Four_empty_3) { TestAllWindows(pvs, Four_empty_3); }
//TEST(PVS, Five_empty) { TestAllWindows(pvs, Five_empty); }
//TEST(PVS, FForum_1) { ::Test(pvs, FForum[1]); }
//TEST(PVS, FForum_2) { ::Test(pvs, FForum[2]); }
//TEST(PVS, FForum_3) { ::Test(pvs, FForum[3]); }
//TEST(PVS, FForum_4) { ::Test(pvs, FForum[4]); }
//TEST(PVS, FForum_5) { ::Test(pvs, FForum[5]); }
//TEST(PVS, FForum_6) { ::Test(pvs, FForum[6]); }
//TEST(PVS, FForum_7) { ::Test(pvs, FForum[7]); }
//TEST(PVS, FForum_8) { ::Test(pvs, FForum[8]); }
//TEST(PVS, FForum_9) { ::Test(pvs, FForum[9]); }
//TEST(PVS, FForum_10) { ::Test(pvs, FForum[10]); }
//
//HT ht{ 1 };
//PVS pvs_tt{ ht, AAGLEM{} };
//
//TEST(PVS_TT, Zero_empty_0) { TestAllWindows(pvs_tt, Zero_empty_0); }
//TEST(PVS_TT, Zero_empty_1) { TestAllWindows(pvs_tt, Zero_empty_1); }
//TEST(PVS_TT, Zero_empty_2) { TestAllWindows(pvs_tt, Zero_empty_2); }
//TEST(PVS_TT, One_empty_0) { TestAllWindows(pvs_tt, One_empty_0); }
//TEST(PVS_TT, One_empty_1) { TestAllWindows(pvs_tt, One_empty_1); }
//TEST(PVS_TT, One_empty_2) { TestAllWindows(pvs_tt, One_empty_2); }
//TEST(PVS_TT, One_empty_3) { TestAllWindows(pvs_tt, One_empty_3); }
//TEST(PVS_TT, Two_empty_0) { TestAllWindows(pvs_tt, Two_empty_0); }
//TEST(PVS_TT, Two_empty_1) { TestAllWindows(pvs_tt, Two_empty_1); }
//TEST(PVS_TT, Two_empty_2) { TestAllWindows(pvs_tt, Two_empty_2); }
//TEST(PVS_TT, Two_empty_3) { TestAllWindows(pvs_tt, Two_empty_3); }
//TEST(PVS_TT, Three_empty_0) { TestAllWindows(pvs_tt, Three_empty_0); }
//TEST(PVS_TT, Three_empty_1) { TestAllWindows(pvs_tt, Three_empty_1); }
//TEST(PVS_TT, Three_empty_2) { TestAllWindows(pvs_tt, Three_empty_2); }
//TEST(PVS_TT, Three_empty_3) { TestAllWindows(pvs_tt, Three_empty_3); }
//TEST(PVS_TT, Four_empty_0) { TestAllWindows(pvs_tt, Four_empty_0); }
//TEST(PVS_TT, Four_empty_1) { TestAllWindows(pvs_tt, Four_empty_1); }
//TEST(PVS_TT, Four_empty_2) { TestAllWindows(pvs_tt, Four_empty_2); }
//TEST(PVS_TT, Four_empty_3) { TestAllWindows(pvs_tt, Four_empty_3); }
//TEST(PVS_TT, Five_empty) { TestAllWindows(pvs_tt, Five_empty); }
//TEST(PVS_TT, FForum_1) { ::Test(pvs_tt, FForum[1]); }
//TEST(PVS_TT, FForum_2) { ::Test(pvs_tt, FForum[2]); }
//TEST(PVS_TT, FForum_3) { ::Test(pvs_tt, FForum[3]); }
//TEST(PVS_TT, FForum_4) { ::Test(pvs_tt, FForum[4]); }
//TEST(PVS_TT, FForum_5) { ::Test(pvs_tt, FForum[5]); }
//TEST(PVS_TT, FForum_6) { ::Test(pvs_tt, FForum[6]); }
//TEST(PVS_TT, FForum_7) { ::Test(pvs_tt, FForum[7]); }
//TEST(PVS_TT, FForum_8) { ::Test(pvs_tt, FForum[8]); }
//TEST(PVS_TT, FForum_9) { ::Test(pvs_tt, FForum[9]); }
//TEST(PVS_TT, FForum_10) { ::Test(pvs_tt, FForum[10]); }
