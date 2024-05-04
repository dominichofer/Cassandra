#include "pch.h"

struct AlgorithmTest : ::testing::TestWithParam<ScoredPosition>
{
	std::vector<ScoredPosition> endgame, fforum;

	void SetUp()
	{
		endgame = LoadScoredPositionFile("..\\data\\endgame.ps");
		fforum = LoadScoredPositionFile("..\\data\\fforum-1-19.ps");
	}

	void test_full_window(Algorithm&& alg, const ScoredPosition& scored_position)
	{
		Result result = alg.Eval(scored_position.pos);
		EXPECT_EQ(result.window, ClosedInterval(scored_position.score, scored_position.score));
	}
	void test_full_window(Algorithm&& alg, const std::vector<ScoredPosition>& scored_positions)
	{
		for (const ScoredPosition& ps : scored_positions)
		{
			Result result = alg.Eval(ps.pos);
			EXPECT_EQ(result.window, ClosedInterval(ps.score, ps.score));
		}
	}

	void test_fail_high(Algorithm&& alg, const ScoredPosition& ps)
	{
		Result result = alg.Eval(ps.pos, OpenInterval{ min_score, std::min<Score>(ps.score, min_score + 1) });
		EXPECT_TRUE(result.window.Contains(ps.score));
	}
	void test_fail_high(Algorithm&& alg, const std::vector<ScoredPosition>& scored_positions)
	{
		for (const ScoredPosition& ps : scored_positions)
		{
			Result result = alg.Eval(ps.pos, OpenInterval{ min_score, std::min<Score>(ps.score, min_score + 1) });
			EXPECT_TRUE(result.window.Contains(ps.score));
		}
	}

	void test_fail_low(Algorithm&& alg, const ScoredPosition& ps)
	{
		Result result = alg.Eval(ps.pos, OpenInterval{ std::max<Score>(ps.score, max_score - 1), max_score });
		EXPECT_TRUE(result.window.Contains(ps.score));
	}
	void test_fail_low(Algorithm&& alg, const std::vector<ScoredPosition>& scored_positions)
	{
		for (const ScoredPosition& ps : scored_positions)
		{
			Result result = alg.Eval(ps.pos, OpenInterval{ std::max<Score>(ps.score, max_score - 1), max_score });
			EXPECT_TRUE(result.window.Contains(ps.score));
		}
	}
};

TEST_F(AlgorithmTest, NegaMax_Endgame_full_window) { test_full_window(NegaMax{}, endgame); }
TEST_F(AlgorithmTest, NegaMax_Endgame_fail_high) { test_fail_high(NegaMax{}, endgame); }
TEST_F(AlgorithmTest, NegaMax_Endgame_fail_low) { test_fail_low(NegaMax{}, endgame); }

TEST_F(AlgorithmTest, AlphaBeta_Endgame_full_window) { test_full_window(AlphaBeta{}, endgame); }
TEST_F(AlgorithmTest, AlphaBeta_Endgame_fail_high) { test_fail_high(AlphaBeta{}, endgame); }
TEST_F(AlgorithmTest, AlphaBeta_Endgame_fail_low) { test_fail_low(AlphaBeta{}, endgame); }
TEST_F(AlgorithmTest, AlphaBeta_FForum_full_window) { test_full_window(AlphaBeta{}, fforum); }
TEST_F(AlgorithmTest, AlphaBeta_FForum_fail_high) { test_fail_high(AlphaBeta{}, fforum); }
TEST_F(AlgorithmTest, AlphaBeta_FForum_fail_low) { test_fail_low(AlphaBeta{}, fforum); }

TEST_F(AlgorithmTest, PVS_Endgame_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, endgame); }
TEST_F(AlgorithmTest, PVS_Endgame_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, endgame); }
TEST_F(AlgorithmTest, PVS_Endgame_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, endgame); }
TEST_F(AlgorithmTest, PVS_FForum_00_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[0]); }
TEST_F(AlgorithmTest, PVS_FForum_01_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[1]); }
TEST_F(AlgorithmTest, PVS_FForum_02_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[2]); }
TEST_F(AlgorithmTest, PVS_FForum_03_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[3]); }
TEST_F(AlgorithmTest, PVS_FForum_04_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[4]); }
TEST_F(AlgorithmTest, PVS_FForum_05_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[5]); }
TEST_F(AlgorithmTest, PVS_FForum_06_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[6]); }
TEST_F(AlgorithmTest, PVS_FForum_07_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[7]); }
TEST_F(AlgorithmTest, PVS_FForum_08_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[8]); }
TEST_F(AlgorithmTest, PVS_FForum_09_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[9]); }
TEST_F(AlgorithmTest, PVS_FForum_10_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[10]); }
TEST_F(AlgorithmTest, PVS_FForum_11_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[11]); }
TEST_F(AlgorithmTest, PVS_FForum_12_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[12]); }
TEST_F(AlgorithmTest, PVS_FForum_13_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[13]); }
TEST_F(AlgorithmTest, PVS_FForum_14_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[14]); }
TEST_F(AlgorithmTest, PVS_FForum_15_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[15]); }
TEST_F(AlgorithmTest, PVS_FForum_16_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[16]); }
TEST_F(AlgorithmTest, PVS_FForum_17_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[17]); }
TEST_F(AlgorithmTest, PVS_FForum_18_full_window) { RAM_HashTable tt{ 1'000'000 }; test_full_window(PVS{ tt, EstimatorStub{} }, fforum[18]); }
TEST_F(AlgorithmTest, PVS_FForum_00_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[0]); }
TEST_F(AlgorithmTest, PVS_FForum_01_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[1]); }
TEST_F(AlgorithmTest, PVS_FForum_02_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[2]); }
TEST_F(AlgorithmTest, PVS_FForum_03_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[3]); }
TEST_F(AlgorithmTest, PVS_FForum_04_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[4]); }
TEST_F(AlgorithmTest, PVS_FForum_05_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[5]); }
TEST_F(AlgorithmTest, PVS_FForum_06_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[6]); }
TEST_F(AlgorithmTest, PVS_FForum_07_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[7]); }
TEST_F(AlgorithmTest, PVS_FForum_08_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[8]); }
TEST_F(AlgorithmTest, PVS_FForum_09_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[9]); }
TEST_F(AlgorithmTest, PVS_FForum_10_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[10]); }
TEST_F(AlgorithmTest, PVS_FForum_11_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[11]); }
TEST_F(AlgorithmTest, PVS_FForum_12_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[12]); }
TEST_F(AlgorithmTest, PVS_FForum_13_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[13]); }
TEST_F(AlgorithmTest, PVS_FForum_14_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[14]); }
TEST_F(AlgorithmTest, PVS_FForum_15_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[15]); }
TEST_F(AlgorithmTest, PVS_FForum_16_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[16]); }
TEST_F(AlgorithmTest, PVS_FForum_17_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[17]); }
TEST_F(AlgorithmTest, PVS_FForum_18_fail_high) { RAM_HashTable tt{ 1'000'000 }; test_fail_high(PVS{ tt, EstimatorStub{} }, fforum[18]); }
TEST_F(AlgorithmTest, PVS_FForum_00_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[0]); }
TEST_F(AlgorithmTest, PVS_FForum_01_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[1]); }
TEST_F(AlgorithmTest, PVS_FForum_02_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[2]); }
TEST_F(AlgorithmTest, PVS_FForum_03_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[3]); }
TEST_F(AlgorithmTest, PVS_FForum_04_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[4]); }
TEST_F(AlgorithmTest, PVS_FForum_05_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[5]); }
TEST_F(AlgorithmTest, PVS_FForum_06_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[6]); }
TEST_F(AlgorithmTest, PVS_FForum_07_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[7]); }
TEST_F(AlgorithmTest, PVS_FForum_08_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[8]); }
TEST_F(AlgorithmTest, PVS_FForum_09_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[9]); }
TEST_F(AlgorithmTest, PVS_FForum_10_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[10]); }
TEST_F(AlgorithmTest, PVS_FForum_11_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[11]); }
TEST_F(AlgorithmTest, PVS_FForum_12_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[12]); }
TEST_F(AlgorithmTest, PVS_FForum_13_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[13]); }
TEST_F(AlgorithmTest, PVS_FForum_14_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[14]); }
TEST_F(AlgorithmTest, PVS_FForum_15_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[15]); }
TEST_F(AlgorithmTest, PVS_FForum_16_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[16]); }
TEST_F(AlgorithmTest, PVS_FForum_17_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[17]); }
TEST_F(AlgorithmTest, PVS_FForum_18_fail_low) { RAM_HashTable tt{ 1'000'000 }; test_fail_low(PVS{ tt, EstimatorStub{} }, fforum[18]); }

//TEST_F(AlgorithmTest, MTD_Endgame)
//{
//	for (const ScoredPosition& ps : endgame)
//	{
//		RAM_HashTable tt{ 1'000'000 };
//		PVS pvs{ tt, EstimatorStub{} };
//		Result result = MTD{ pvs }.Eval(ps.pos);
//		EXPECT_EQ(result.GetScore(), ps.score);
//	}
//}
//
//TEST_F(AlgorithmTest, MTD_FForum1)
//{
//	for (const ScoredPosition& ps : fforum1)
//	{
//		RAM_HashTable tt{ 1'000'000 };
//		PVS pvs{ tt, EstimatorStub{} };
//		Result result = MTD{ pvs }.Eval(ps.pos);
//		EXPECT_EQ(result.GetScore(), ps.score);
//	}
//}
//
//TEST_F(AlgorithmTest, IDAB_Endgame)
//{
//	for (const ScoredPosition& ps : endgame)
//	{
//		RAM_HashTable tt{ 1'000'000 };
//		PVS pvs{ tt, EstimatorStub{} };
//		Result result = IDAB{ pvs }.Eval(ps.pos);
//		EXPECT_EQ(result.GetScore(), ps.score);
//	}
//}
//
//TEST_F(AlgorithmTest, IDAB_FForum1)
//{
//	for (const ScoredPosition& ps : fforum1)
//	{
//		RAM_HashTable tt{ 1'000'000 };
//		PVS pvs{ tt, EstimatorStub{} };
//		Result result = IDAB{ pvs }.Eval(ps.pos);
//		EXPECT_EQ(result.GetScore(), ps.score);
//	}
//}

//TEST_F(AlgorithmTest, DTS_Endgame)
//{
//	for (const ScoredPosition& ps : endgame)
//	{
//		RAM_HashTable tt{ 1'000'000 };
//		PVS pvs{ tt, EstimatorStub{} };
//		MoveSorter move_sorter{ tt, pvs };
//		Result result = DTS{ tt, pvs, move_sorter, 3 }.Eval(ps.pos);
//		EXPECT_EQ(result.score, ps.score);
//	}
//}

//TEST_F(AlgorithmTest, DTS_FForum1)
//{
//	for (const ScoredPosition& ps : fforum1)
//	{
//		HT tt{ 1'000'000 };
//		PVS pvs{ tt, EstimatorStub{} };
//		MoveSorter move_sorter{ tt, pvs };
//		Result result = DTS{ tt, pvs, move_sorter, 5 }.Eval(ps.pos);
//		EXPECT_EQ(result.score, ps.score);
//	}
//}

//TEST_F(AlgorithmTest, IDAB_DTS_Endgame)
//{
//	for (const ScoredPosition& ps : endgame)
//	{
//		HT tt{ 1'000'000 };
//		PVS pvs{ tt, EstimatorStub{} };
//		MoveSorter move_sorter{ tt, pvs };
//		DTS dts{ tt, pvs, move_sorter, 3 };
//		Result result = IDAB{ dts }.Eval(ps.pos);
//		EXPECT_EQ(result.score, ps.score);
//	}
//}
//
//TEST_F(AlgorithmTest, IDAB_DTS_FForum1)
//{
//	for (const ScoredPosition& ps : fforum1)
//	{
//		HT tt{ 1'000'000 };
//		PVS pvs{ tt, EstimatorStub{} };
//		MoveSorter move_sorter{ tt, pvs };
//		DTS dts{ tt, pvs, move_sorter, 3 };
//		Result result = IDAB{ dts }.Eval(ps.pos);
//		EXPECT_EQ(result.score, ps.score);
//	}
//}

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
