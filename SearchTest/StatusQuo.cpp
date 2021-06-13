#include "pch.h"
//
//// The signature of a PV-Search looks like this: ClosedInterval PVS(Position, OpenInterval, ... ),
//// where PVS finds the best move within (alpha, beta), where neither alpha nor beta matters.
//// Thus PVS returns an interval of one of these classes:
////   [-inf, score <= alpha]. This indicates that all moves were below the window of request.
////   [score, score], with alpha < score < beta. This indicates that the best move was within the window of request and was equal to score.
////   [score >= beta, +inf]. This indicates that one move was above the window of request.
////
//// SCORE <= ALPHA:
////      alpha  beta
//// ------(------)------> window of request
//// ----]---------------> move
//// ----]---------------> result
////
//// ALPHA < SCORE < BETA:
////      alpha  beta
//// ------(------)------> window of request
//// --------[]----------> move 1
//// ----------[]--------> move 2
//// ----------[]--------> result
////
//// SCORE >= BETA:
////      alpha  beta
//// ------(------)------> window of request
//// ----------------[---> move
//// ----------------[---> result
////
//// StautsQuo keeps track of what happened for the current position and has to fulfill the following requirements:
////   A move with score (score <= alpha) doesn't give a result, but updates internal values.			(See '')
////   An only move with score (score < alpha) gives a result of [-inf, score].						(See '')
////   A move with score (alpha < score < beta) doesn't give a result, but updates internal values.	(See '')
////   An only move with score (alpha < score < beta) gives a result of [score, score].				(See '')
////   A move with score (beta < score) gives a result of [score, +inf].								(See 'UpperCutTest')
//
//// {ExactScore, MinBound, MaxBound} x {low score, mid score, high score} x {request certainty : 2} x {move certainty : 2} x {score : 129} x {depth : 3}
//
//// Use cases:
//// - improve with move of exact score with score <= alpha
//// - improve with move of exact score with alpha < score < beta
//// - improve with move of exact score with beta <= score
//// - 
//
//// arbitrary values
//constexpr int alpha = 5;
//constexpr int beta = 10;
//constexpr int depth = 16;
//constexpr Confidence certainty(1.0);
//Search::Request intensity{ {alpha, beta}, depth, certainty };
//
//namespace Update
//{
//	TEST(StatusQuo_ImproveWithMove, low_low)
//	{
//		for (int lower : range(min_score, alpha + 1))
//			for (int upper : range(lower, alpha + 1))
//			{
//				StatusQuo status_quo(intensity);
//
//				// This simulates a child returning from a beta cut.
//				Result novum = Result({ lower, upper }, intensity.depth - 1, intensity.certainty, Field::A1, 10 /*nodes*/);
//				status_quo.Improve(novum, Field::A2);
//
//				EXPECT_FALSE(status_quo.IsUpperCut());
//				EXPECT_EQ(status_quo.SearchWindow().lower(), alpha);
//				EXPECT_EQ(status_quo.SearchWindow().upper(), beta);
//				EXPECT_EQ(status_quo.GetResult().window.lower(), lower);
//				EXPECT_EQ(status_quo.GetResult().window.upper(), upper);
//				EXPECT_EQ(status_quo.GetResult().best_move, Field::A2);
//				EXPECT_EQ(status_quo.GetResult().depth, intensity.depth);
//				EXPECT_EQ(status_quo.GetResult().certainty, intensity.certainty);
//				EXPECT_EQ(status_quo.GetResult().node_count, novum.node_count + 1);
//			}
//	}
//
//	TEST(StatusQuo_ImproveWithMove, CutTest_mid_produces_no_result_and_updates_internals)
//	{
//		for (int score : range(alpha + 1, beta))
//		{
//			StatusQuo status_quo(intensity);
//
//			Result novum = Result::ExactScore(score, intensity.depth - 1, intensity.certainty, Field::A1, 10 /*nodes*/);
//			status_quo.Improve(novum, Field::A2);
//
//			EXPECT_FALSE(status_quo.IsUpperCut());
//			EXPECT_EQ(status_quo.SearchWindow().lower(), score);
//			EXPECT_EQ(status_quo.SearchWindow().upper(), beta);
//			EXPECT_EQ(status_quo.GetResult().window.lower(), score);
//			EXPECT_EQ(status_quo.GetResult().window.upper(), score);
//			EXPECT_EQ(status_quo.GetResult().best_move, Field::A2);
//			EXPECT_EQ(status_quo.GetResult().depth, intensity.depth);
//			EXPECT_EQ(status_quo.GetResult().certainty, intensity.certainty);
//			EXPECT_EQ(status_quo.GetResult().node_count, novum.node_count + 1);
//		}
//	}
//
//	TEST(StatusQuo_ImproveWithMove, CutTest_high_produces_result)
//	{
//		for (int lower : range(beta, max_score + 1))
//			for (int upper : range(lower, max_score + 1))
//			{
//				StatusQuo status_quo(intensity);
//
//				// This simulates a child producing a beta cut.
//				Result novum = Result({ lower, upper }, intensity.depth - 1, intensity.certainty, Field::A1, 10 /*nodes*/);
//				status_quo.Improve(novum, Field::A2);
//
//				EXPECT_TRUE(status_quo.IsUpperCut());
//				auto result = status_quo.GetResult();
//				EXPECT_EQ(result.window.lower(), lower);
//				EXPECT_EQ(result.window.upper(), max_score);
//				EXPECT_EQ(result.depth, intensity.depth);
//				EXPECT_EQ(result.certainty, intensity.certainty);
//				EXPECT_EQ(result.best_move, Field::A2);
//				EXPECT_EQ(result.node_count, novum.node_count + 1);
//			}
//	}
//
//	TEST(StatusQuo_ImproveWithMove, CutTest_exact_full_window_search_produces_no_result)
//	{
//		for (int score : range(min_score, max_score))
//		{
//			Request intensity = Request::ExactScore({});
//			StatusQuo status_quo(intensity);
//
//			status_quo.Improve(Result::ExactScore(score, intensity.depth - 1, intensity.certainty, Field::invalid, 1 /*node_count*/), Field::A1);
//
//			EXPECT_FALSE(status_quo.IsUpperCut());
//		}
//	}
//}
//////////////////////////////////
//namespace SingleMove
//{
//	TEST(StatusQuo_ImproveWithMove, SingleMove_low)
//	{
//		for (int lower : range(min_score, alpha + 1))
//			for (int upper : range(lower, alpha + 1))
//			{
//				StatusQuo status_quo(intensity);
//
//				// This simulates a child returning from a beta cut.
//				status_quo.Improve(Result({ lower, upper }, intensity.depth - 1, intensity.certainty, Field::invalid, 1 /*nodes*/), Field::A1);
//				
//				EXPECT_FALSE(status_quo.IsUpperCut());
//				EXPECT_EQ(status_quo.GetResult().window.lower(), lower);
//				EXPECT_EQ(status_quo.GetResult().window.upper(), upper);
//				EXPECT_EQ(status_quo.GetResult().depth, intensity.depth);
//				EXPECT_EQ(status_quo.GetResult().certainty, intensity.certainty);
//				EXPECT_EQ(status_quo.GetResult().best_move, Field::A1);
//				EXPECT_EQ(status_quo.GetResult().node_count, 2);
//			}
//	}
//
//	TEST(StatusQuo_ImproveWithMove, SingleMove_mid)
//	{
//		for (int score : range(alpha + 1, beta))
//		{
//			StatusQuo status_quo(intensity);
//
//			status_quo.Improve(Result::ExactScore(score, intensity.depth - 1, intensity.certainty, Field::invalid, 1 /*nodes*/), Field::A1);
//			
//			EXPECT_FALSE(status_quo.IsUpperCut());
//			EXPECT_EQ(status_quo.GetResult().window.lower(), score);
//			EXPECT_EQ(status_quo.GetResult().window.upper(), score);
//			EXPECT_EQ(status_quo.GetResult().depth, intensity.depth);
//			EXPECT_EQ(status_quo.GetResult().certainty, intensity.certainty);
//			EXPECT_EQ(status_quo.GetResult().best_move, Field::A1);
//			EXPECT_EQ(status_quo.GetResult().node_count, 2);
//		}
//	}
//
//	// SingleMove_mid is covered by CutTest_high_produces_result.
//
//	TEST(StatusQuo_ImproveWithMove, SingleMove_exact_full_window_search)
//	{
//		for (int score : range(min_score, max_score))
//		{
//			Request intensity = Request::ExactScore({});
//			StatusQuo status_quo(intensity);
//
//			status_quo.Improve(Result::ExactScore(score, intensity.depth - 1, intensity.certainty, Field::invalid, 1 /*node_count*/), Field::A1);
//			
//			EXPECT_FALSE(status_quo.IsUpperCut());
//			EXPECT_EQ(status_quo.GetResult().window.lower(), score);
//			EXPECT_EQ(status_quo.GetResult().window.upper(), score);
//			EXPECT_EQ(status_quo.GetResult().depth, intensity.depth);
//			EXPECT_EQ(status_quo.GetResult().certainty, intensity.certainty);
//			EXPECT_EQ(status_quo.GetResult().best_move, Field::A1);
//			EXPECT_EQ(status_quo.GetResult().node_count, 2);
//		}
//	}
//}
//
//TEST(StatusQuo_ImproveWithMove, TwoMoves_mid_lower_first)
//{
//	for (int lower : range(alpha + 1, beta))
//		for (int upper : range(lower + 1, beta))
//		{
//			StatusQuo status_quo(intensity);
//
//			status_quo.Improve(Result::ExactScore(lower, intensity.depth - 1, intensity.certainty, Field::invalid, 1 /*nodes*/), Field::A1);
//			status_quo.Improve(Result::ExactScore(upper, intensity.depth - 1, intensity.certainty, Field::invalid, 1 /*nodes*/), Field::A2);
//			
//			EXPECT_FALSE(status_quo.IsUpperCut());
//			EXPECT_EQ(status_quo.GetResult().window.lower(), upper);
//			EXPECT_EQ(status_quo.GetResult().window.upper(), upper);
//			EXPECT_EQ(status_quo.GetResult().depth, intensity.depth);
//			EXPECT_EQ(status_quo.GetResult().certainty, intensity.certainty);
//			EXPECT_EQ(status_quo.GetResult().best_move, Field::A2);
//			EXPECT_EQ(status_quo.GetResult().node_count, 3);
//		}
//}
//
//TEST(StatusQuo_ImproveWithMove, TwoMoves_mid_higher_first)
//{
//	for (int lower : range(alpha + 1, beta))
//		for (int upper : range(lower + 1, beta))
//		{
//			StatusQuo status_quo(intensity);
//
//			status_quo.Improve(Result::ExactScore(upper, intensity.depth - 1, intensity.certainty, Field::invalid, 1 /*nodes*/), Field::A2);
//			status_quo.Improve(Result::ExactScore(lower, intensity.depth - 1, intensity.certainty, Field::invalid, 1 /*nodes*/), Field::A1);
//			
//			EXPECT_FALSE(status_quo.IsUpperCut());
//			EXPECT_EQ(status_quo.GetResult().window.lower(), upper);
//			EXPECT_EQ(status_quo.GetResult().window.upper(), upper);
//			EXPECT_EQ(status_quo.GetResult().depth, intensity.depth);
//			EXPECT_EQ(status_quo.GetResult().certainty, intensity.certainty);
//			EXPECT_EQ(status_quo.GetResult().best_move, Field::A2);
//			EXPECT_EQ(status_quo.GetResult().node_count, 3);
//		}
//}
//
//TEST(Requests_Improve, CutTest_exact_always_cuts)
//{
//	for (int score : range(min_score, max_score + 1))
//	{
//		Requests limits(intensity);
//		Result novum = Result::ExactScore(score, 28, Confidence(2.4), Field::A1, 10 /*nodes*/);
//
//		limits.Improve(novum);
//
//		EXPECT_TRUE(limits.HasResult());
//		EXPECT_EQ(limits.GetResult().window.lower(), score);
//		EXPECT_EQ(limits.GetResult().window.upper(), score);
//		EXPECT_EQ(limits.GetResult().depth, novum.depth);
//		EXPECT_EQ(limits.GetResult().certainty, novum.certainty);
//		EXPECT_EQ(limits.GetResult().best_move, novum.best_move);
//		EXPECT_EQ(limits.GetResult().node_count, novum.node_count + 1);
//	}
//}
//
//TEST(Requests_Improve, CutTest_low_low)
//{
//	for (int lower : range(min_score, alpha + 1))
//		for (int upper : range(lower, alpha + 1))
//		{
//			Requests limits(intensity);
//			Result novum({ lower, upper }, 28, Confidence(2.4), Field::A1, 10 /*nodes*/);
//			
//			limits.Improve(novum);
//
//			EXPECT_TRUE(limits.HasResult());
//			EXPECT_EQ(limits.GetResult().window.lower(), lower);
//			EXPECT_EQ(limits.GetResult().window.upper(), upper);
//			EXPECT_EQ(limits.GetResult().depth, novum.depth);
//			EXPECT_EQ(limits.GetResult().certainty, novum.certainty);
//			EXPECT_EQ(limits.GetResult().best_move, novum.best_move);
//			EXPECT_EQ(limits.GetResult().node_count, novum.node_count + 1);
//		}
//}
//
////TEST(Requests_Improve, CutTest_low_mid_overlap)
////{
////	for (int lower : range(min_score, alpha + 1))
////		for (int upper : range(alpha + 1, beta))
////		{
////			StatusQuo status_quo(intensity);
////			Result result({ lower, upper }, 28, Confidence(2.4), Field::A1, 1 /*nodes*/);
////
////			status_quo.ImproveWithRestriction(result);
////
////			EXPECT_FALSE(status_quo.IsUpperCut());
////			EXPECT_EQ(status_quo.SearchWindow().lower(), alpha);
////			EXPECT_EQ(status_quo.SearchWindow().upper(), upper + 1);
////			EXPECT_EQ(status_quo.GetResult().window.lower(), lower);
////			EXPECT_EQ(status_quo.GetResult().window.upper(), upper);
////			EXPECT_EQ(status_quo.GetResult().best_move, Field::invalid);
////			EXPECT_EQ(status_quo.GetResult().depth, result.depth);
////			EXPECT_EQ(status_quo.GetResult().certainty, result.certainty);
////			EXPECT_EQ(status_quo.GetResult().node_count, 1);
////		}
////}
////
////TEST(Requests_Improve, CutTest_low_high)
////{
////	for (int lower : range(min_score, alpha + 1))
////		for (int upper : range(beta, max_score + 1))
////		{
////			Requests limits(intensity);
////			Result result({ lower, upper }, 28, Confidence(2.4), Field::A1, 1 /*nodes*/);
////
////			limits.Improve(result);
////
////			EXPECT_FALSE(limits.HasResult());
////			EXPECT_EQ(limits.searching.lower(), alpha);
////			EXPECT_EQ(limits.searching.upper(), beta);
////			EXPECT_EQ(limits.possible.lower(), lower);
////			EXPECT_EQ(limits.possible.upper(), upper);
////			EXPECT_EQ(limits.best_move, Field::invalid);
////			EXPECT_EQ(limits.worst_depth, result.depth);
////			EXPECT_EQ(limits.worst_certainty, result.certainty);
////			EXPECT_EQ(limits.node_count, 1);
////		}
////}
////
////TEST(Requests_Improve, CutTest_mid_high)
////{
////	for (int lower : range(alpha + 1, beta))
////		for (int upper : range(beta, max_score + 1))
////		{
////			Requests limits(intensity);
////			Result result({ lower, upper }, 28, Confidence(2.4), Field::A1, 1 /*nodes*/);
////
////			limits.Improve(result);
////
////			EXPECT_FALSE(limits.HasResult());
////			EXPECT_EQ(limits.searching.lower(), lower - 1);
////			EXPECT_EQ(limits.searching.upper(), beta);
////			EXPECT_EQ(limits.possible.lower(), lower);
////			EXPECT_EQ(limits.possible.upper(), upper);
////			EXPECT_EQ(limits.best_move, Field::invalid);
////			EXPECT_EQ(limits.worst_depth, result.depth);
////			EXPECT_EQ(limits.worst_certainty, result.certainty);
////			EXPECT_EQ(limits.node_count, 1);
////		}
////}
//
//TEST(Requests_Improve, CutTest_high_high)
//{
//	for (int lower : range(beta, max_score + 1))
//		for (int upper : range(lower, max_score + 1))
//		{
//			Requests limits(intensity);
//			Result novum({ lower, upper }, 28, Confidence(2.4), Field::A1, 10 /*nodes*/);
//
//			limits.Improve(novum);
//
//			EXPECT_TRUE(limits.HasResult());
//			EXPECT_EQ(limits.GetResult().window.lower(), lower);
//			EXPECT_EQ(limits.GetResult().window.upper(), upper);
//			EXPECT_EQ(limits.GetResult().depth, novum.depth);
//			EXPECT_EQ(limits.GetResult().certainty, novum.certainty);
//			EXPECT_EQ(limits.GetResult().best_move, novum.best_move);
//			EXPECT_EQ(limits.GetResult().node_count, novum.node_count + 1);
//		}
//}
//
//// TODO: Improve with two infos!
