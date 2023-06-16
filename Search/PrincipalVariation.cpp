#include "PrincipalVariation.h"
#include "Stability.h"
#include <chrono>
#include <limits>
#include <iostream>

Status::Status(int alpha)
	: alpha(alpha)
	, best_score(-inf_score)
	, best_move(Field::PS)
	, worst_confidence_level(std::numeric_limits<float>::infinity())
	, smallest_depth(64)
{}

void Status::Update(Result result, Field move)
{
	worst_confidence_level = std::min(worst_confidence_level, result.confidence_level);
	smallest_depth = std::min(smallest_depth, result.depth + 1);
	if (-result.score > best_score)
	{
		best_score = -result.score;
		best_move = move;
	}
}

Result Status::GetResult()
{
	ResultType type = (best_score > alpha) ? ResultType::exact : ResultType::fail_low;
	return Result(type, best_score, smallest_depth, worst_confidence_level, best_move);
}

PVS::PVS(HT& tt, const Estimator& estimator) noexcept : tt(tt), estimator(estimator) {}

ResultTimeNodes PVS::Eval(const Position& pos)
{
	return Eval(pos, { -inf_score, +inf_score }, pos.EmptyCount(), std::numeric_limits<float>::infinity());
}

ResultTimeNodes PVS::Eval(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	nodes = 0;
	auto start = std::chrono::high_resolution_clock::now();
	auto result = PVS_N(pos, window, depth, confidence_level);
	auto time = std::chrono::high_resolution_clock::now() - start;
	return { result, time, nodes };
}

Result PVS::PVS_N(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -PVS_N(passed, -window, depth, confidence_level);
		return EndScore(pos);
	}
	//if (moves.size() == 1)
	//	return -PVS_N(Play(pos, moves.front()), -window, depth - 1, confidence_level); // TODO: This needs to return depth + 1!

	if (depth == 0)
		return Eval_d0(pos);
	if (pos.EmptyCount() <= 7)
		return Result::Exact(AlphaBeta::Eval_N(pos, window), depth, std::numeric_limits<float>::infinity(), Field::PS);
		
	// Transposition table
	if (auto look_up = tt.LookUp(pos); look_up.has_value()) {
		auto t = look_up.value();
		if (t.depth >= depth and t.confidence_level >= confidence_level and (t.IsExact() or not t.Window().Overlaps(window)))
			return t;
	}

	auto status = Status(window.lower);
	bool first = true;
	for (Field move : Sorted(pos, window, depth, confidence_level))
	{
		if (not first)
		{
			auto zero_window = OpenInterval(window.lower, window.lower + 1);
			auto result = ZWS_N(Play(pos, move), -zero_window, depth - 1, confidence_level);
			if (-result.score < zero_window) {
				status.Update(result, move);
				continue;
			}
			if (-result.score > window) { // beta cut
				auto ret = result.BetaCut(move);
				tt.Update(pos, ret);
				return ret;
			}
		}

		auto result = PVS_N(Play(pos, move), -window, depth - 1, confidence_level);
		if (-result.score > window) { // beta cut
			auto ret = result.BetaCut(move);
			tt.Update(pos, ret);
			return ret;
		}
		status.Update(result, move);
		window.lower = std::max(window.lower, -result.score);
		first = false;
	}
	auto ret = status.GetResult();
	tt.Update(pos, ret);
	return ret;
}

Result PVS::ZWS_N(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -ZWS_N(passed, -window, depth, std::numeric_limits<float>::infinity());
		return EndScore(pos);
	}
	//if (moves.size() == 1)
	//	return -ZWS_N(Play(pos, moves.front()), -window, depth - 1, confidence_level); // TODO: This needs to return depth + 1!

	switch (depth)
	{
	case 0: return Eval_d0(pos);
	//case 1: return Eval_d1(pos);
	}
	if (pos.EmptyCount() <= 7)
		return Result::Exact(AlphaBeta::Eval_N(pos, window), pos.EmptyCount(), std::numeric_limits<float>::infinity(), Field::PS);

	// Stability
	if (auto max = StabilityBasedMaxScore(pos); max < window)
		return Result::FailLow(max, pos.EmptyCount(), std::numeric_limits<float>::infinity(), Field::PS);

	// Transposition table
	if (auto look_up = tt.LookUp(pos); look_up.has_value()) {
		auto t = look_up.value();
		if (t.depth >= depth and t.confidence_level >= confidence_level and (t.IsExact() or not t.Window().Overlaps(window)))
			return t;
	}

	// Multi Prob Cut
	if (auto mpc = MPC(pos, window, depth, confidence_level); mpc.has_value())
		return mpc.value();

	auto status = Status(window.lower);
	for (Field move : Sorted(pos, window, depth, confidence_level))
	{
		auto result = ZWS_N(Play(pos, move), -window, depth - 1, confidence_level);
		if (-result.score > window) { // beta cut
			auto ret = result.BetaCut(move);
			tt.Update(pos, ret);
			return ret;
		}
		status.Update(result, move);
	}
	auto ret = status.GetResult();
	tt.Update(pos, ret);
	return ret;
}

Result PVS::EndScore(const Position& pos)
{
	nodes++;
	return Result::Exact(::EndScore(pos), pos.EmptyCount(), std::numeric_limits<float>::infinity(), Field::PS);
}

Result PVS::Eval_d1(const Position& pos)
{
	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -Eval_d1(passed);
		return EndScore(pos);
	}

	int best_score = -inf_score;
	Field best_move = Field::PS;
	for (Field move : moves)
	{
		int score = -Eval_0(Play(pos, move));
		if (score > best_score)
		{
			best_score = score;
			best_move = move;
		}
	}
	return Result::Exact(best_score, 1, std::numeric_limits<float>::infinity(), best_move);
}

Result PVS::Eval_d0(const Position& pos)
{
	nodes++;
	return Result::Exact(estimator.Score(pos), 0, std::numeric_limits<float>::infinity(), Field::PS);
}

const uint32_t SquareValue[] = {
	9, 2, 8, 6, 6, 8, 2, 9,
	2, 1, 3, 4, 4, 3, 1, 2,
	8, 3, 7, 5, 5, 7, 3, 8,
	6, 4, 5, 0, 0, 5, 4, 6,
	6, 4, 5, 0, 0, 5, 4, 6,
	8, 3, 7, 5, 5, 7, 3, 8,
	2, 1, 3, 4, 4, 3, 1, 2,
	9, 2, 8, 6, 6, 8, 2, 9,
};

int double_corner_popcount(uint64_t b)
{
	return std::popcount(b) + std::popcount(b & 0x8100000000000081ULL);
}

SortedMoves PVS::Sorted(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	Field tt_move = Field::PS;
	if (auto look_up = tt.LookUp(pos); look_up.has_value())
		tt_move = look_up.value().best_move;

	auto metric = [&](Field move) -> uint32_t
	{
		if (move == tt_move)
			return 0x800000U;

		Position next = Play(pos, move);
		uint64_t O = next.Opponent();
		uint64_t E = next.Empties();

		uint32_t score = SquareValue[static_cast<uint8_t>(move)];
		score += (36 - double_corner_popcount(EightNeighboursAndSelf(O) & E)) << 4; // potential mobility, with corner bonus
		score += std::popcount(StableEdges(next) & O) << 10;
		score += (36 - double_corner_popcount(PossibleMoves(next))) << 15; // possible moves, with corner bonus
		return score;
	};
	return SortedMoves(PossibleMoves(pos), metric);
}

std::optional<Result> PVS::MPC(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	if (confidence_level == std::numeric_limits<float>::infinity())
		return std::nullopt;
	if (depth < 5)
		return std::nullopt;

	float eval_0 = estimator.Score(pos);

	float sd_0 = estimator.Accuracy(pos, 0, depth);
	int confidence_margin_d0 = std::ceil(confidence_level * sd_0);
	OpenInterval conficence_window_d0(window.lower - confidence_margin_d0, window.upper + confidence_margin_d0);

	if (eval_0 > conficence_window_d0)
		return Result::FailHigh(window.upper, depth, (eval_0 - window.upper) / sd_0, Field::PS);
	if (eval_0 < conficence_window_d0)
		return Result::FailLow(window.lower, depth, (window.lower - eval_0) / sd_0, Field::PS);

	//if (depth < 6)
	//	return std::nullopt;
	//int reduced_depth = depth / 3;
	////if (pos.EmptyCount() - reduced_depth < 10)
	////	return std::nullopt;

	//float sd = estimator.Accuracy(pos, reduced_depth, depth);
	//int confidence_margin = std::ceil(confidence_level * sd);
	//OpenInterval conficence_window(window.lower - confidence_margin, window.upper + confidence_margin);

	//if (eval_0 > window and conficence_window < max_score)
	//{
	//	OpenInterval zero_window(conficence_window.upper - 1, conficence_window.upper);
	//	Result shallow_result = ZWS_N(pos, zero_window, reduced_depth, std::numeric_limits<float>::infinity());
	//	if (-shallow_result.score > zero_window)
	//		return Result::FailHigh(conficence_window.upper, depth, (-shallow_result.score - window.upper) / sd, shallow_result.best_move);
	//}

	//if (eval_0 < window and conficence_window > min_score)
	//{
	//	OpenInterval zero_window(conficence_window.lower, conficence_window.lower + 1);
	//	Result shallow_result = ZWS_N(pos, zero_window, reduced_depth, std::numeric_limits<float>::infinity());
	//	if (-shallow_result.score < zero_window)
	//		return Result::FailLow(conficence_window.lower, depth, (window.lower + shallow_result.score) / sd, shallow_result.best_move);
	//}

	return std::nullopt;
}