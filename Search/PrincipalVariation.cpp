#include "Game/Game.h"
#include "Algorithm.h"
#include "Stability.h"
#include <chrono>
#include <limits>
#include <iostream>

class Status
{
	int alpha;
	int best_score;
	Field best_move;
	float worst_confidence_level;
	int smallest_depth;
public:
	Status(int alpha)
		: alpha(alpha)
		, best_score(-inf_score)
		, best_move(Field::PS)
		, worst_confidence_level(inf)
		, smallest_depth(64)
	{}

	void Update(const Result& result, Field move)
	{
		worst_confidence_level = std::min(worst_confidence_level, result.confidence_level);
		smallest_depth = std::min(smallest_depth, result.depth + 1);
		if (-result.score > best_score)
		{
			best_score = -result.score;
			best_move = move;
		}
	}
	Result GetResult()
	{
		ResultType type = (best_score > alpha) ? ResultType::exact : ResultType::fail_low;
		return Result(type, best_score, smallest_depth, worst_confidence_level, best_move);
	}
};

PVS::PVS(HT& tt, const Estimator& estimator) noexcept : tt(tt), estimator(estimator) {}

Result PVS::Eval(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	return PVS_N(pos, window, std::min(depth, pos.EmptyCount()), confidence_level);
}

Result PVS::PVS_N(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	const bool midgame = (depth < pos.EmptyCount());
	if (midgame and depth <= 2)
		return Eval_dN(pos, window, depth);
	if (not midgame and depth == 0)
		return EndScore(pos);

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -PVS_N(passed, -window, depth, confidence_level);
		return EndScore(pos);
	}

	// Stability
	if (auto max = StabilityBasedMaxScore(pos); max < window)
		return Result::FailLow(max, pos.EmptyCount(), inf, Field::PS);

	// Transposition table
	if (auto ttc = TTC(pos, window, depth, confidence_level); ttc.has_value())
		return ttc.value();

	auto status = Status(window.lower);
	bool first = true;
	for (Field move : Sorted(pos, depth, confidence_level))
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
	const bool midgame = (depth < pos.EmptyCount());
	if (midgame and depth <= 2)
		return Eval_dN(pos, window, depth);
	if (not midgame and pos.EmptyCount() <= 7)
		return Result::Exact(AlphaBeta::Eval_P(pos, window), pos.EmptyCount(), inf, Field::PS);

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -ZWS_N(passed, -window, depth, confidence_level);
		return EndScore(pos);
	}

	// Stability Cut
	if (auto max = StabilityBasedMaxScore(pos); max < window)
		return Result::FailLow(max, pos.EmptyCount(), inf, Field::PS);

	// Transposition Table Cut
	if (auto ttc = TTC(pos, window, depth, confidence_level); ttc.has_value())
		return ttc.value();

	// Enhanced Transposition Cut
	if (midgame ? depth > 5 : depth > 14)
		if (auto etc = ETC(pos, window, depth, confidence_level); etc.has_value())
			return etc.value();

	// Multi Prob Cut
	if (midgame ? depth > 3 : depth > 14)
		if (auto mpc = MPC(pos, window, depth, confidence_level); mpc.has_value())
			return mpc.value();

	auto status = Status(window.lower);
	for (Field move : Sorted(pos, depth, confidence_level))
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

Result PVS::Eval_dN(const Position& pos, OpenInterval window, int depth)
{
	if (depth == 0)
		return Eval_d0(pos);

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -Eval_dN(passed, -window, depth);
		return EndScore(pos);
	}

	auto status = Status(window.lower);
	for (Field move : moves)
	{
		auto result = Eval_dN(Play(pos, move), -window, depth - 1);
		if (-result.score > window)
			return result.BetaCut(move);
		status.Update(result, move);
		window.lower = std::max(window.lower, -result.score);
	}
	return status.GetResult();
}

Result PVS::Eval_d0(const Position& pos)
{
	nodes++;
	int score = std::clamp(static_cast<int>(std::round(estimator.Score(pos))), min_score, max_score);
	return Result::Exact(score, 0, inf, Field::PS);
}

Result PVS::EndScore(const Position& pos)
{
	nodes++;
	return Result::Exact(::EndScore(pos), pos.EmptyCount(), inf, Field::PS);
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

SortedMoves PVS::Sorted(const Position& pos, int depth, float confidence_level)
{
	Field tt_move = Field::PS;
	if (auto look_up = tt.LookUp(pos); look_up.has_value())
		tt_move = look_up.value().best_move;
	int sort_depth = (depth - 14) / 2;

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
		if (sort_depth >= 0)
			score += (32 - PVS_N(next, { -inf_score, +inf_score}, sort_depth, confidence_level).score) << 15;
		return score;
	};
	return SortedMoves(PossibleMoves(pos), metric);
}

std::optional<Result> PVS::TTC(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	if (auto look_up = tt.LookUp(pos); look_up.has_value())
		if (auto t = look_up.value(); t.depth >= depth and t.confidence_level >= confidence_level)
		{
			if (t.IsExact())
				return Result::Exact(t.window.lower, t.depth, t.confidence_level, t.best_move);
			if (t.window > window)
				return Result::FailHigh(t.window.lower, t.depth, t.confidence_level, t.best_move);
			if (t.window < window)
				return Result::FailLow(t.window.upper, t.depth, t.confidence_level, t.best_move);
		}
	return std::nullopt;
}

std::optional<Result> PVS::ETC(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	for (Field move : PossibleMoves(pos))
		if (auto look_up = tt.LookUp(Play(pos, move)); look_up.has_value())
			if (auto t = look_up.value(); t.depth >= depth - 1 and t.confidence_level >= confidence_level and t.window < -window)
				return Result::FailHigh(-t.window.upper, t.depth + 1, confidence_level, move);
	return std::nullopt;
}

std::optional<Result> PVS::MPC(const Position& pos, OpenInterval window, int D, float confidence_level)
{
	if (confidence_level == inf or pos.EmptyCount() >= 55)
		return std::nullopt;

	float sd_0 = estimator.Accuracy(pos.EmptyCount(), 0, D);
	int margin_0 = static_cast<int>(std::ceil(confidence_level * sd_0));
	OpenInterval window_0{ window.lower - margin_0, window.upper + margin_0 };

	float eval_0 = estimator.Score(pos);
	if (eval_0 > window_0)
		return Result::FailHigh(window.upper, D, (eval_0 - window.upper) / sd_0, Field::PS);
	if (eval_0 < window_0)
		return Result::FailLow(window.lower, D, (window.lower - eval_0) / sd_0, Field::PS);

	int d = D / 2;
	float sd_d = estimator.Accuracy(pos.EmptyCount(), d, D);
	int margin_d = static_cast<int>(std::ceil(confidence_level * sd_d));
	OpenInterval window_d{ window.lower - margin_d, window.upper + margin_d };

	if (eval_0 > window_d and window_d < max_score)
	{
		OpenInterval zero_window{ window_d.upper - 1, window_d.upper };
		Result eval_d = ZWS_N(pos, zero_window, d, confidence_level);
		if (-eval_d.score > window_d)
			return Result::FailHigh(window.upper, D, (-eval_d.score - window.upper) / sd_d, eval_d.best_move);
	}

	if (eval_0 < window_d and window_d > min_score)
	{
		OpenInterval zero_window{ window_d.lower, window_d.lower + 1 };
		Result eval_d = ZWS_N(pos, zero_window, d, confidence_level);
		if (-eval_d.score < window_d)
			return Result::FailLow(window.lower, D, (window.lower + eval_d.score) / sd_d, eval_d.best_move);
	}

	return std::nullopt;
}
