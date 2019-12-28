#include "PVSearch.h"
#include "Core/Machine.h"
#include <algorithm>

using namespace Search;


Result PVSearch::Eval(Position pos, Intensity intensity)
{
	if (pos.EmptyCount() <= 4)
		return AlphaBetaFailSoft{}.Eval(pos, intensity);
	return PVS_N(pos, intensity);
}

Result PVSearch::PVS_N(const Position& pos, const Intensity& intensity)
{
	if (pos.EmptyCount() <= 4)
		return Eval(pos, intensity);
	
	Moves moves = PossibleMoves(pos);
	if (moves.empty())
	{
		const auto passed = PlayPass(pos);
		if (PossibleMoves(passed).empty())
			return Result::ExactScore(EvalGameOver(pos), intensity, Field::invalid, 1 /*node_count*/);
		return -PVS_N(passed, -intensity);
	}

	StatusQuo status_quo(intensity);
	TT_Updater tt_updater(pos, tt, status_quo);

	status_quo.ImproveWithAny(tt.LookUp(pos));
	if (status_quo.HasResult())
		return status_quo.GetResult();

	for (auto move : moves)
	{
		const auto result = -PVS_N(Play(pos, move), status_quo);
		status_quo.ImproveWithMove(result, move);
		if (status_quo.HasResult())
			return status_quo.GetResult();
	}

	status_quo.AllMovesTried(intensity);
	return status_quo.GetResult();
}

//Result PVSearch::ZWS_N(Position pos, Intensity intensity)
//{
//	return Result();
//}

PVSearch::StatusQuo::operator Intensity() const noexcept
{
	return -intensity - 1;
}

void PVSearch::StatusQuo::ImproveWithMove(const Result& novum, Field move)
{
	assert(novum.depth + 1 >= worst_depth);
	assert(novum.selectivity <= worst_selectivity);

	intensity.window.lower = std::max(intensity.window.lower, novum.window.lower);
	if (novum.window.lower > best_score)
	{
		best_score = novum.window.lower;
		best_move = move;
	}
	worst_depth = std::min(worst_depth, novum.depth + 1);
	worst_selectivity = std::max(worst_selectivity, novum.selectivity);
	node_count += novum.node_count;

	if (best_score >= intensity.window.upper)
		result = Result::MinBound(intensity.window.lower, worst_depth, worst_selectivity, best_move, node_count);
}

void PVSearch::StatusQuo::ImproveWithAny(const Result& novum)
{
	const bool as_deep = (novum.depth >= intensity.depth);
	const bool as_selective = (novum.selectivity <= intensity.selectivity);

	if (as_deep and as_selective)
	{
		if (novum.window.lower == novum.window.upper) // exact score
			result = Result::ExactScore(novum.window.lower, novum.depth, novum.selectivity, novum.best_move, node_count);
		else if (novum.window.lower >= intensity.window.upper) // upper cut
			result = Result::MinBound(novum.window.lower, novum.depth, novum.selectivity, novum.best_move, node_count);
		else if (novum.window.upper <= intensity.window.lower) // lower cut
			result = Result::MinBound(novum.window.upper, novum.depth, novum.selectivity, novum.best_move, node_count);
		else
		{
			intensity.window.lower = std::max(intensity.window.lower, novum.window.lower);
			intensity.window.upper = std::min(intensity.window.upper, novum.window.upper);
		}
	}
}

void Search::PVSearch::StatusQuo::ImproveWithAny(const std::optional<Result>& novum)
{
	if (novum)
		ImproveWithAny(novum.value());
}

void PVSearch::StatusQuo::AllMovesTried(const Intensity& requested) // TODO: Remove Intensity, because the condition can be inferred via best_move???
{
	assert(worst_depth >= requested.depth);
	assert(worst_selectivity <= requested.selectivity);
	assert(best_score < requested.window.upper);

	if (best_score > requested.window.lower)
		result = Result::ExactScore(best_score, worst_depth, worst_selectivity, best_move, node_count);
	else
		result = Result::MaxBound(requested.window.lower, worst_depth, worst_selectivity, best_move, node_count);
}
