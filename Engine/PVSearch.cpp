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

Result PVSearch::PVS_N(const Position& pos, const Intensity intensity)
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

	//if (const auto tt_value = tt.LookUp(pos))


	StatusQuo status_quo(intensity);
	for (auto move : moves)
	{
		const auto result = -PVS_N(Play(pos, move), status_quo);
		if (result.Exceeds(status_quo.Window()))
		{
			const auto ret = status_quo.UpperCut(result);
			tt.Update(pos, ret);
			return ret;
		}
		status_quo.ImproveWith(result);
	}

	const auto ret = status_quo.AllMovesTried(intensity);
	tt.Update(pos, ret);
	return ret;
}

//Result PVSearch::ZWS_N(Position pos, Intensity intensity)
//{
//	return Result();
//}

PVSearch::StatusQuo::operator Intensity() const noexcept
{
	return -intensity - 1;
}

void PVSearch::StatusQuo::ImproveWith(Result novum)
{
	assert(novum.depth + 1 >= worst_depth);
	assert(novum.selectivity <= worst_selectivity);

	if (novum.window.lower > best_score)
	{
		intensity.window.lower = std::max(intensity.window.lower, novum.window.lower);
		best_score = novum.window.lower;
		best_move = novum.best_move;
	}
	worst_depth = std::max(worst_depth, novum.depth + 1);
	worst_selectivity = std::min(worst_selectivity, novum.selectivity);
	node_count += novum.node_count;
}

void Search::PVSearch::StatusQuo::ImproveWith(const std::optional<PVS_Info>& info)
{
	if (!info)
		return;


}

Result PVSearch::StatusQuo::UpperCut(Result result)
{
	assert(result.Exceeds(intensity.window));

	ImproveWith(result);
	return Result::MinBound(result.window.lower, worst_depth, worst_selectivity, best_move, node_count);
}

Result PVSearch::StatusQuo::AllMovesTried(Intensity intensity)
{
	assert(worst_depth >= intensity.depth);
	assert(worst_selectivity <= intensity.selectivity);

	if (best_score > intensity.window.lower)
		return Result::ExactScore(best_score, worst_depth, worst_selectivity, best_move, node_count);
	return Result::MaxBound(intensity.window.lower, worst_depth, worst_selectivity, best_move, node_count);
}
