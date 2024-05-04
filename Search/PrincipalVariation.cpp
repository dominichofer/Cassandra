#include "Game/Game.h"
#include "Algorithm.h"
#include "Status.h"
#include <vector>

Result BetaCut(const Result& result, Field move)
{
	return Result::FailHigh(result.window.lower, result.intensity + 1, move);
}

Result EndResult(const Position& pos)
{
	Score score = EndScore(pos);
	return Result::Exact(score, pos.EmptyCount(), Field::PS);
}

PVS::PVS(HashTable& tt, const Estimator& estimator) noexcept
	: tt(tt)
	, estimator(estimator)
{}

Result PVS::Eval(const Position& pos, OpenInterval window, Intensity intensity)
{
	return PVS_N(pos, window, intensity);
}

SortedMoves PVS::Sorted(const Position& pos, Intensity intensity)
{
	return MoveSorter(tt, *this).Sorted(pos, intensity);
}

Result PVS::PVS_N(const Position& pos, OpenInterval window, Intensity intensity)
{
	if (IsStop())
		return {};
	const bool midgame = (intensity.depth < pos.EmptyCount());
	if (midgame and intensity.depth <= 2)
		return Eval_dN(pos, window, intensity.depth);
	if (not midgame and pos.EmptyCount() <= 7)
		return AlphaBeta::Eval(pos, window, intensity);

	nodes++;
	if (not PossibleMoves(pos))
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -PVS_N(passed, -window, intensity);
		return EndResult(pos);
	}

	// Transposition table
	if (auto tte = TTC(pos, window, intensity); tte.has_value())
		return tte.value();

	Status status{ window.lower };
	bool first = true;
	for (Field move : Sorted(pos, intensity))
	{
		if (not first)
		{
			auto zero_window = OpenInterval{ window.lower, window.lower + 1 };
			auto result = -ZWS_N(Play(pos, move), -zero_window, intensity - 1);
			if (result.window < window) {
				status.Update(result, move);
				continue;
			}
		}

		auto result = -PVS_N(Play(pos, move), -window, intensity - 1);
		if (result.window > window) { // beta cut
			auto ret = BetaCut(result, move);
			InsertTT(pos, ret);
			return ret;
		}
		status.Update(result, move);
		window.lower = std::max(window.lower, result.window.lower);
		first = false;
	}
	auto ret = status.GetResult();
	InsertTT(pos, ret);
	return ret;
}

Result PVS::ZWS_N(const Position& pos, OpenInterval window, Intensity intensity)
{
	if (IsStop())
		return {};
	const bool midgame = (intensity.depth < pos.EmptyCount());
	if (midgame and intensity.depth <= 2)
		return Eval_dN(pos, window, intensity.depth);
	if (not midgame and pos.EmptyCount() <= 7)
		return AlphaBeta::Eval(pos, window, intensity);

	nodes++;
	if (not PossibleMoves(pos))
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -ZWS_N(passed, -window, intensity);
		return EndResult(pos);
	}

	// Stability Cut
	if (auto max = StabilityBasedMaxScore(pos); max < window)
		return Result::FailLow(max, pos.EmptyCount(), Field::PS);

	// Transposition Table Cut
	if (auto ttc = TTC(pos, window, intensity); ttc.has_value())
		return ttc.value();

	// Enhanced Transposition Cut
	if (midgame ? intensity.depth > 5 : intensity.depth > 12)
		if (auto etc = ETC(pos, window, intensity); etc.has_value())
			return etc.value();

	// Multi Prob Cut
	if (midgame or intensity.depth > 12)
		if (auto mpc = MPC(pos, window, intensity); mpc.has_value())
			return mpc.value();

	//if (depth > 12)
	//	return Parallel_ZWS_N(pos, window, intensity);

	Status status{ window.lower };
	for (Field move : Sorted(pos, intensity))
	{
		auto result = -ZWS_N(Play(pos, move), -window, intensity - 1);
		if (result.window > window) { // beta cut
			auto ret = BetaCut(result, move);
			InsertTT(pos, ret);
			return ret;
		}
		status.Update(result, move);
	}
	auto ret = status.GetResult();
	InsertTT(pos, ret);
	return ret;
}
//
//Result PVS::Parallel_ZWS_N(const Position& pos, OpenInterval window, Intensity intensity)
//{
//	auto moves = Sorted(pos, depth, level);
//	Status status{ window.lower };
//
//	// First move
//	auto move = moves[0];
//	auto result = ZWS_N(Play(pos, move), -window, depth - 1, level);
//	if (-result.score > window) { // beta cut
//		auto ret = BetaCut(result, move);
//		InsertTT(pos, ret);
//		return ret;
//	}
//	status.Update(result, move);
//
//	PVS pvs{ tt, estimator };
//	//parallel_nodes = std::addressof(pvs);
//	Result ret;
//	#pragma omp parallel for num_threads(3)
//	for (int i = 1; i < moves.size(); i++)
//	{
//		auto move = moves[i];
//		auto result = pvs.ZWS_N(Play(pos, move), -window, depth - 1, level);
//		if (pvs.IsStop())
//			continue;
//		if (-result.score > window) { // beta cut
//			if (not pvs.Stop()) {
//				ret = BetaCut(result, move);
//				InsertTT(pos, ret);
//			}
//			continue;
//		}
//		#pragma omp critical
//		status.Update(result, move);
//	}
//	//parallel_nodes = nullptr;
//	if (pvs.IsStop())
//		return ret;
//
//	//for (int i = 1; i < moves.size(); i++)
//	//{
//	//	auto move = moves[i];
//	//	auto result = ZWS_N(Play(pos, move), -window, depth - 1, level);
//	//	if (-result.score > window) { // beta cut
//	//		auto ret = BetaCut(result, move);
//	//		InsertTT(pos, ret);
//	//		return ret;
//	//	}
//	//	status.Update(result, move);
//	//}
//	{
//		auto ret = status.GetResult();
//		InsertTT(pos, ret);
//		return ret;
//	}
//}

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
		return EndResult(pos);
	}

	Status status{ window.lower };
	for (Field move : moves)
	{
		auto result = -Eval_dN(Play(pos, move), -window, depth - 1);
		if (result.window > window) // beta cut
			return Result::FailHigh(result.window.lower, result.intensity + 1, move);
		status.Update(result, move);
		window.lower = std::max(window.lower, result.window.lower);
	}
	return status.GetResult();
}

Result PVS::Eval_d0(const Position& pos)
{
	nodes++;
	Score score = std::clamp<int>(static_cast<int>(std::round(estimator.Score(pos))), min_score, max_score);
	return Result::Exact(score, 0, Field::PS);
}

std::optional<Result> PVS::TTC(const Position& pos, OpenInterval window, Intensity intensity)
{
	auto look_up = tt.LookUp(pos);
	if (auto look_up = tt.LookUp(pos); look_up.has_value())
		if (auto t = look_up.value(); t.intensity >= intensity)
			if (t.IsExact() or not t.window.Overlaps(window))
				return t;
	return std::nullopt;
}

std::optional<Result> PVS::ETC(const Position& pos, OpenInterval window, Intensity intensity)
{
	bool all_lower = true;
	for (Field move : PossibleMoves(pos))
	{
		if (auto look_up = tt.LookUp(Play(pos, move)); look_up.has_value())
			if (auto t = look_up.value(); t.intensity + 1 >= intensity)
			{
				if (-t.window > window)
					return Result::FailHigh(-t.window.upper, t.intensity + 1, move);
				if (-t.window < window)
					continue;
			}
		all_lower = false;
	}
	if (all_lower)
		return Result::FailLow(window.lower, intensity, Field::PS);
	return std::nullopt;
}

std::optional<Result> PVS::MPC(const Position& pos, OpenInterval window, Intensity intensity)
{
	if (intensity.IsExact() or pos.EmptyCount() >= 55)
		return std::nullopt;

	int8_t D = intensity.depth;
	float sd_0 = estimator.Accuracy(pos.EmptyCount(), 0, D);
	int margin_0 = static_cast<int>(std::ceil(intensity.level * sd_0));
	OpenInterval window_0{ window.lower - margin_0, window.upper + margin_0 };

	float eval_0 = estimator.Score(pos);
	if (eval_0 > window_0)
		return Result::FailHigh(window.upper, { D, (eval_0 - window.upper) / sd_0 }, Field::PS);
	if (eval_0 < window_0)
		return Result::FailLow(window.lower, { D, (window.lower - eval_0) / sd_0 }, Field::PS);

	int8_t d = D / 2;
	float sd_d = estimator.Accuracy(pos.EmptyCount(), d, D);
	int margin_d = static_cast<int>(std::ceil(intensity.level * sd_d));
	OpenInterval window_d{ window.lower - margin_d, window.upper + margin_d };

	if (eval_0 > window_d and window_d < max_score)
	{
		OpenInterval zero_window{ window_d.upper - 1, window_d.upper };
		Result eval_d = ZWS_N(pos, zero_window, { d, intensity.level });
		if (-eval_d.window > window_d)
			return Result::FailHigh(window.upper, { D, (-eval_d.GetScore() - window.upper) / sd_d }, eval_d.best_move);
	}

	if (eval_0 < window_d and window_d > min_score)
	{
		OpenInterval zero_window{ window_d.lower, window_d.lower + 1 };
		Result eval_d = ZWS_N(pos, zero_window, { d, intensity.level });
		if (-eval_d.window < window_d)
			return Result::FailLow(window.lower, {D, (window.lower + eval_d.GetScore()) / sd_d}, eval_d.best_move);
	}

	return std::nullopt;
}

void PVS::InsertTT(const Position& pos, const Result& result)
{
	if (not IsStop())
		tt.Insert(pos, result);
}
