#include "Core/Core.h"
#include "Algorithm.h"
#include "SortedMoves.h"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace Search;

float Sigma(int D, int d, int E) noexcept
{
	static const auto [alpha, beta, gamma, delta, epsilon] = std::make_tuple(-0.191047,1.00824,0.24472,-0.0118514,1.13362);
	//float alpha = -0.21311527f;
	//float beta = 1.06454983f;
	//float gamma = 0.26639884f;
	//float delta = -0.02005392f;
	//float epsilon = 2.09164003f;

	float sigma = (std::expf(alpha * d) + beta) * std::powf(D - d, gamma) * (delta * E + epsilon);
	assert(sigma > 0.0f);
	return sigma;
}

Result PVS::Eval(const Position& pos, const Request& request)
{
	return PVS_N(pos, request);
}

std::optional<Result> PVS::MPC(const Position& pos, const Request& request)
{
	if ((request.depth() < 4) || (request.certainty() == ConfidenceLevel::Certain()))
		return std::nullopt;

	// log.AddSearch("MPC", pos, request);
	float sigma_0 = Sigma(request.depth(), 0, pos.EmptyCount());
	//int upper_0 = static_cast<int>(std::ceil(request.window.upper() + sigma_0 * request.certainty().sigmas()));
	//int lower_0 = static_cast<int>(std::floor(request.window.lower() - sigma_0 * request.certainty().sigmas()));
	//float upper_0 = request.window.upper() + sigma_0 * request.certainty().sigmas();
	//float lower_0 = request.window.lower() - sigma_0 * request.certainty().sigmas();
	float eval_0 = evaluator.Eval(pos);
	//if (eval_0 >= upper_0) {
	//	float sigmas = (eval_0 - request.window.upper()) / sigma_0;
	//	auto result = Result{ {request.depth(), ConfidenceLevel{sigmas}}, {request.window.upper(), max_score} };
	//	// log.Add("Upper cut", result);
	//	return result;
	//}
	//if (eval_0 <= lower_0) {
	//	float sigmas = (request.window.lower() - eval_0) / sigma_0;
	//	auto result = Result{ {request.depth(), ConfidenceLevel{sigmas}}, {min_score, request.window.lower()} };
	//	// log.Add("Lower cut", result);
	//	return result;
	//}

	for (int reduced_depth : {1})
	{
	float sigma = Sigma(request.depth(), reduced_depth, pos.EmptyCount());
	int upper = static_cast<int>(std::ceil(request.window.upper() + sigma * request.certainty().sigmas()));
	int lower = static_cast<int>(std::floor(request.window.lower() - sigma * request.certainty().sigmas()));

	if (upper < max_score && eval_0 >= upper) {
		Request upper_zws = Request::Certain(reduced_depth, {upper-1, upper});
		Result shallow_result = ZWS_N(pos, upper_zws);
		if (shallow_result.window > upper_zws.window) { // Fail high
			float sigmas = (shallow_result.window.lower() - request.window.upper()) / Sigma(request.depth(), shallow_result.depth(), pos.EmptyCount());
			assert(sigmas > request.certainty().sigmas());
			auto result = Result{ {request.depth(), ConfidenceLevel{sigmas}}, {request.window.upper(), max_score} };
			// log.Add("Upper cut", result);
			return result;
		}
	}

	if (lower > min_score && eval_0 <= lower) {
		Request lower_zws = Request::Certain(reduced_depth, {lower, lower+1});
		Result shallow_result = ZWS_N(pos, lower_zws);
		if (shallow_result.window < lower_zws.window) { // Fail low
			float sigmas = (request.window.lower() - shallow_result.window.upper()) / Sigma(request.depth(), shallow_result.depth(), pos.EmptyCount());
			auto result = Result{ {request.depth(), ConfidenceLevel{sigmas}}, {min_score, request.window.lower()} };
			// log.Add("Lower cut", result);
			return result;
		}
	}
	}
	// log.FinalizeSearch();
	return std::nullopt;
}

Result PVS::PVS_N(const Position& pos, const Request& request)
{
	// log.AddSearch("PVS", pos, request);
	if (request.depth() <= 1) {
		auto result = Eval_dN(pos, request);
		// log.Add("depth == 0", result);
		return result;
	}
	if (pos.EmptyCount() <= PVS_to_AlphaBetaFailSoft && request.depth() == pos.EmptyCount()) {
		auto result = AlphaBetaFailSoft::Eval(pos, request);
		// log.Add("low empty count", result);
		return result;
	}

	node_count++;
	Moves moves = PossibleMoves(pos);
	if (!moves) {
		const auto passed = PlayPass(pos);
		if (HasMoves(passed)) {
			auto result = -PVS_N(passed, -request);
			// log.Add(Field::pass, result);
			// log.Add("No more moves", result);
			return result;
		}
		auto result = Result::Exact(pos, EvalGameOver(pos));
		// log.Add("Game Over", result);
		return result;
	}
	if (auto max = StabilityBasedMaxScore(pos); max < request) {
		auto result = Result::CertainFailLow(pos.EmptyCount(), max);
		// log.Add("Stability", result);
		return result;
	}

	Findings findings;
	if (auto look_up = tt.LookUp(pos); look_up.has_value()) {
		// log.Add(look_up.value());
		auto& result = look_up.value().result;
		if (result.intensity >= request.intensity) {
			if (result.window.IsSingleton() || !result.window.Overlaps(request.window)) {
				// log.Add("TT", result);
				return result;
			}
			findings.best_score = result.window.lower();
			findings.lowest_intensity = result.intensity;
		}
		findings.best_move = look_up.value().best_move;
	}
	if (auto mpc = MPC(pos, request); mpc.has_value()) {
		// log.Add("Prob cut", mpc.value());
		return mpc.value();
	}
	// IID
	if (findings.best_move == Field::invalid) {
		int reduced_depth = (request.depth() == pos.EmptyCount()) ? request.depth() - 10 : request.depth() - 2;
		if (reduced_depth >= 3) {
			PVS_N(pos, {{reduced_depth, request.intensity.certainty}, OpenInterval::Whole()});
			if (auto look_up = tt.LookUp(pos); look_up.has_value())
				findings.best_move = look_up.value().best_move;
		}
	}

	bool first = true;
	SortedMoves sorted_moves(moves, [&](Field move) { return MoveOrderingScorer(pos, move, findings.best_move); });
	for (const auto& move : sorted_moves)
	{
		if (not first)
		{
			auto zws = NextZWS(request, findings);
			auto result = -ZWS_N(Play(pos, move.second), zws) + 1;
			// log.Add(move.second, result);
			if (result > request) { // Beta cut
				result = Result::FailHigh(result);
				tt.Update(pos, {result, move.second});
				// log.Add("Beta cut", result);
				return result;
			}
			if (not (result > -zws)) {
				findings.Add(result, move.second);
				continue;
			}
		}
		first = false;

		auto result = -PVS_N(Play(pos, move.second), NextFWS(request, findings)) + 1;
		// log.Add(move.second, result);
		findings.Add(result, move.second);
		if (result > request) { // Beta cut
			result = Result::FailHigh(result);
			tt.Update(pos, {result, move.second});
			// log.Add("Beta cut", result);
			return result;
		}
		findings.Add(result, move.second);
	}
	const auto result = AllMovesSearched(request, findings);
	tt.Update(pos, {result, findings.best_move});
	// log.Add("No more moves", result);
	return result;
}

Result PVS::ZWS_N(const Position& pos, const Request& request)
{
	// log.AddSearch("ZWS", pos, request);
	if (request.depth() <= 1) {
		auto result = Eval_dN(pos, request);
		// log.Add("depth == 0", result);
		return result;
	}
	if (pos.EmptyCount() <= ZWS_to_AlphaBetaFailSoft && request.depth() == pos.EmptyCount()) {
		auto result = AlphaBetaFailSoft::Eval(pos, request);
		// log.Add("low empty count", result);
		return result;
	}

	node_count++;
	Moves moves = PossibleMoves(pos);
	if (!moves) {
		const auto passed = PlayPass(pos);
		if (HasMoves(passed)) {
			auto result = -PVS_N(passed, -request);
			// log.Add(Field::pass, result);
			// log.Add("No more moves", result);
			return result;
		}
		auto result = Result::Exact(pos, EvalGameOver(pos));
		// log.Add("Game Over", result);
		return result;
	}
	if (auto max = StabilityBasedMaxScore(pos); max < request) {
		auto result = Result::CertainFailLow(pos.EmptyCount(), max);
		// log.Add("Stability", result);
		return result;
	}

	Findings findings;
	if (auto look_up = tt.LookUp(pos); look_up.has_value()) {
		// log.Add(look_up.value());
		auto& result = look_up.value().result;
		if (result.intensity >= request.intensity) {
			if (result.window.IsSingleton() || !result.window.Overlaps(request.window)) {
				// log.Add("TT", result);
				return result;
			}
			findings.best_score = result.window.lower();
			findings.lowest_intensity = result.intensity;
		}
		findings.best_move = look_up.value().best_move;
	}
	if (auto mpc = MPC(pos, request); mpc.has_value()) {
		// log.Add("Prob cut", mpc.value());
		return mpc.value();
	}
	if (auto look_up = tt.LookUp(pos); look_up.has_value()) {
		// log.Add(look_up.value());
		auto& result = look_up.value().result;
		if (result.intensity >= request.intensity) {
			if (result.window.IsSingleton() || !result.window.Overlaps(request.window)) {
				// log.Add("TT", result);
				return result;
			}
			findings.best_score = result.window.lower();
			findings.lowest_intensity = result.intensity;
		}
		findings.best_move = look_up.value().best_move;
	}
	// ETC
	for (const auto& move : moves)
	{
		Position next = Play(pos, move);
		//if (auto max = StabilityBasedMaxScore(next); -max > request) {
		//	auto result = Result::CertainFailHigh(pos.EmptyCount(), -max);
		//	// log.Add("Stability", result);
		//	return result;
		//}
		if (auto look_up = tt.LookUp(next); look_up.has_value()) {
			// log.Add(look_up.value());
			auto result = -look_up.value().result + 1;
			if (result.intensity >= request.intensity) {
				if (result.window > request.window) {
					// log.Add("TT", result);
					return result;
				}
			}
		}
	}

	SortedMoves sorted_moves(moves, [&](Field move) 
							 {
								 int score = MoveOrderingScorer(pos, move, findings.best_move);
								 int sort_depth = std::clamp((request.intensity.depth + ((findings.best_move == Field::invalid) ? 0 : -2) - 15) / 3, 0, 6);
								 return score - (PVS_N(Play(pos, move), Request::Certain(sort_depth)).window.lower() << 16);
							 });
	//SortedMoves sorted_moves(moves, [&](Field move) { return MoveOrderingScorer(pos, move, findings.best_move); });
	for (const auto& move : sorted_moves)
	{
		auto result = -ZWS_N(Play(pos, move.second), NextZWS(request, findings)) + 1;
		// log.Add(move.second, result);
		if (result > request) { // Beta cut
			result = Result::FailHigh(result);
			tt.Update(pos, {result, move.second});
			// log.Add("Beta cut", result);
			return result;
		}
		findings.Add(result, move.second);
	}
	const auto result = AllMovesSearched(request, findings);
	tt.Update(pos, {result, findings.best_move});
	// log.Add("No more moves", result);
	return result;
}

int PVS::Eval_d0(const Position& pos)
{
	node_count++;
	int unbound_score = static_cast<int>(std::round(evaluator.Eval(pos)));
	return std::clamp(unbound_score, min_score, max_score);
}

int PVS::Eval_d1(const Position& pos, const OpenInterval& w)
{
	node_count++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		const auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_d1(passed, -w);
		return EvalGameOver(pos);
	}

	int best_score = -inf_score;
	for (const auto& move : moves)
	{
		const auto score = -Eval_d0(Play(pos, move));
		if (score > w)
			return score;
		best_score = std::max(best_score, score);
	}
	return best_score;
}

Result PVS::Eval_dN(const Position& pos, const Request& request)
{
	int score;
	if (request.depth() == 0)
		score = Eval_d0(pos);
	else
		score = Eval_d1(pos, request);
	return Result::CertainFailSoft(request, request.depth(), score);
}
