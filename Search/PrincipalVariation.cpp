#include "Core/Core.h"
#include "Algorithm.h"
#include "SortedMoves.h"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace Search;

int PVS::Eval(const Position& pos, const Intensity& intensity, const OpenInterval& window)
{
	return PVS_N(pos, intensity, window).Score();
}

int PVS::Eval(const Position& pos, const Intensity& intensity)
{
	return PVS_N(pos, intensity, OpenInterval::Whole()).Score();
}

int PVS::Eval(const Position& pos, const OpenInterval& window)
{
	return PVS_N(pos, Intensity::Exact(pos), window).Score();
}

int PVS::Eval(const Position& pos)
{
	return PVS_N(pos, Intensity::Exact(pos), OpenInterval::Whole()).Score();
}

uint64_t aaa(const Position& pos) noexcept
{
	auto b = pos.Empties();
	b |= ((b >> 1) & 0x7F7F7F7F7F7F7F7Fui64) | ((b << 1) & 0xFEFEFEFEFEFEFEFEui64);
	b |= (b >> 8) | (b << 8);
	return b & pos.Opponent();
}
//static const int8_t FieldValue[64] = {
//	9, 2, 8, 6, 6, 8, 2, 9,
//	2, 1, 3, 4, 4, 3, 1, 2,
//	8, 3, 7, 5, 5, 7, 3, 8,
//	6, 4, 5, 0, 0, 5, 4, 6,
//	6, 4, 5, 0, 0, 5, 4, 6,
//	8, 3, 7, 5, 5, 7, 3, 8,
//	2, 1, 3, 4, 4, 3, 1, 2,
//	9, 2, 8, 6, 6, 8, 2, 9,
//};

int32_t PVS::MoveOrderingScorer(const Position& pos, Field move, Field best_move, Field best_move_2, int alpha, int depth) noexcept
{
	if (move == best_move)
		return 1 << 29;
	if (move == best_move_2)
		return 1 << 28;

	Position next = Play(pos, move);

	int score = 0;// FieldValue[static_cast<uint8_t>(move)];
	//if (pos.ParityQuadrants() & BitBoard(move)) {
	//	if (pos.EmptyCount() < 12) score += 1 << 2;
	//	else if (pos.EmptyCount() < 21) score += 1 << 1;
	//	else if (pos.EmptyCount() < 30) score += 1;
	//}
	score -= DoubleCornerPopcount(PotentialMoves(next)) << 5; // w_potential_mobility
	score -= popcount(aaa(next)) << 6; // w_potential_mobility
	score -= DoubleCornerPopcount(PossibleMoves(next)) << 15; // w_mobility
	score += popcount(StableEdges(next) & next.Opponent()) << 11; // w_edge_stability

	//int sort_depth;
	//int min_depth = 9;
	//if (pos.EmptyCount() <= 27) min_depth += (30 - pos.EmptyCount()) / 3;
	if (depth >= 9)
	{
		score -= PVS_shallow(next, Intensity(std::clamp((depth - 15) / 3, 0, 6))).Score() << 15; // w_eval

		//sort_depth = (depth - 15) / 3;
		////if (pos.EmptyCount() >= 27) ++sort_depth;
		//sort_depth = std::clamp(sort_depth, 0, 6);
		////int sort_alpha = std::max(min_score, alpha - 8);
		////auto r = Request(Intensity::Certain(sort_depth), { min_score, -sort_alpha });
		//auto r = Request::Certain(sort_depth);
		//switch (sort_depth)
		//{
		//case 0:
		//	score -= PVS_shallow(next, r).window.lower() << 15; // w_eval
		//	break;
		//case 1:
		//case 2:
		//	score -= PVS_shallow(next, r).window.lower() << 15; // w_eval
		//	break;
		//default:
		//	//if (auto look_up = tt.LookUp(next); look_up.has_value())
		//	//	score += 1 << 15;
		//	score -= PVS_shallow(next, r).window.lower() << 15; // w_eval
		//	break;
		//}
	}
	return score;
}

float Sigma(int D, int d, int E) noexcept
{
	const auto [alpha, beta, gamma, delta, epsilon] = std::make_tuple(-0.177498, 0.938986, 0.261467, -0.0096483, 1.12807); // R^2 = 0.925636
	return (std::expf(alpha * d) + beta) * std::powf(D - d, gamma) * (delta * E + epsilon);

	//static const auto [alpha, beta, gamma, delta, epsilon] = std::make_tuple(-0.0090121, 0.0348359, -0.0836982, 2.59089, 2.51293); // R^2 = 0.864748
	//float sigma = alpha * E + beta * D + gamma * d;
	//sigma = sigma * sigma + delta * sigma + epsilon;
	//return sigma;

	//const double EVAL_A = -0.10026799;
	//const double EVAL_B = 0.31027733;
	//const double EVAL_C = -0.57772603;
	//const double EVAL_a = 0.07585621;
	//const double EVAL_b = 1.16492647;
	//const double EVAL_c = 5.4171698;
	//double sigma = EVAL_A * E + EVAL_B * D + EVAL_C * d;
	//sigma = EVAL_a * sigma * sigma + EVAL_b * sigma + EVAL_c;
	//return sigma / 2.0;
}

std::optional<Result> PVS::MPC(const Position& pos, const Intensity& intensity, const OpenInterval& window)
{
	//static int probcut_level = 0;
	if (intensity.IsCertain() /*|| probcut_level >= 2*/)
		return std::nullopt;

	// log.AddSearch("MPC", pos, request);
	float t = intensity.certainty.sigmas();
	float sigma_0 = Sigma(intensity.depth, 0, pos.EmptyCount());
	float eval_0 = evaluator.Eval(pos);
	//int upper_0 = static_cast<int>(std::ceil(request.window.upper() + 2 * sigma_0 * t));
	//int lower_0 = static_cast<int>(std::floor(request.window.lower() - 2 * sigma_0 * t));
	//if (eval_0 >= upper_0) {
	//	auto result = Result{ intensity, {request.window.upper(), max_score} };
	//	// log.Add("Upper cut", result);
	//	return result;
	//}
	//if (eval_0 <= lower_0) {
	//	auto result = Result{ intensity, {min_score, request.window.lower()} };
	//	// log.Add("Lower cut", result);
	//	return result;
	//}

	//probcut_level++;
	{
		int reduced_depth = intensity.depth / 4 * 2 /*+ (intensity.depth & 1)*/;
		//if (reduced_depth == 0)
		//	reduced_depth = intensity.depth - 2;

		float sigma = Sigma(intensity.depth, reduced_depth, pos.EmptyCount());
		int probcut_error = t * sigma + 0.5;
		int eval_error = sigma_0;// t * 0.5 * (sigma_0 + sigma) + 0.5;
		//int upper = static_cast<int>(std::ceil(request.window.upper() + sigma * t));
		//int lower = static_cast<int>(std::floor(request.window.lower() - sigma * t));

		int alpha = window.lower();
		int beta = alpha + 1;
		int eval_beta = beta - eval_error;
		int probcut_beta = beta + probcut_error;
		Confidence cl = Confidence::Certain();
		if (intensity.certainty == 1.1_sigmas)
			cl = Confidence(1.5);
		if (intensity.certainty == 1.5_sigmas)
			cl = Confidence(2.0);
		if (intensity.certainty == 2.0_sigmas)
			cl = Confidence(2.6);
		if (eval_0 >= eval_beta && probcut_beta < max_score) {
			OpenInterval upper_zws(probcut_beta - 1, probcut_beta);
			Result shallow_result = ZWS_N(pos, { reduced_depth, cl }, upper_zws, true);
			if (shallow_result.window > upper_zws) { // Fail high
				//float sigmas = (shallow_result.window.lower() - request.window.upper()) / Sigma(intensity.depth, shallow_result.depth(), pos.EmptyCount());
				//assert(sigmas > t);
				auto result = Result::FailHigh(intensity, window.upper());
				// log.Add("Upper cut", result);
				//probcut_level--;
				return result;
			}
		}

		int eval_alpha = alpha + eval_error;
		int probcut_alpha = alpha - probcut_error;
		if (eval_0 < eval_alpha && probcut_alpha > min_score) {
			OpenInterval lower_zws(probcut_alpha, probcut_alpha + 1);
			Result shallow_result = ZWS_N(pos, { reduced_depth, cl }, lower_zws, false);
			if (shallow_result.window < lower_zws) { // Fail low
				//float sigmas = (request.window.lower() - shallow_result.window.upper()) / Sigma(intensity.depth, shallow_result.depth(), pos.EmptyCount());
				//assert(sigmas > t);
				auto result = Result::FailLow(intensity, window.lower());
				// log.Add("Lower cut", result);
				//probcut_level--;
				return result;
			}
		}
	}
	// log.FinalizeSearch();
	//probcut_level--;
	return std::nullopt;
}

Result PVS::PVS_N(const Position& pos, const Intensity& intensity, const OpenInterval& window)
{
	//static int pv_ext = 12;
	//if (pos.EmptyCount() <= pv_ext && intensity.depth < pos.EmptyCount())
	//	return PVS_N(pos, Request(pos.EmptyCount(), request.certainty(), request.window));
	// log.AddSearch("PVS", pos, request);
	if (intensity.depth <= 2) {
		auto result = Eval_dN(pos, intensity, window);
		// log.Add("depth == 0", result);
		return result;
	}
	if (pos.EmptyCount() <= 7 && intensity.depth == pos.EmptyCount()) {
		int score = AlphaBetaFailSoft::Eval(pos, intensity, window);
		// log.Add("low empty count", score);
		return Result(intensity, score);
	}

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves) {
		auto passed = PlayPass(pos);
		if (HasMoves(passed)) {
			auto result = -PVS_N(passed, intensity, -window);
			// log.Add(Field::pass, result);
			// log.Add("No more moves", result);
			return result;
		}
		auto result = Result::Exact(pos, EvalGameOver(pos));
		// log.Add("Game Over", result);
		return result;
	}
	//if (auto max = StabilityBasedMaxScore(pos); max < request) {
	//	auto result = Result::FailLow(pos.EmptyCount(), max);
	//	// log.Add("Stability", result);
	//	return result;
	//}

	Findings findings;
	if (auto look_up = tt.LookUp(pos); look_up.has_value()) {
		// log.Add(look_up.value());
		auto& result = look_up.value().result;
		//if (result.intensity >= intensity) {
		////	if (result.window.IsSingleton() || !result.window.Overlaps(request.window)) {
		////		// log.Add("TT", result);
		////		return result;
		////	}
		//	findings.best_score = result.Score();
		//	findings.lowest_intensity = result.intensity;
		//}
		findings.best_move = look_up.value().best_move;
		findings.best_move_2 = look_up.value().best_move_2;
	}
	//if (auto mpc = MPC(pos, request); mpc.has_value()) {
	//	// log.Add("Prob cut", mpc.value());
	//	return mpc.value();
	//}
	// IID
	//if (findings.best_move == Field::invalid && moves.size() > 1) {
	//	int reduced_depth = (intensity.depth == pos.EmptyCount()) ? intensity.depth - 10 : intensity.depth - 2;
	//	if (reduced_depth >= 3) {
	//		//int tmp = pv_ext;
	//		//pv_ext = 0;
	//		PVS_N(pos, { {reduced_depth, Confidence(1.1)}, OpenInterval::Whole() });
	//		//pv_ext = tmp;
	//		if (auto look_up = tt.LookUp(pos); look_up.has_value()) {
	//			findings.best_move = look_up.value().best_move;
	//			findings.best_move_2 = look_up.value().best_move_2;
	//		}
	//	}
	//}

	bool first = true;
	SortedMoves sorted_moves(moves, [&](Field move)
		{ return MoveOrderingScorer(pos, move, findings.best_move, findings.best_move_2, window.lower(), intensity.depth); });
	for (const auto& move : sorted_moves)
	{
		if (not first)
		{
			auto zws = NextZeroWindow(window, findings.best_score);
			auto result = -ZWS_N(Play(pos, move.second), intensity - 1, zws, true) + 1;
			// log.Add(move.second, result);
			if (result.window > window) { // Beta cut
				result.FailedHigh();
				tt.Update(pos, { result, move.second });
				// log.Add("Beta cut", result);
				return result;
			}
			if (not (result.window > -zws)) {
				findings.Add(result, move.second);
				continue;
			}
		}
		first = false;

		auto result = -PVS_N(Play(pos, move.second), intensity - 1, NextFullWindow(window, findings.best_score)) + 1;
		// log.Add(move.second, result);
		findings.Add(result, move.second);
		if (result.window > window) { // Beta cut
			result.FailedHigh();
			tt.Update(pos, { result, move.second });
			// log.Add("Beta cut", result);
			return result;
		}
		findings.Add(result, move.second);
	}
	const auto result = AllMovesSearched(window, findings);
	tt.Update(pos, {result, findings.best_move});
	// log.Add("No more moves", result);
	return result;
}

Result PVS::ZWS_N(const Position& pos, const Intensity& intensity, const OpenInterval& window, bool cut_node)
{
	// log.AddSearch("ZWS", pos, request);
	if (intensity.depth <= 3 && intensity.depth < pos.EmptyCount()) {
		auto result = ZWS_shallow(pos, intensity, window);
		// log.Add("depth == 0", result);
		return result;
	}
	if (pos.EmptyCount() < 15 && intensity.depth == pos.EmptyCount()) {
		auto result = ZWS_endgame(pos, intensity, window);
		// log.Add("low empty count", result);
		return result;
	}

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves) {
		auto passed = PlayPass(pos);
		if (HasMoves(passed)) {
			auto result = -ZWS_N(passed, intensity, -window, !cut_node);
			// log.Add(Field::pass, result);
			// log.Add("No more moves", result);
			return result;
		}
		auto result = Result::Exact(pos, EvalGameOver(pos));
		// log.Add("Game Over", result);
		return result;
	}
	if (auto max = StabilityBasedMaxScore(pos); max < window) {
		auto result = Result::FailLow(pos.EmptyCount(), max);
		// log.Add("Stability", result);
		return result;
	}

	Findings findings;
	if (auto look_up = tt.LookUp(pos); look_up.has_value()) {
		// log.Add(look_up.value());
		auto& result = look_up.value().result;
		if (result.intensity >= intensity) {
			if (result.window.IsSingleton() || !result.window.Overlaps(window)) {
				// log.Add("TT", result);
				return result;
			}
			//findings.best_score = result.Score();
			//findings.lowest_intensity = result.intensity;
		}
		findings.best_move = look_up.value().best_move;
		findings.best_move_2 = look_up.value().best_move_2;
	}
	if (auto mpc = MPC(pos, intensity, window); mpc.has_value()) {
		// log.Add("Prob cut", mpc.value());
		return mpc.value();
	}

	SortedMoves sorted_moves(moves, [&](Field move)
		{ return MoveOrderingScorer(pos, move, findings.best_move, findings.best_move_2, window.lower(), intensity.depth - 2); });

	// ETC
	//if (intensity.depth > 5)
	{
		for (Field move : moves)
		{
			Position next = Play(pos, move);
			if (auto max = StabilityBasedMaxScore(next); -max > window) {
				auto result = Result::FailHigh(pos.EmptyCount(), -max);
				// log.Add("Stability", result);
				return result;
			}
			if (auto look_up = tt.LookUp(next); look_up.has_value()) {
				// log.Add(look_up.value());
				auto result = -look_up.value().result + 1;
				if (result.intensity >= intensity) {
					if (result.window > window) {
						// log.Add("TT", result);
						return result;
					}
				}
			}
		}
	}
	for (const auto& move : sorted_moves)
	{
		auto result = -ZWS_N(Play(pos, move.second), intensity - 1, NextZeroWindow(window, findings.best_score), !cut_node) + 1;
		// log.Add(move.second, result);
		if (result.window > window) { // Beta cut
			result.FailedHigh();
			tt.Update(pos, { result, move.second });
			// log.Add("Beta cut", result);
			return result;
		}
		findings.Add(result, move.second);
	}
	const auto result = AllMovesSearched(window, findings);
	tt.Update(pos, {result, findings.best_move});
	// log.Add("No more moves", result);
	return result;
}

Result PVS::PVS_shallow(const Position& pos, const Intensity& intensity, const OpenInterval& window)
{
	// log.AddSearch("PVS", pos, request);
	if (intensity.depth <= 2) {
		auto result = Eval_dN(pos, intensity, window);
		// log.Add("depth == 0", result);
		return result;
	}

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves) {
		auto passed = PlayPass(pos);
		if (HasMoves(passed)) {
			auto result = -PVS_shallow(passed, intensity, -window);
			// log.Add(Field::pass, result);
			// log.Add("No more moves", result);
			return result;
		}
		auto result = Result::Exact(pos, EvalGameOver(pos));
		// log.Add("Game Over", result);
		return result;
	}
	if (auto max = StabilityBasedMaxScore(pos)) {
		if (max < window)
		{
			auto result = Result::FailLow(pos.EmptyCount(), max);
			// log.Add("Stability", result);
			return result;
		}
		//window.TryDecreaseUpper(max + 1); // TODO: Make this work again!
	}

	Findings findings;
	if (auto look_up = tt.LookUp(pos); look_up.has_value()) {
		// log.Add(look_up.value());
		auto& result = look_up.value().result;
		if (result.intensity >= intensity) {
			//if (result.window.IsSingleton() || !result.window.Overlaps(request.window)) {
			//	// log.Add("TT", result);
			//	return result;
			//}
			//findings.best_score = result.Score();
			//findings.lowest_intensity = result.intensity;
		}
		findings.best_move = look_up.value().best_move;
		findings.best_move_2 = look_up.value().best_move_2;
	}

	bool first = true;
	SortedMoves sorted_moves(moves, [&](Field move)
		{ return MoveOrderingScorer(pos, move, findings.best_move, findings.best_move_2, window.lower(), intensity.depth); });
	for (const auto& move : sorted_moves)
	{
		if (not first)
		{
			auto zws = NextZeroWindow(window, findings.best_score);
			auto result = -ZWS_shallow(Play(pos, move.second), intensity -1, zws) + 1;
			// log.Add(move.second, result);
			if (result.window > window) { // Beta cut
				result.FailedHigh();
				tt.Update(pos, { result, move.second });
				// log.Add("Beta cut", result);
				return result;
			}
			if (not (result.window > -zws)) {
				findings.Add(result, move.second);
				continue;
			}
		}
		first = false;

		auto result = -PVS_shallow(Play(pos, move.second), intensity - 1, NextFullWindow(window, findings.best_score)) + 1;
		// log.Add(move.second, result);
		findings.Add(result, move.second);
		if (result.window > window) { // Beta cut
			result.FailedHigh();
			tt.Update(pos, { result, move.second });
			// log.Add("Beta cut", result);
			return result;
		}
		findings.Add(result, move.second);
	}
	const auto result = AllMovesSearched(window, findings);
	tt.Update(pos, { result, findings.best_move });
	// log.Add("No more moves", result);
	return result;
}

Result PVS::ZWS_shallow(const Position& pos, const Intensity& intensity, const OpenInterval& window)
{
	// log.AddSearch("ZWS", pos, request);
	if (intensity.depth <= 2) {
		auto result = Eval_dN(pos, intensity, window);
		// log.Add("depth == 0", result);
		return result;
	}

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves) {
		auto passed = PlayPass(pos);
		if (HasMoves(passed)) {
			auto result = -ZWS_shallow(passed, intensity, -window);
			// log.Add(Field::pass, result);
			// log.Add("No more moves", result);
			return result;
		}
		auto result = Result::Exact(pos, EvalGameOver(pos));
		// log.Add("Game Over", result);
		return result;
	}
	if (auto max = StabilityBasedMaxScore(pos); max < window) {
		auto result = Result::FailLow(pos.EmptyCount(), max);
		// log.Add("Stability", result);
		return result;
	}

	Findings findings;
	if (auto look_up = tt.LookUp(pos); look_up.has_value()) {
		// log.Add(look_up.value());
		auto& result = look_up.value().result;
		if (result.intensity >= intensity) {
			if (result.window.IsSingleton() || !result.window.Overlaps(window)) {
				// log.Add("TT", result);
				return result;
			}
			//findings.best_score = result.Score();
			//findings.lowest_intensity = result.intensity;
		}
		findings.best_move = look_up.value().best_move;
		findings.best_move_2 = look_up.value().best_move_2;
	}

	SortedMoves sorted_moves(moves, [&](Field move)
		{ return MoveOrderingScorer(pos, move, findings.best_move, findings.best_move_2, window.lower(), intensity.depth); });
	for (const auto& move : sorted_moves)
	{
		auto result = -ZWS_shallow(Play(pos, move.second), intensity - 1, NextZeroWindow(window, findings.best_score)) + 1;
		// log.Add(move.second, result);
		if (result.window > window) { // Beta cut
			result.FailedHigh();
			tt.Update(pos, { result, move.second });
			// log.Add("Beta cut", result);
			return result;
		}
		findings.Add(result, move.second);
	}
	const auto result = AllMovesSearched(window, findings);
	tt.Update(pos, { result, findings.best_move });
	// log.Add("No more moves", result);
	return result;
}

// Stability, TT, MoveOrderingScorer
Search::Result PVS::ZWS_endgame(const Position& pos, const Intensity& intensity, const OpenInterval& window)
{
	if (pos.EmptyCount() <= 8) {
		auto score = AlphaBetaFailSoft::Eval(pos, intensity, window);
		// log.Add("low empty count", score);
		return Search::Result::Exact(pos, score);
	}

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves) {
		auto passed = PlayPass(pos);
		if (HasMoves(passed)) {
			auto result = -ZWS_endgame(passed, intensity, -window);
			// log.Add(Field::pass, result);
			// log.Add("No more moves", result);
			return result;
		}
		auto result = Result::Exact(pos, EvalGameOver(pos));
		// log.Add("Game Over", result);
		return result;
	}
	if (auto max = StabilityBasedMaxScore(pos); max < window) {
		auto result = Result::FailLow(pos, max);
		// log.Add("Stability", result);
		return result;
	}

	Findings findings;
	if (auto look_up = tt.LookUp(pos); look_up.has_value()) {
		// log.Add(look_up.value());
		auto& result = look_up.value().result;
		if (result.intensity >= intensity) {
			if (result.window.IsSingleton() || !result.window.Overlaps(window)) {
				// log.Add("TT", result);
				return result;
			}
			//findings.best_score = result.Score();
			//findings.lowest_intensity = result.intensity;
		}
		findings.best_move = look_up.value().best_move;
		findings.best_move_2 = look_up.value().best_move_2;
	}

	SortedMoves sorted_moves(moves, [&](Field move)
		{ return MoveOrderingScorer(pos, move, findings.best_move, findings.best_move_2, window.lower(), 0); });
	for (const auto& move : sorted_moves)
	{
		auto result = -ZWS_endgame(Play(pos, move.second), intensity - 1, NextZeroWindow(window, findings.best_score)) + 1;
		// log.Add(move.second, result);
		if (result.window > window) { // Beta cut
			result.FailedHigh();
			tt.Update(pos, { result, move.second });
			// log.Add("Beta cut", result);
			return result;
		}
		findings.Add(result, move.second);
	}
	const auto result = AllMovesSearched(window, findings);
	tt.Update(pos, { result, findings.best_move });
	// log.Add("No more moves", result);
	return result;
}

Result PVS::Eval_dN(const Position& pos, const Intensity& intensity, const OpenInterval& window)
{
	int score = Eval_dN(pos, intensity.depth, window);
	if (score > window)
		return Result::FailHigh({ intensity.depth }, score);
	return Result::FailLow({ intensity.depth }, score);
}

int PVS::Eval_dN(const Position& pos, int depth, OpenInterval window)
{
	if (depth == 0)
		return Eval_d0(pos);

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_dN(passed, depth, -window);
		return EvalGameOver(pos);
	}

	int best_score = -inf_score;
	for (Field move : moves)
	{
		const auto score = -Eval_dN(Play(pos, move), depth - 1, -window);
		if (score > window)
			return score;
		window.TryIncreaseLower(score);
		if (score > best_score)
			best_score = score;
	}
	return best_score;
}

int PVS::Eval_d0(const Position& pos)
{
	nodes++;
	int unbound_score = static_cast<int>(std::round(evaluator.Eval(pos)));
	return std::clamp(unbound_score, min_score, max_score);
}