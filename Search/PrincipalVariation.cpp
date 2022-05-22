#include "PrincipalVariation.h"
#include "Stability.h"

int to_score(float value)
{
	return std::clamp(static_cast<int>(std::round(value)), min_score, max_score);
}

int PVS::Eval(const Position& pos, Intensity intensity, OpenInterval window)
{
	nodes = 0;
	return PVS_N(pos, intensity, window);
}

ScoreMove PVS::Eval_BestMove(const Position& pos, Intensity intensity, OpenInterval window)
{
	nodes = 0;
	return Eval_BestMove_N(pos, intensity, window);
}

void PVS::clear()
{
	tt.clear();
}

ScoreMove PVS::Eval_BestMove_N(const Position& pos, Intensity intensity, OpenInterval window)
{
	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_BestMove_N(passed, intensity, -window);
		return EvalGameOver(pos);
	}

	ScoreMove best;
	for (Field move : SortMoves(moves, pos))
	{
		int score = -PVS_N(Play(pos, move), intensity - 1, -window);
		if (score > window)
			return { score, move };
		best.ImproveWith(score, move);
		window.TryIncreaseLower(score);
	}
	return best;
}

IntensityScore PVS::PVS_N(const Position& pos, Intensity intensity, const OpenInterval& window)
{
	if (pos.EmptyCount() <= 7 and intensity.depth >= pos.EmptyCount())
		return AlphaBetaFailSuperSoft::Eval_N(pos, window);
	if (intensity.depth <= 2)
		return Eval_dN(pos, intensity, window);
	//if (pos.EmptyCount() <= 14 and intensity.depth < pos.EmptyCount())
	//	return PVS_N(pos, pos.EmptyCount(), window);

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -PVS_N(passed, intensity, -window);
		return EvalGameOver(pos);
	}

	Findings findings(intensity, window);
	if (auto look_up = tt.LookUp(pos); look_up.has_value())
		if (findings.Add(look_up.value()))
			return findings;

	Finally finally([&]() { tt.Update(pos, findings); });
	bool first = true;
	if (findings.move == Field::invalid and intensity.depth > 16)
		findings.move = Eval_BestMove_N(pos, intensity.depth / 4, { -inf_score, +inf_score }).move;
	if (findings.move != Field::invalid)
	{
		auto ret = -PVS_N(Play(pos, findings.move), intensity - 1, -findings.NextFullWindow());
		if (findings.Add(ret, findings.move))
			return findings;
		first = false;
		moves.erase(findings.move);
	}

	for (Field move : SortMoves(moves, pos))
	{
		if (not first)
		{
			auto zero_window = findings.NextZeroWindow();
			auto ret = -ZWS_N(Play(pos, move), intensity - 1, -zero_window);
			if (findings.Add(ret, move))
				return findings;
			if (ret < zero_window)
				continue;
		}
		first = false;

		auto ret = -PVS_N(Play(pos, move), intensity - 1, -findings.NextFullWindow());
		if (findings.Add(ret, move))
			return findings;
	}
	return findings;
}

IntensityScore PVS::ZWS_N(const Position& pos, Intensity intensity, const OpenInterval& window)
{
	if (pos.EmptyCount() <= 7 and intensity.depth >= pos.EmptyCount())
		return AlphaBetaFailSuperSoft::Eval_N(pos, window);
	if (intensity.depth <= 2)
		return Eval_dN(pos, intensity, window);

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -ZWS_N(passed, intensity, -window);
		return EvalGameOver(pos);
	}

	if (auto max = StabilityBasedMaxScore(pos); max < window)
		return max;

	Findings findings(intensity, window);
	if (auto look_up = tt.LookUp(pos); look_up.has_value())
		if (findings.Add(look_up.value()))
			return findings;
	if (auto mpc = MPC(pos, intensity, window); mpc.has_value())
		return mpc.value();
	if (findings.move == Field::invalid)
		if (auto look_up = tt.LookUp(pos); look_up.has_value())
			if (findings.Add(look_up.value()))
				return findings;

	Finally finally([&]() { tt.Update(pos, findings); });
	if (findings.move == Field::invalid and intensity.depth > 16)
		findings.move = Eval_BestMove_N(pos, intensity.depth / 4, { -inf_score, +inf_score }).move;
	if (findings.move != Field::invalid)
	{
		auto ret = -ZWS_N(Play(pos, findings.move), intensity - 1, -findings.NextFullWindow());
		if (findings.Add(ret, findings.move))
			return findings;
		moves.erase(findings.move);
	}

	for (Field move : SortMoves(moves, pos))
	{
		auto ret = -ZWS_N(Play(pos, move), intensity - 1, -findings.NextFullWindow());
		if (findings.Add(ret, move))
			return findings;
	}
	return findings;
}

std::optional<IntensityScore> PVS::MPC(const Position& pos, Intensity intensity, const OpenInterval& window)
{
	if (intensity.IsCertain())
		return std::nullopt;

	int D = intensity.depth;
	int d = D / 2;
	int E = pos.EmptyCount();

	float t = intensity.certainty.sigmas();

	float eval_0 = evaluator.Eval(pos);
	float sd_0 = evaluator.Accuracy(D, 0, E);

	if (eval_0 >= window.Upper() + t * sd_0) // fail high
		return IntensityScore{ intensity, window.Upper() };
	if (eval_0 <= window.Lower() - t * sd_0) // fail low
		return IntensityScore{ intensity, window.Lower() };

	// confidence of reduced search
	Confidence c = Confidence::Certain();
	if (intensity.certainty == 1.1_sigmas)
		c = 1.5_sigmas;
	if (intensity.certainty == 1.5_sigmas)
		c = 2.0_sigmas;
	if (intensity.certainty == 1.5_sigmas)
		c = 2.6_sigmas;

	float sd_d = evaluator.Accuracy(D, d, E);
	int lower_limit = std::round(window.Lower() - t * sd_d);
	int upper_limit = std::round(window.Upper() + t * sd_d);

	if (eval_0 >= upper_limit and upper_limit <= max_score)
	{
		int shallow_result = ZWS_N(pos, { d, c }, { upper_limit - 1, upper_limit });
		if (shallow_result >= upper_limit) // fail high
			return IntensityScore{ intensity, window.Upper() };
	}

	if (eval_0 <= lower_limit and lower_limit >= min_score)
	{
		int shallow_result = ZWS_N(pos, { d, c }, { lower_limit, lower_limit + 1 });
		if (shallow_result <= lower_limit) // fail low
			return IntensityScore{ intensity, window.Lower() };
	}

	return std::nullopt;
}

IntensityScore PVS::Eval_dN(const Position& pos, Intensity intensity, OpenInterval window)
{
	nodes++;
	if (intensity.depth == 0)
		return Eval_d0(pos);

	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_dN(passed, intensity, -window);
		return EvalGameOver(pos);
	}
	int best_score = -inf_score;
	for (Field move : moves)
	{
		int score = -Eval_dN(Play(pos, move), intensity - 1, -window);
		if (score > window)
			return { intensity.depth, score };
		best_score = std::max(best_score, score);
		window.TryIncreaseLower(score);
	}
	return { intensity.depth, best_score };
}

IntensityScore PVS::Eval_d0(const Position& pos)
{
	nodes++;
	return { /*depth*/ 0, to_score(evaluator.Eval(pos)) };
}

SortedMoves PVS::SortMoves(Moves moves, const Position& pos)
{
	return SortedMoves(moves, [pos](Field move)
		{
			Position next = Play(pos, move);
			auto P = next.Player();
			auto O = next.Opponent();
			auto E = next.Empties();
			auto P9 = EightNeighboursAndSelf(P);
			auto O9 = EightNeighboursAndSelf(O);
			auto E9 = EightNeighboursAndSelf(E);
			auto pm = PossibleMoves(next);
			auto se = StableEdges(next);

			uint32_t score = 0;
			score += popcount(O & E9) << 8;
			score += popcount(E & O9) << 9;
			score += (64 - popcount(se & O)) << 14;
			score += (pm.size() + (pm & BitBoard::Corners()).size()) << 18;
			return score;
		});
}
