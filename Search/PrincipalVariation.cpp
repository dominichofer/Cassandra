#include "PrincipalVariation.h"
#include "Stability.h"

ContextualResult PVS::Eval(const Position& pos, Intensity intensity, OpenInterval window)
{
	nodes = 0;
	return PVS_N(pos, intensity, window);
}

void PVS::clear()
{
	tt.clear();
}

//ScoreMove PVS::Eval_BestMove_N(const Position& pos, Intensity intensity, OpenInterval window)
//{
//	nodes++;
//	Moves moves = PossibleMoves(pos);
//	if (!moves)
//	{
//		auto passed = PlayPass(pos);
//		if (HasMoves(passed))
//			return -Eval_BestMove_N(passed, intensity, -window);
//		return EvalGameOver(pos);
//	}
//
//	ScoreMove best;
//	for (Field move : SortMoves(moves, pos, intensity.depth))
//	{
//		int score = -PVS_N(Play(pos, move), intensity - 1, -window);
//		if (score > window)
//			return { score, move };
//		best.ImproveWith(score, move);
//		window.TryIncreaseLower(score);
//	}
//	return best;
//}

ContextualResult PVS::PVS_N(const Position& pos, Intensity intensity, const OpenInterval& window)
{
	const bool endgame = (intensity.depth >= pos.EmptyCount());
	if (pos.EmptyCount() <= 7 and endgame)
		return AlphaBetaFailSuperSoft::Eval(pos, window);
	if (intensity.depth <= 2)
		return Eval_dN(pos, intensity, window);
	//if (pos.EmptyCount() <= 10 and not endgame)
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
	if (moves.size() == 1)
		return -PVS_N(Play(pos, moves.front()), intensity - 1, -window) + 1;

	Findings findings(intensity, window);
	if (auto look_up = tt.LookUp(pos); look_up.has_value())
		if (findings.Add(look_up.value()))
			return findings;

	Finally finally([&]() { tt.Update(pos, findings); });
	bool first = true;
	//if (findings.Move() == Field::invalid and intensity.depth > 14)
	//	findings.SetMove(PVS_N(pos, intensity.depth / 4, { -inf_score, +inf_score }).move);
	if (findings.Move() != Field::invalid)
	{
		auto ret = -PVS_N(Play(pos, findings.Move()), intensity - 1, -findings.NextFullWindow());
		if (findings.AddOption(ret, findings.Move()))
			return findings;
		first = false;
		moves.erase(findings.Move());
	}

	for (Field move : SortMoves(moves, pos, intensity.depth))
	{
		if (not first)
		{
			auto zero_window = findings.NextZeroWindow();
			auto ret = -ZWS_N(Play(pos, move), intensity - 1, -zero_window);
			if (findings.AddOption(ret, move))
				return findings;
			if (ret < zero_window)
				continue;
		}
		first = false;

		auto ret = -PVS_N(Play(pos, move), intensity - 1, -findings.NextFullWindow());
		if (findings.AddOption(ret, move))
			return findings;
	}
	return findings;
}

ContextualResult PVS::ZWS_N(const Position& pos, Intensity intensity, const OpenInterval& window)
{
	const bool endgame = (intensity.depth >= pos.EmptyCount());
	if (pos.EmptyCount() <= 7 and endgame)
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
	if (moves.size() == 1)
		return -ZWS_N(Play(pos, moves.front()), intensity - 1, -window) + 1;

	if (auto max = StabilityBasedMaxScore(pos); max < window)
		return max;

	Findings findings(intensity, window);
	if (auto look_up = tt.LookUp(pos); look_up.has_value())
		if (findings.Add(look_up.value()))
			return findings;
	if (auto mpc = MPC(pos, intensity, window); mpc.has_value())
		return mpc.value();
	//if (findings.Move() == Field::invalid)
	//	if (auto look_up = tt.LookUp(pos); look_up.has_value())
	//		if (findings.Add(look_up.value()))
	//			return findings;
	
	// ETC
	//if (intensity.depth > 5 and intensity.depth < pos.EmptyCount())
	//{
	//	for (Field move : moves)
	//	{
	//		Position next = Play(pos, move);
	//		if (auto max = -StabilityBasedMaxScore(next); max > window)
	//			return max;
	//		if (auto look_up = tt.LookUp(next); look_up.has_value())
	//			if (look_up.value().intensity + 1 >= intensity and -look_up.value().window > window)
	//				return { look_up.value().intensity + 1, -look_up.value().window.Upper() };
	//	}
	//}

	Finally finally([&]() { tt.Update(pos, findings); });
	if (findings.Move() == Field::invalid and intensity.depth > 14)
		findings.SetMove(PVS_N(pos, intensity.depth / 4, { -inf_score, +inf_score }).move);
	if (findings.Move() != Field::invalid)
	{
		Field move = findings.Move();
		auto ret = -ZWS_N(Play(pos, move), intensity - 1, -findings.NextFullWindow());
		if (findings.AddOption(ret, move))
			return findings;
		moves.erase(findings.Move());
	}

	for (Field move : SortMoves(moves, pos, intensity.depth))
	{
		auto ret = -ZWS_N(Play(pos, move), intensity - 1, -findings.NextFullWindow());
		if (findings.AddOption(ret, move))
			return findings;
	}
	return findings;
}

std::optional<ContextualResult> PVS::MPC(const Position& pos, Intensity intensity, const OpenInterval& window)
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
		return ContextualResult{ intensity, window.Upper() };
	if (eval_0 <= window.Lower() - t * sd_0) // fail low
		return ContextualResult{ intensity, window.Lower() };

	// confidence of reduced search
	Confidence c = Confidence::Certain();
	if (intensity.certainty == 0.8_sigmas)
		c = 1.1_sigmas;
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
			return ContextualResult{ intensity, window.Upper() };
	}

	if (eval_0 <= lower_limit and lower_limit >= min_score)
	{
		int shallow_result = ZWS_N(pos, { d, c }, { lower_limit, lower_limit + 1 });
		if (shallow_result <= lower_limit) // fail low
			return ContextualResult{ intensity, window.Lower() };
	}

	return std::nullopt;
}

ContextualResult PVS::Eval_dN(const Position& pos, Intensity intensity, OpenInterval window)
{
	if (intensity.depth == 0)
		return Eval_d0(pos);

	nodes++;
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

int to_score(float value)
{
	return std::clamp(static_cast<int>(std::round(value)), min_score, max_score);
}

ContextualResult PVS::Eval_d0(const Position& pos)
{
	nodes++;
	return { /*depth*/ 0, to_score(evaluator.Eval(pos)) };
}

SortedMoves PVS::SortMoves(Moves moves, const Position& pos, int depth)
{
	//int sort_depth;
	//int min_depth = 9;
	//if (pos.EmptyCount() <= 27)
	//	min_depth += (30 - pos.EmptyCount()) / 3;
	//if (depth >= min_depth)
	//{
	//	sort_depth = (depth - 15) / 3;
	//	if (pos.EmptyCount() >= 27)
	//		++sort_depth;
	//	if (sort_depth < 0)
	//		sort_depth = 0;
	//	else if (sort_depth > 6)
	//		sort_depth = 6;
	//}
	//else
	//	sort_depth = -1;

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
			int sc = ((((0x0100000000000001ULL & O) << 1) | ((0x8000000000000080ULL & O) >> 1) | ((0x0000000000000081ULL & O) << 8) | ((0x8100000000000000ULL & O) >> 8) | 0x8100000000000081ULL) & O);
			auto se = StableEdges(next);
			int potential_mobility = popcount(E & O9) + popcount(E & O9 & BitBoard::Corners());

			uint32_t score = 0;
			//if (sort_depth < 0)
			//{
				//score += potential_mobility << 8;
				//score += (64 - popcount(sc)) << 13;
				//score += (pm.size() + (pm & BitBoard::Corners()).size()) << 18;
			//}
			//else
			//{
				//score += potential_mobility << 8;
				//score += (64 - popcount(se & O)) << 13;
				//score += (pm.size() + (pm & BitBoard::Corners()).size()) << 18;
			//	//switch (sort_depth)
			//	//{
			//	//	case 0:
			//	//		score += (Eval_d0(next) >> 1) << 18; // 1 level score bonus
			//	//		break;
			//	//	case 1:
			//	//	case 2:
			//	//		score += ((Eval_dN(next, sort_depth, { -inf_score, +inf_score }))) << 18;  // 2 level score bonus
			//	//		break;
			//	//	default:
			//	//		//score += 1 << 18;
			//	//		//if (tt.LookUp(pos).has_value())
			//	//		//	score -= 1 << 18; // bonus if the position leads to a position stored in the hash-table
			//	//		score += ((Eval_dN(next, sort_depth, { -inf_score, +inf_score }))) << 18; // > 3 level bonus
			//	//		break;
			//	//}
				score += (popcount(O & E9) + popcount(O & E9 & BitBoard::Corners())) << 8;
				score += potential_mobility << 9; // get_potential_moves
				score += (64 - popcount(se & O)) << 14;
				score += (pm.size() + (pm & BitBoard::Corners()).size()) << 18;
			//}

			return score;
		});
}
