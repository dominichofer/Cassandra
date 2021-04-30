#include "Algorithm.h"
#include "Core/Core.h"
#include "SortedMoves.h"
#include <algorithm>

using namespace Search;

Result AlphaBetaFailSoft::Eval(const Position& pos, const Request& request)
{
	auto score = Eval(pos, static_cast<OpenInterval>(request));
	return Result::ExactFailSoft(request, pos, score);
}

int AlphaBetaFailSoft::Eval(const Position& pos, OpenInterval w)
{
	Moves moves{ pos.Empties() };
	const auto move1 = moves.ExtractFirst();
	const auto move2 = moves.ExtractFirst();
	const auto move3 = moves.ExtractFirst();
	switch (pos.EmptyCount())
	{
		case 0: return NegaMax::Eval_0(pos);
		case 1: return NegaMax::Eval_1(pos, move1);
		case 2: return Eval_2(pos, w, move1, move2);
		case 3: return Eval_3(pos, w, move1, move2, move3);
		default:
			if (pos.EmptyCount() <= Eval_to_ParitySort)
				return Eval_P(pos, w);
			return Eval_N(pos, w);
	}
}

int AlphaBetaFailSoft::Eval_2(const Position& pos, OpenInterval w, const Field move1, const Field move2)
{
	assert(pos.EmptyCount() == 2);
	node_count++;
	int bestscore = -inf_score;

	if (const auto flips = Flips(pos, move1))
	{
		bestscore = -Eval_1(Play(pos, move1, flips), move2);
		if (bestscore > w)
			return bestscore;
	}
	if (const auto flips = Flips(pos, move2))
	{
		const auto score = -Eval_1(Play(pos, move2, flips), move1);
		return std::max(bestscore, score);
	}

	if (bestscore != -inf_score)
		return bestscore;

	const auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_2(passed, -w, move1, move2);
	return EvalGameOver(pos);
}

int AlphaBetaFailSoft::Eval_3(const Position& pos, OpenInterval w, const Field move1, const Field move2, const Field move3)
{
	assert(pos.EmptyCount() == 3);
	node_count++;
	int bestscore = -inf_score;

	if (const auto flips = Flips(pos, move1))
	{
		bestscore = -Eval_2(Play(pos, move1, flips), -w, move2, move3);
		if (bestscore > w)
			return bestscore;
		w.TryIncreaseLower(bestscore);
	}
	if (const auto flips = Flips(pos, move2))
	{
		const auto score = -Eval_2(Play(pos, move2, flips), -w, move1, move3);
		if (score > w)
			return score;
		w.TryIncreaseLower(score);
		if (score > bestscore)
			bestscore = score;
	}

	if (const auto flips = Flips(pos, move3))
		return std::max(bestscore, -Eval_2(Play(pos, move3, flips), -w, move1, move2));

	if (bestscore != -inf_score)
		return bestscore;

	const auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_3(passed, -w, move1, move2, move3);
	return EvalGameOver(pos);
}

int AlphaBetaFailSoft::Eval_P(const Position& pos, OpenInterval w)
{
	assert(pos.EmptyCount() >= 3);
	assert(pos.EmptyCount() <= Eval_to_ParitySort);
	if (pos.EmptyCount() == 3) {
		Moves moves{ pos.Empties() };
		const auto move1 = moves.ExtractFirst();
		const auto move2 = moves.ExtractFirst();
		const auto move3 = moves.ExtractFirst();
		return Eval_3(pos, w, move1, move2, move3);
	}

	node_count++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{ 
		const auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_P(passed, -w);
		return EvalGameOver(pos);
	}

	int bestscore = -inf_score;
	Moves parity_moves = moves;
	const auto pq = pos.ParityQuadrants();
	for (auto filter : {pq, ~pq})
		for (const auto& move : moves.Filtered(filter))
		{
			const auto score = -Eval_P(Play(pos, move), -w);
			if (score > w)
				return score;
			w.TryIncreaseLower(score);
			if (score > bestscore)
				bestscore = score;
		}
	return bestscore;
}

int AlphaBetaFailSoft::Eval_N(const Position& pos, OpenInterval w)
{
	assert(pos.EmptyCount() >= Eval_to_ParitySort);
	if (pos.EmptyCount() == Eval_to_ParitySort)
		return Eval_P(pos, w);

	node_count++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		const auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_N(passed, -w);
		return EvalGameOver(pos);
	}
	
	//if (auto max = StabilityBasedMaxScore(pos); max < w)
	//	return max;

	SortedMoves sorted_moves(moves, [&](Field move) { return MoveOrderingScorer(pos, move); });
	for (const auto& move : sorted_moves)
	{
		const auto score = -Eval_N(Play(pos, move.second), -w);
		if (score > w)
			return score;
		w.TryIncreaseLower(score);
	}
	return w.lower();
}
