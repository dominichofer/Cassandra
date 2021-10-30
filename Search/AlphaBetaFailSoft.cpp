#include "Algorithm.h"
#include "Core/Core.h"
#include "SortedMoves.h"
#include <algorithm>

using namespace Search;

int AlphaBetaFailSoft::Eval(const Position& pos, const Intensity&, const OpenInterval& window)
{
	return Eval_N(pos, window);
}

int AlphaBetaFailSoft::Eval(const Position& pos, const Intensity&)
{
	return Eval_N(pos, OpenInterval::Whole());
}

int AlphaBetaFailSoft::Eval(const Position& pos, const OpenInterval& window)
{
	return Eval_N(pos, window);
}

int AlphaBetaFailSoft::Eval(const Position& pos)
{
	return Eval_N(pos, OpenInterval::Whole());
}

int AlphaBetaFailSoft::Eval_N(const Position& pos, OpenInterval window)
{
	if (pos.EmptyCount() <= Eval_to_ParitySort)
	{
		Moves moves{ pos.Empties() };
		Field move1 = moves.front(); moves.pop_front();
		Field move2 = moves.front(); moves.pop_front();
		Field move3 = moves.front();
		switch (pos.EmptyCount())
		{
			case 0: return NegaMax::Eval_0(pos);
			case 1: return NegaMax::Eval_1(pos, move1);
			case 2: return Eval_2(pos, window, move1, move2);
			case 3: return Eval_3(pos, window, move1, move2, move3);
			default: return Eval_P(pos, window);
		}
	}

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_N(passed, -window);
		return EvalGameOver(pos);
	}

	//if (auto max = StabilityBasedMaxScore(pos); max < window)
	//	return max;

	int best_score = -inf_score;
	SortedMoves sorted_moves(moves, [&](Field move) { return MoveOrderingScorer(pos, move); });
	for (const auto& move : sorted_moves)
	{
		int score = -Eval_N(Play(pos, move.second), -window);
		if (score > window)
			return score;
		window.TryIncreaseLower(score);
		if (score > best_score)
			best_score = score;
	}
	return best_score;
}

int AlphaBetaFailSoft::Eval_P(const Position& pos, OpenInterval window)
{
	assert(pos.EmptyCount() <= Eval_to_ParitySort);
	if (pos.EmptyCount() <= 3)
	{
		Moves moves{ pos.Empties() };
		Field move1 = moves.front(); moves.pop_front();
		Field move2 = moves.front(); moves.pop_front();
		Field move3 = moves.front();
		switch (pos.EmptyCount())
		{
			case 0: return NegaMax::Eval_0(pos);
			case 1: return NegaMax::Eval_1(pos, move1);
			case 2: return Eval_2(pos, window, move1, move2);
			case 3: return Eval_3(pos, window, move1, move2, move3);
		}
	}

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_P(passed, -window);
		return EvalGameOver(pos);
	}

	int best_score = -inf_score;
	auto pq = pos.ParityQuadrants();
	for (auto filter : { pq, ~pq })
		for (Field move : moves & filter)
		{
			int score = -Eval_P(Play(pos, move), -window);
			if (score > window)
				return score;
			window.TryIncreaseLower(score);
			if (score > best_score)
				best_score = score;
		}
	return best_score;
}

int AlphaBetaFailSoft::Eval_3(const Position& pos, OpenInterval window, Field move1, Field move2, Field move3)
{
	assert(pos.EmptyCount() == 3);
	nodes++;
	int best_score = -inf_score;

	if (auto flips = Flips(pos, move1))
	{
		best_score = -Eval_2(Play(pos, move1, flips), -window, move2, move3);
		if (best_score > window)
			return best_score;
		window.TryIncreaseLower(best_score);
	}
	if (auto flips = Flips(pos, move2))
	{
		int score = -Eval_2(Play(pos, move2, flips), -window, move1, move3);
		if (score > window)
			return score;
		window.TryIncreaseLower(score);
		if (score > best_score)
			best_score = score;
	}

	if (auto flips = Flips(pos, move3))
		return std::max(best_score, -Eval_2(Play(pos, move3, flips), -window, move1, move2));

	if (best_score != -inf_score)
		return best_score;

	auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_3(passed, -window, move1, move2, move3);
	return EvalGameOver(pos);
}

int AlphaBetaFailSoft::Eval_2(const Position& pos, OpenInterval window, Field move1, Field move2)
{
	assert(pos.EmptyCount() == 2);
	nodes++;
	int best_score = -inf_score;

	if (auto flips = Flips(pos, move1))
	{
		best_score = -Eval_1(Play(pos, move1, flips), move2);
		if (best_score > window)
			return best_score;
	}
	if (auto flips = Flips(pos, move2))
		return std::max(best_score, -Eval_1(Play(pos, move2, flips), move1));

	if (best_score != -inf_score)
		return best_score;

	auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_2(passed, -window, move1, move2);
	return EvalGameOver(pos);
}