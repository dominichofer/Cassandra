#include "AlphaBetaFailSoft.h"
#include "Core/Core.h"
#include <algorithm>

using namespace Search;

Result AlphaBetaFailSoft::Eval(Position pos, Intensity intensity)
{
	node_counter = 0;
	Score score = Eval_triage(pos, intensity.window);

	if (score > intensity.window)
		return Result::MinBound(score, pos.EmptyCount(), Selectivity::None, Field::invalid, node_counter);
	if (score < intensity.window)
		return Result::MaxBound(score, pos.EmptyCount(), Selectivity::None, Field::invalid, node_counter);
	return Result::ExactScore(score, pos.EmptyCount(), Selectivity::None, Field::invalid, node_counter);
}

Score AlphaBetaFailSoft::Eval_triage(const Position& pos, OpenInterval w)
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
		default: return Eval_N(pos, w);
	}
}

Score AlphaBetaFailSoft::Eval_2(const Position& pos, OpenInterval w, const Field move1, const Field move2)
{
	node_counter++;
	Score bestscore = -infinity;

	if (const auto flips = Flips(pos, move1))
	{
		bestscore = -NegaMax::Eval_1(Play(pos, move1, flips), move2);
		if (bestscore > w)
			return bestscore;
	}

	if (const auto flips = Flips(pos, move2))
	{
		const auto score = -NegaMax::Eval_1(Play(pos, move2, flips), move1);
		return std::max(bestscore, score);
	}

	if (bestscore != -infinity)
		return bestscore;

	const auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_2(passed, -w, move1, move2);
	return EvalGameOver(pos);
}

Score AlphaBetaFailSoft::Eval_3(const Position& pos, OpenInterval w, const Field move1, const Field move2, const Field move3)
{
	node_counter++;
	Score bestscore = -infinity;

	if (const auto flips = Flips(pos, move1))
	{
		bestscore = -Eval_2(Play(pos, move1, flips), -w, move2, move3);
		if (bestscore > w)
			return bestscore;
		w.try_increase_lower(bestscore);
	}

	if (const auto flips = Flips(pos, move2))
	{
		const auto score = -Eval_2(Play(pos, move2, flips), -w, move1, move3);
		if (score > w)
			return score;
		w.try_increase_lower(score);
		bestscore = std::max(bestscore, score);
	}

	if (const auto flips = Flips(pos, move3))
		return std::max(bestscore, -Eval_2(Play(pos, move3, flips), -w, move1, move2));

	if (bestscore != -infinity)
		return bestscore;

	const auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_3(passed, -w, move1, move2, move3);
	return EvalGameOver(pos);
}

Score AlphaBetaFailSoft::Eval_N(const Position& pos, OpenInterval w)
{
	if (pos.EmptyCount() <= 3)
		return Eval_triage(pos, w);

	node_counter++;

	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		const auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_N(passed, -w);
		return EvalGameOver(pos);
	}

	Score bestscore = -infinity;
	for (auto move : moves)
	{
		const auto score = -Eval_N(Play(pos, move), -w);
		if (score > w)
			return score;
		w.try_increase_lower(score);
		bestscore = std::max(bestscore, score);
	}

	return bestscore;
}
