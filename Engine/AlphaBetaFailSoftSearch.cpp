#include "AlphaBetaFailSoftSearch.h"
#include "Core/Machine.h"
#include <algorithm>

using namespace Search;

Result AlphaBetaFailSoft::Eval(Position pos, Intensity intensity)
{
	node_counter = 0;
	Score score = Eval_triage(pos, intensity.window);
	return Result::ExactScore(score, pos.EmptyCount(), Selectivity::None, Field::invalid, node_counter);
}

Score AlphaBetaFailSoft::Eval_triage(const Position& pos, ExclusiveInterval w)
{
	auto moves = Moves(pos.Empties());
	const auto move1 = moves.front(); moves.pop_front();
	const auto move2 = moves.front(); moves.pop_front();
	const auto move3 = moves.front(); moves.pop_front();
	const auto move4 = moves.front(); moves.pop_front();
	switch (pos.EmptyCount())
	{
		case 0: return NegaMax::Eval_0(pos);
		case 1: return NegaMax::Eval_1(pos, move1);
		case 2: return Eval_2(pos, w, move1, move2);
		case 3: return Eval_3(pos, w, move1, move2, move3);
		case 4: return Eval_4(pos, w, move1, move2, move3, move4);
		default: return Eval_N(pos, w);
	}
}

Score AlphaBetaFailSoft::Eval_2(const Position& pos, ExclusiveInterval w, const Field move1, const Field move2)
{
	node_counter++;
	Score bestscore = -infinity;

	if (const auto flips = Flips(pos, move1))
	{
		bestscore = -NegaMax::Eval_1(Play(pos, move1, flips), move2);
		if (bestscore >= w.upper)
			return bestscore;
	}

	if (const auto flips = Flips(pos, move2))
	{
		const auto score = -NegaMax::Eval_1(Play(pos, move2, flips), move1);
		return std::max(bestscore, score);
	}

	if (bestscore != -infinity)
		return bestscore;
	bestscore = infinity;

	const auto passed = PlayPass(pos);
	node_counter++;

	if (const auto flips = Flips(passed, move1))
	{
		bestscore = NegaMax::Eval_1(Play(passed, move1, flips), move2);
		if (bestscore <= w.lower)
			return bestscore;
	}

	if (const auto flips = Flips(passed, move2))
		return std::min(bestscore, NegaMax::Eval_1(Play(passed, move2, flips), move1));

	if (bestscore != infinity)
		return bestscore;

	node_counter--;
	return -EvalGameOver(passed);
}

Score AlphaBetaFailSoft::Eval_3(const Position& pos, ExclusiveInterval w, const Field move1, const Field move2, const Field move3)
{
	node_counter++;
	Score bestscore = -infinity;

	if (const auto flips = Flips(pos, move1))
	{
		bestscore = -Eval_2(Play(pos, move1, flips), -w, move2, move3);
		if (bestscore >= w.upper)
			return bestscore;
		w.lower = std::max(w.lower, bestscore);
	}

	if (const auto flips = Flips(pos, move2))
	{
		const auto score = -Eval_2(Play(pos, move2, flips), -w, move1, move3);
		if (score >= w.upper)
			return score;
		w.lower = std::max(w.lower, score);
		bestscore = std::max(bestscore, score);
	}

	if (const auto flips = Flips(pos, move3))
		return std::max(bestscore, -Eval_2(Play(pos, move3, flips), -w, move1, move2));

	if (bestscore != -infinity)
		return bestscore;
	bestscore = infinity;

	const auto passed = PlayPass(pos);
	node_counter++;

	if (const auto flips = Flips(passed, move1))
	{
		bestscore = Eval_2(Play(passed, move1, flips), w, move2, move3);
		if (bestscore <= w.lower)
			return bestscore;
		w.upper = std::min(w.upper, bestscore);
	}

	if (const auto flips = Flips(passed, move2))
	{
		const auto score = Eval_2(Play(passed, move2, flips), w, move1, move3);
		if (score <= w.lower)
			return score;
		w.upper = std::min(w.upper, score);
		bestscore = std::min(bestscore, score);
	}

	if (const auto flips = Flips(passed, move3))
		return std::min(bestscore, Eval_2(Play(passed, move3, flips), w, move1, move2));

	if (bestscore != infinity)
		return bestscore;

	node_counter--;
	return -EvalGameOver(passed);
}

Score AlphaBetaFailSoft::Eval_4(const Position& pos, ExclusiveInterval w, const Field move1, const Field move2, const Field move3, const Field move4)
{
	node_counter++;
	Score bestscore = -infinity;

	if (const auto flips = Flips(pos, move1))
	{
		bestscore = -Eval_3(Play(pos, move1, flips), -w, move2, move3, move4);
		if (bestscore >= w.upper)
			return bestscore;
		if (bestscore > w.lower)
			w.lower = bestscore;
	}

	if (const auto flips = Flips(pos, move2))
	{
		const auto score = -Eval_3(Play(pos, move2, flips), -w, move1, move3, move4);
		if (score >= w.upper)
			return score;
		w.lower = std::max(w.lower, score);
		bestscore = std::max(bestscore, score);
	}

	if (const auto flips = Flips(pos, move3))
	{
		const auto score = -Eval_3(Play(pos, move3, flips), -w, move1, move2, move4);
		if (score >= w.upper)
			return score;
		w.lower = std::max(w.lower, score);
		bestscore = std::max(bestscore, score);
	}

	if (const auto flips = Flips(pos, move4))
		return std::max(bestscore, -Eval_3(Play(pos, move4, flips), -w, move1, move2, move3));

	if (bestscore != -infinity)
		return bestscore;
	bestscore = infinity;

	const auto passed = PlayPass(pos);
	node_counter++;

	if (const auto flips = Flips(passed, move1))
	{
		bestscore = Eval_3(Play(passed, move1, flips), w, move2, move3, move4);
		if (bestscore <= w.lower)
			return bestscore;
		w.upper = std::min(w.upper, bestscore);
	}

	if (const auto flips = Flips(passed, move2))
	{
		const auto score = Eval_3(Play(passed, move2, flips), w, move1, move3, move4);
		if (score <= w.lower)
			return score;
		w.upper = std::min(w.upper, score);
		bestscore = std::min(bestscore, score);
	}

	if (const auto flips = Flips(passed, move3))
	{
		const auto score = Eval_3(Play(passed, move3, flips), w, move1, move2, move4);
		if (score <= w.lower)
			return score;
		w.upper = std::min(w.upper, score);
		bestscore = std::min(bestscore, score);
	}

	if (const auto flips = Flips(passed, move4))
		return std::min(bestscore, Eval_3(Play(passed, move4, flips), w, move1, move2, move3));

	if (bestscore != infinity)
		return bestscore;

	node_counter--;
	return -EvalGameOver(passed);
}

Score AlphaBetaFailSoft::Eval_N(const Position& pos, ExclusiveInterval w)
{
	if (pos.EmptyCount() <= 4)
		return Eval_triage(pos, w);

	node_counter++;

	Moves moves = PossibleMoves(pos);
	if (moves.empty())
	{
		const auto passed = PlayPass(pos);
		if (!PossibleMoves(passed).empty())
			return -Eval_N(passed, -w);
		return EvalGameOver(pos);
	}

	Score bestscore = -infinity;
	for (auto move : moves)
	{
		const auto score = -Eval_N(Play(pos, move), -w);
		if (score >= w.upper)
			return score;
		w.lower = std::max(w.lower, score);
		bestscore = std::max(bestscore, score);
	}

	return bestscore;
}
