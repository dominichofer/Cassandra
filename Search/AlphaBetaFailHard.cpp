#include "AlphaBetaFailHard.h"
#include "Core/Core.h"
#include <algorithm>

using namespace Search;

Result AlphaBetaFailHard::Eval(Position pos, Intensity intensity)
{
	node_counter = 0;
	Score score = Eval_triage(pos, intensity.window);

	if (score > intensity.window)
		return Result::MinBound(intensity.window.upper(), pos.EmptyCount(), Selectivity::None, Field::invalid, node_counter);
	if (score < intensity.window)
		return Result::MaxBound(intensity.window.lower(), pos.EmptyCount(), Selectivity::None, Field::invalid, node_counter);
	return Result::ExactScore(score, pos.EmptyCount(), Selectivity::None, Field::invalid, node_counter);
}

Score AlphaBetaFailHard::Eval_triage(const Position& pos, OpenInterval w)
{
	Moves moves{ pos.Empties() };
	const auto move1 = moves.ExtractFirst();
	const auto move2 = moves.ExtractFirst();
	const auto move3 = moves.ExtractFirst();
	switch (pos.EmptyCount())
	{
		case 0: return Eval_0(pos, w);
		case 1: return Eval_1(pos, w, move1);
		case 2: return Eval_2(pos, w, move1, move2);
		case 3: return Eval_3(pos, w, move1, move2, move3);
		default: return Eval_N(pos, w);
	}
}

Score AlphaBetaFailHard::Eval_0(const Position& pos, const OpenInterval w)
{
	return w.clamp(NegaMax::Eval_0(pos));
}

Score AlphaBetaFailHard::Eval_1(const Position& pos, const OpenInterval w, const Field move1)
{
	return w.clamp(NegaMax::Eval_1(pos, move1));
}

Score AlphaBetaFailHard::Eval_2(const Position& pos, OpenInterval w, const Field move1, const Field move2)
{
	node_counter++;
	Score score = infinity;

	if (const auto flips = Flips(pos, move1)) {
		score = -Eval_1(Play(pos, move1, flips), -w, move2);
		if (score > w)
			return w.upper();
		w.try_increase_lower(score);
	}

	if (const auto flips = Flips(pos, move2))
		return w.clamp(-Eval_1(Play(pos, move2, flips), -w, move1));

	if (score != infinity)
		return w.lower();

	const auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_2(passed, -w, move1, move2);
	return w.clamp(EvalGameOver(pos));
}

Score AlphaBetaFailHard::Eval_3(const Position& pos, OpenInterval w, const Field move1, const Field move2, const Field move3)
{
	node_counter++;
	Score score = infinity;

	if (const auto flips = Flips(pos, move1)) {
		score = -Eval_2(Play(pos, move1, flips), -w, move2, move3);
		if (score > w)
			return w.upper();
		w.try_increase_lower(score);
	}

	if (const auto flips = Flips(pos, move2)) {
		score = -Eval_2(Play(pos, move2, flips), -w, move1, move3);
		if (score > w)
			return w.upper();
		w.try_increase_lower(score);
	}

	if (const auto flips = Flips(pos, move3))
		return w.clamp(-Eval_2(Play(pos, move3, flips), -w, move1, move2));

	if (score != infinity)
		return w.lower();

	const auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_3(passed, -w, move1, move2, move3);
	return w.clamp(EvalGameOver(pos));
}

Score AlphaBetaFailHard::Eval_N(const Position& pos, OpenInterval w)
{
	if (pos.EmptyCount() <= 3)
		return Eval_triage(pos, w);

	node_counter++;

	Moves moves = PossibleMoves(pos);
	if (!moves) {
		const auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_N(passed, -w);
		return w.clamp(EvalGameOver(pos));
	}

	for (auto move : moves)
	{
		const auto score = -Eval_N(Play(pos, move), -w);
		if (score > w)
			return w.upper();
		w.try_increase_lower(score);
	}

	return w.lower();
}

