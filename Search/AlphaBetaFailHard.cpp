#include "AlphaBetaFailHard.h"
#include <algorithm>
#include <cassert>

int AlphaBetaFailHard::Eval(const Position& pos, Intensity, OpenInterval window)
{
	nodes = 0;
	return Eval_N(pos, window);
}

ScoreMove AlphaBetaFailHard::Eval_BestMove(const Position& pos, Intensity, OpenInterval window)
{
	nodes = 0;
	return Eval_BestMove_N(pos, window);
}

ScoreMove AlphaBetaFailHard::Eval_BestMove_N(const Position& pos, OpenInterval window)
{
	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_BestMove_N(passed, -window);
		return EvalGameOver(pos);
	}

	ScoreMove best;
	for (Field move : moves)
	{
		int score = -Eval_N(Play(pos, move), -window);
		if (score > window)
			return { window.Upper(), move };
		best.ImproveWith(score, move);
		window.TryIncreaseLower(score);
	}
	return best;
}

int AlphaBetaFailHard::Eval_N(const Position& pos, OpenInterval window)
{
	if (pos.EmptyCount() <= 3)
	{
		Moves moves{ pos.Empties() };
		Field move1 = moves.front(); moves.pop_front();
		Field move2 = moves.front(); moves.pop_front();
		Field move3 = moves.front();
		switch (pos.EmptyCount())
		{
			case 0: return Eval_0(pos, window);
			case 1: return Eval_1(pos, window, move1);
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
			return -Eval_N(passed, -window);
		return window.clamp(EvalGameOver(pos));
	}

	for (Field move : moves)
	{
		int score = -Eval_N(Play(pos, move), -window);
		if (score > window)
			return window.Upper();
		window.TryIncreaseLower(score);
	}
	return window.Lower();
}

int AlphaBetaFailHard::Eval_3(const Position& pos, OpenInterval window, Field move1, Field move2, Field move3)
{
	assert(pos.EmptyCount() == 3);
	assert(pos.Empties().Get(move1));
	assert(pos.Empties().Get(move2));
	assert(pos.Empties().Get(move3));
	nodes++;
	int score = -inf_score;

	if (auto flips = Flips(pos, move1))
	{
		score = -Eval_2(Play(pos, move1, flips), -window, move2, move3);
		if (score > window)
			return window.Upper();
		window.TryIncreaseLower(score);
	}

	if (auto flips = Flips(pos, move2))
	{
		score = -Eval_2(Play(pos, move2, flips), -window, move1, move3);
		if (score > window)
			return window.Upper();
		window.TryIncreaseLower(score);
	}

	if (auto flips = Flips(pos, move3))
	{
		score = -Eval_2(Play(pos, move3, flips), -window, move1, move2);
		return window.clamp(score);
	}

	if (score != -inf_score)
		return window.Lower();

	auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_3(passed, -window, move1, move2, move3);
	return window.clamp(EvalGameOver(pos));
}

int AlphaBetaFailHard::Eval_2(const Position& pos, OpenInterval window, Field move1, Field move2)
{
	assert(pos.EmptyCount() == 2);
	assert(pos.Empties().Get(move1));
	assert(pos.Empties().Get(move2));
	nodes++;
	int score = -inf_score;

	if (auto flips = Flips(pos, move1))
	{
		score = -Eval_1(Play(pos, move1, flips), -window, move2);
		if (score > window)
			return window.Upper();
		window.TryIncreaseLower(score);
	}

	if (auto flips = Flips(pos, move2))
	{
		score = -Eval_1(Play(pos, move2, flips), -window, move1);
		return window.clamp(score);
	}

	if (score != -inf_score)
		return window.Lower();

	auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_2(passed, -window, move1, move2);
	return window.clamp(EvalGameOver(pos));
}

int AlphaBetaFailHard::Eval_1(const Position& pos, OpenInterval window, Field move1)
{
	assert(pos.EmptyCount() == 1);
	assert(pos.Empties().Get(move1));
	return window.clamp(NegaMax::Eval_1(pos, move1));
}

int AlphaBetaFailHard::Eval_0(const Position& pos, OpenInterval window)
{
	return window.clamp(NegaMax::Eval_0(pos));
}