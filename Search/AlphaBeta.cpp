#include "Game/Game.h"
#include "Algorithm.h"
#include <algorithm>
#include <cassert>
#include <chrono>

Result AlphaBeta::Eval(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	return Result::Exact(Eval_N(pos, window), pos.EmptyCount(), inf, Field::PS);
}

int AlphaBeta::Eval_N(const Position& pos, OpenInterval window)
{
	if (pos.EmptyCount() <= 7)
		return Eval_P(pos, window);

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -Eval_N(passed, -window);
		return EndScore(pos);
	}

	int best_score = -inf_score;
	for (Field move : moves)
	{
		int score = -Eval_N(Play(pos, move), -window);
		if (score > window)
			return score;
		best_score = std::max(best_score, score);
		window.lower = std::max(window.lower, score);
	}
	return best_score;
}

int AlphaBeta::Eval_P(const Position& pos, OpenInterval window)
{
	if (pos.EmptyCount() <= 3)
	{
		Moves moves{ pos.Empties() };
		Field move1 = moves.front(); moves.pop_front();
		Field move2 = moves.front(); moves.pop_front();
		Field move3 = moves.front();
		switch (pos.EmptyCount())
		{
		case 3: return Eval_3(pos, window, move1, move2, move3);
		case 2: return Eval_2(pos, window, move1, move2);
		case 1: return Eval_1(pos, move1);
		case 0: return Eval_0(pos);
		}
	}

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -Eval_P(passed, -window);
		return EndScore(pos);
	}

	int best_score = -inf_score;
	auto pq = ParityQuadrants(pos.Empties());
	for (auto filter : { pq, ~pq })
		for (Field move : moves & filter)
		{
			int score = -Eval_P(Play(pos, move), -window);
			if (score > window)
				return score;
			best_score = std::max(best_score, score);
			window.lower = std::max(window.lower, score);
		}
	return best_score;
}

int AlphaBeta::Eval_3(const Position& pos, OpenInterval window, Field move1, Field move2, Field move3)
{
	assert(pos.EmptyCount() == 3);
	nodes++;
	int best_score = -inf_score;

	if (auto flips = Flips(pos, move1))
	{
		best_score = -Eval_2(Play(pos, move1, flips), -window, move2, move3);
		if (best_score > window)
			return best_score;
		window.lower = std::max(window.lower, best_score);
	}

	if (auto flips = Flips(pos, move2))
	{
		int score = -Eval_2(Play(pos, move2, flips), -window, move1, move3);
		if (score > window)
			return score;
		best_score = std::max(best_score, score);
		window.lower = std::max(window.lower, score);
	}

	if (auto flips = Flips(pos, move3))
	{
		int score = -Eval_2(Play(pos, move3, flips), -window, move1, move2);
		return std::max(best_score, score);
	}

	if (best_score != -inf_score)
		return best_score;

	if (auto passed = PlayPass(pos); PossibleMoves(passed))
		return -Eval_3(passed, -window, move1, move2, move3);
	return EndScore(pos);
}

int AlphaBeta::Eval_2(const Position& pos, OpenInterval window, Field move1, Field move2)
{
	assert(pos.EmptyCount() == 2);
	nodes++;
	int best_score = -inf_score;

	if (auto flips = Flips(pos, move1))
	{
		best_score = -Eval_1(Play(pos, move1, flips), move2);
		if (best_score > window)
			return best_score;
		window.lower = std::max(window.lower, best_score);
	}

	if (auto flips = Flips(pos, move2))
	{
		int score = -Eval_1(Play(pos, move2, flips), move1);
		return std::max(best_score, score);
	}
	
	if (best_score != -inf_score)
		return best_score;

	
	if (auto passed = PlayPass(pos); PossibleMoves(passed))
		return -Eval_2(passed, -window, move1, move2);
	return EndScore(pos);
}
