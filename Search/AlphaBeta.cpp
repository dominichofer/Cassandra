#include "AlphaBeta.h"
#include <algorithm>
#include <cassert>
#include <chrono>

ScoreTimeNodes AlphaBeta::Eval(const Position& pos, OpenInterval window)
{
	nodes = 0;
	auto start = std::chrono::high_resolution_clock::now();
	int score = Eval_N(pos, window);
	auto time = std::chrono::high_resolution_clock::now() - start;
	return { score, time, nodes };
}

int AlphaBeta::Eval_N(const Position& pos, OpenInterval window)
{
	if (pos.EmptyCount() <= 7)
		return Eval_P(pos, window);
	if (pos.EmptyCount() <= 4)
	{
		Moves moves{ pos.Empties() };
		Field move1 = moves.front(); moves.pop_front();
		Field move2 = moves.front(); moves.pop_front();
		Field move3 = moves.front(); /*moves.pop_front();
		Field move4 = moves.front();*/
		switch (pos.EmptyCount())
		{
		case 0: return Eval_0(pos);
		case 1: return Eval_1(pos, move1);
		case 2: return Eval_2(pos, window, move1, move2);
		case 3: return Eval_3(pos, window, move1, move2, move3);
		//case 4: return Eval_4(pos, window, move1, move2, move3, move4);
		}
	}

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -Eval_N(passed, -window);
		return EndScore(pos);
	}

	for (Field move : moves)
	{
		int score = -Eval_N(Play(pos, move), -window);
		if (score > window)
			return score;
		window.lower = std::max(window.lower, score);
	}
	return window.lower;
}

int AlphaBeta::Eval_P(const Position& pos, OpenInterval window)
{
	if (pos.EmptyCount() <= 4)
	{
		Moves moves{ pos.Empties() };
		Field move1 = moves.front(); moves.pop_front();
		Field move2 = moves.front(); moves.pop_front();
		Field move3 = moves.front(); /*moves.pop_front();
		Field move4 = moves.front();*/
		switch (pos.EmptyCount())
		{
		case 0: return Eval_0(pos);
		case 1: return Eval_1(pos, move1);
		case 2: return Eval_2(pos, window, move1, move2);
		case 3: return Eval_3(pos, window, move1, move2, move3);
		//case 4: return Eval_4(pos, window, move1, move2, move3, move4);
		}
	}

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -Eval_N(passed, -window);
		return EndScore(pos);
	}

	auto pq = ParityQuadrants(pos.Empties());
	for (auto filter : { pq, ~pq })
		for (Field move : moves & filter)
		{
			int score = -Eval_N(Play(pos, move), -window);
			if (score > window)
				return score;
			window.lower = std::max(window.lower, score);
		}
	return window.lower;
}

int AlphaBeta::Eval_4(const Position& pos, OpenInterval window, Field move1, Field move2, Field move3, Field move4)
{
	assert(pos.EmptyCount() == 4);
	nodes++;
	int score = -inf_score;

	if (auto flips = Flips(pos, move1))
	{
		score = -Eval_3(Play(pos, move1, flips), -window, move2, move3, move4);
		if (score > window)
			return score;
		window.lower = std::max(window.lower, score);
	}

	if (auto flips = Flips(pos, move2))
	{
		score = -Eval_3(Play(pos, move2, flips), -window, move1, move3, move4);
		if (score > window)
			return score;
		window.lower = std::max(window.lower, score);
	}

	if (auto flips = Flips(pos, move3))
	{
		score = -Eval_3(Play(pos, move3, flips), -window, move1, move2, move4);
		if (score > window)
			return score;
		window.lower = std::max(window.lower, score);
	}

	if (auto flips = Flips(pos, move4))
	{
		score = -Eval_3(Play(pos, move4, flips), -window, move1, move2, move3);
		return std::max(window.lower, score);
	}

	if (score != -inf_score)
		return window.lower;

	auto passed = PlayPass(pos);
	if (PossibleMoves(passed))
		return -Eval_4(passed, -window, move1, move2, move3, move4);
	return EndScore(pos);
}

int AlphaBeta::Eval_3(const Position& pos, OpenInterval window, Field move1, Field move2, Field move3)
{
	assert(pos.EmptyCount() == 3);
	nodes++;
	int score = -inf_score;

	if (auto flips = Flips(pos, move1))
	{
		score = -Eval_2(Play(pos, move1, flips), -window, move2, move3);
		if (score > window)
			return score;
		window.lower = std::max(window.lower, score);
	}

	if (auto flips = Flips(pos, move2))
	{
		score = -Eval_2(Play(pos, move2, flips), -window, move1, move3);
		if (score > window)
			return score;
		window.lower = std::max(window.lower, score);
	}

	if (auto flips = Flips(pos, move3))
	{
		score = -Eval_2(Play(pos, move3, flips), -window, move1, move2);
		return std::max(window.lower, score);
	}

	if (score != -inf_score)
		return window.lower;

	auto passed = PlayPass(pos);
	if (PossibleMoves(passed))
		return -Eval_3(passed, -window, move1, move2, move3);
	return EndScore(pos);
}

int AlphaBeta::Eval_2(const Position& pos, OpenInterval window, Field move1, Field move2)
{
	assert(pos.EmptyCount() == 2);
	nodes++;
	int score = -inf_score;

	if (auto flips = Flips(pos, move1))
	{
		score = -Eval_1(Play(pos, move1, flips), move2);
		if (score > window)
			return score;
		window.lower = std::max(window.lower, score);
	}

	if (auto flips = Flips(pos, move2))
	{
		score = -Eval_1(Play(pos, move2, flips), move1);
		return std::max(window.lower, score);
	}

	if (score != -inf_score)
		return window.lower;

	auto passed = PlayPass(pos);
	if (PossibleMoves(passed))
		return -Eval_2(passed, -window, move1, move2);
	return EndScore(pos);
}
