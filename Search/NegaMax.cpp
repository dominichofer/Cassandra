#include "NegaMax.h"
#include <algorithm>
#include <cassert>
#include <chrono>

ScoreTimeNodes NegaMax::Eval(const Position& pos)
{
	nodes = 0;
	auto start = std::chrono::high_resolution_clock::now();
	int score = Eval_N(pos);
	auto time = std::chrono::high_resolution_clock::now() - start;
	return { score, time, nodes };
}

int NegaMax::Eval_N(const Position& pos)
{
	if (pos.EmptyCount() <= 3)
	{
		Moves moves{ pos.Empties() };
		Field move1 = moves.front(); moves.pop_front();
		Field move2 = moves.front(); moves.pop_front();
		Field move3 = moves.front();
		switch (pos.EmptyCount())
		{
			case 0: return Eval_0(pos);
			case 1: return Eval_1(pos, move1);
			case 2: return Eval_2(pos, move1, move2);
			case 3: return Eval_3(pos, move1, move2, move3);
		}
	}

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		auto passed = PlayPass(pos);
		if (PossibleMoves(passed))
			return -Eval_N(passed);
		return EndScore(pos);
	}

	int score = -inf_score;
	for (Field move : moves)
		score = std::max(score, -Eval_N(Play(pos, move)));
	return score;
}

int NegaMax::Eval_3(const Position& pos, Field move1, Field move2, Field move3)
{
	assert(pos.EmptyCount() == 3);
	nodes++;
	int score = -inf_score;

	if (auto flips = Flips(pos, move1))
		score = -Eval_2(Play(pos, move1, flips), move2, move3);

	if (auto flips = Flips(pos, move2))
		score = std::max(score, -Eval_2(Play(pos, move2, flips), move1, move3));

	if (auto flips = Flips(pos, move3))
		return std::max(score, -Eval_2(Play(pos, move3, flips), move1, move2));

	if (score != -inf_score)
		return score;

	auto passed = PlayPass(pos);
	if (PossibleMoves(passed))
		return -Eval_3(passed, move1, move2, move3);
	return EndScore(pos);
}

int NegaMax::Eval_2(const Position& pos, Field move1, Field move2)
{
	assert(pos.EmptyCount() == 2);
	nodes++;
	int score = -inf_score;

	if (auto flips = Flips(pos, move1))
		score = -Eval_1(Play(pos, move1, flips), move2);

	if (auto flips = Flips(pos, move2))
		return std::max(score, -Eval_1(Play(pos, move2, flips), move1));

	if (score != -inf_score)
		return score;

	auto passed = PlayPass(pos);
	if (PossibleMoves(passed))
		return -Eval_2(passed, move1, move2);
	return EndScore(pos);
}

int NegaMax::Eval_1(const Position& pos, Field move1)
{
	assert(pos.EmptyCount() == 1);
	int score = std::popcount(pos.Player()) - 31; // Assumes Player can play.

	if (auto diff = CountLastFlip(pos, move1))
	{
		nodes += 2;
		return score + diff;
	}
	if (auto diff = CountLastFlip(PlayPass(pos), move1))
	{
		nodes += 3;
		return score - diff - 1;
	}
	nodes++;
	return score - (score <= 0);
}

int NegaMax::Eval_0(const Position& pos)
{
	assert(pos.EmptyCount() == 0);
	nodes++;
	return EndScore(pos);
}
