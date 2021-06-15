#include "Algorithm.h"
#include <algorithm>

int NegaMax::Eval(const Position& pos)
{
	Moves moves{ pos.Empties() };
	const auto move1 = moves.ExtractFirst();
	const auto move2 = moves.ExtractFirst();
	const auto move3 = moves.ExtractFirst();
	switch (pos.EmptyCount())
	{
		case 0: return Eval_0(pos);
		case 1: return Eval_1(pos, move1);
		case 2: return Eval_2(pos, move1, move2);
		case 3: return Eval_3(pos, move1, move2, move3);
		default: return Eval_N(pos);
	}
}

int NegaMax::Eval_0(const Position& pos)
{
	assert(pos.EmptyCount() == 0);
	nodes++;
	return EvalGameOver(pos);
}

int NegaMax::Eval_1(const Position& pos, Field move1)
{
	assert(pos.EmptyCount() == 1);
	const int score = popcount(pos.Player()) - 31; // Assumes Player can play.

	if (const auto diff = CountLastFlip(pos, move1))
	{
		nodes += 2;
		return score + diff;
	}
	if (const auto diff = CountLastFlip(PlayPass(pos), move1))
	{
		nodes += 3;
		return score - diff - 1;
	}

	nodes++;
	return (score > 0) ? score : score - 1;
}

int NegaMax::Eval_2(const Position& pos, Field move1, Field move2)
{
	assert(pos.EmptyCount() == 2);
	nodes++;
	int score = -inf_score;

	if (const auto flips = Flips(pos, move1))
		score = -Eval_1(Play(pos, move1, flips), move2);

	if (const auto flips = Flips(pos, move2))
		score = std::max(score, -Eval_1(Play(pos, move2, flips), move1));

	if (score != -inf_score)
		return score;

	const auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_2(passed, move1, move2);
	return EvalGameOver(pos);
}

int NegaMax::Eval_3(const Position& pos, Field move1, Field move2, Field move3)
{
	assert(pos.EmptyCount() == 3);
	nodes++;
	int score = -inf_score;

	if (const auto flips = Flips(pos, move1))
		score = -Eval_2(Play(pos, move1, flips), move2, move3);

	if (const auto flips = Flips(pos, move2))
		score = std::max(score, -Eval_2(Play(pos, move2, flips), move1, move3));

	if (const auto flips = Flips(pos, move3))
		score = std::max(score, -Eval_2(Play(pos, move3, flips), move1, move2));

	if (score != -inf_score)
		return score;

	const auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_3(passed, move1, move2, move3);
	return EvalGameOver(pos);
}

int NegaMax::Eval_N(const Position& pos)
{
	assert(pos.EmptyCount() >= 3);
	if (pos.EmptyCount() == 3)
		return Eval(pos);

	nodes++;
	const Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		const auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_N(passed);
		return EvalGameOver(pos);
	}

	int score = -inf_score;
	for (const auto& move : moves)
		score = std::max(score, -Eval_N(Play(pos, move)));
	return score;
}
