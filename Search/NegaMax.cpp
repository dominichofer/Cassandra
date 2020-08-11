#include "NegaMax.h"
#include <algorithm>

using namespace Search;

Result NegaMax::Eval(Position pos, Intensity)
{
	node_counter = 0;
	Score score = Eval_triage(pos);
	return Result::ExactScore(score, pos.EmptyCount(), Selectivity::None, Field::invalid, node_counter);
}

Score NegaMax::Eval_triage(const Position& pos)
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

Score NegaMax::Eval_0(const Position& pos)
{
	node_counter++;
	return EvalGameOver(pos);
}

Score NegaMax::Eval_1(const Position& pos, const Field move1)
{
	const Score score = static_cast<Score>(2 * popcount(pos.Player())) - 63; // == popcount(pos.P) - popcount(pos.O)

	if (const auto diff = CountLastFlip(pos, move1))
	{
		node_counter += 2;
		return score + diff + 1;
	}
	if (const auto diff = CountLastFlip(PlayPass(pos), move1))
	{
		node_counter += 3;
		return score - diff - 1;
	}

	node_counter++;
	return (score > 0) ? score + 1 : score - 1;
}

Score NegaMax::Eval_2(const Position& pos, const Field move1, const Field move2)
{
	node_counter++;
	Score score = -infinity;

	if (const auto flips = Flips(pos, move1))
		score = -Eval_1(Play(pos, move1, flips), move2);

	if (const auto flips = Flips(pos, move2))
		score = std::max(score, -Eval_1(Play(pos, move2, flips), move1));

	if (score != -infinity)
		return score;

	const auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_2(passed, move1, move2);
	return EvalGameOver(pos);
}

Score NegaMax::Eval_3(const Position& pos, const Field move1, const Field move2, const Field move3)
{
	node_counter++;
	Score score = -infinity;

	if (const auto flips = Flips(pos, move1))
		score = -Eval_2(Play(pos, move1, flips), move2, move3);

	if (const auto flips = Flips(pos, move2))
		score = std::max(score, -Eval_2(Play(pos, move2, flips), move1, move3));

	if (const auto flips = Flips(pos, move3))
		score = std::max(score, -Eval_2(Play(pos, move3, flips), move1, move2));

	if (score != -infinity)
		return score;

	const auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_3(passed, move1, move2, move3);
	return EvalGameOver(pos);
}

Score NegaMax::Eval_N(const Position& pos)
{
	if (pos.EmptyCount() <= 3)
		return Eval_triage(pos);

	node_counter++;

	const Moves moves = PossibleMoves(pos);
	if (!moves)
	{
		const auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_N(passed);
		return EvalGameOver(pos);
	}

	Score score = -infinity;
	for (auto move : moves)
		score = std::max(score, -Eval_N(Play(pos, move)));

	return score;
}
