#include "Algorithm.h"
#include "Core/Core.h"
#include <algorithm>

using namespace Search;

Result AlphaBetaFailHard::Eval(const Position& pos, const Request& request)
{
	auto score = Eval(pos, static_cast<OpenInterval>(request));
	return Result::ExactFailHard(request, pos, score);
}

int AlphaBetaFailHard::Eval(const Position& pos, OpenInterval window)
{
	Moves moves{ pos.Empties() };
	const auto move1 = moves.ExtractFirst();
	const auto move2 = moves.ExtractFirst();
	const auto move3 = moves.ExtractFirst();
	switch (pos.EmptyCount())
	{
		case 0: return Eval_0(pos, window);
		case 1: return Eval_1(pos, window, move1);
		case 2: return Eval_2(pos, window, move1, move2);
		case 3: return Eval_3(pos, window, move1, move2, move3);
		default: return Eval_N(pos, window);
	}
}

int AlphaBetaFailHard::Eval_0(const Position& pos, OpenInterval window)
{
	assert(pos.EmptyCount() == 0);
	return window.clamp(NegaMax::Eval_0(pos));
}

int AlphaBetaFailHard::Eval_1(const Position& pos, OpenInterval window, Field move1)
{
	assert(pos.EmptyCount() == 1);
	return window.clamp(NegaMax::Eval_1(pos, move1));
}

int AlphaBetaFailHard::Eval_2(const Position& pos, OpenInterval window, Field move1, Field move2)
{
	assert(pos.EmptyCount() == 2);
	nodes++;
	int score = inf_score;

	if (const auto flips = Flips(pos, move1)) {
		score = -Eval_1(Play(pos, move1, flips), -window, move2);
		if (score > window)
			return window.upper();
		window.TryIncreaseLower(score);
	}

	if (const auto flips = Flips(pos, move2))
		return window.clamp(-Eval_1(Play(pos, move2, flips), -window, move1));

	if (score != inf_score)
		return window.lower();

	const auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_2(passed, -window, move1, move2);
	return window.clamp(EvalGameOver(pos));
}

int AlphaBetaFailHard::Eval_3(const Position& pos, OpenInterval window, Field move1, Field move2, Field move3)
{
	assert(pos.EmptyCount() == 3);
	nodes++;
	int score = inf_score;

	if (const auto flips = Flips(pos, move1)) {
		score = -Eval_2(Play(pos, move1, flips), -window, move2, move3);
		if (score > window)
			return window.upper();
		window.TryIncreaseLower(score);
	}

	if (const auto flips = Flips(pos, move2)) {
		score = -Eval_2(Play(pos, move2, flips), -window, move1, move3);
		if (score > window)
			return window.upper();
		window.TryIncreaseLower(score);
	}

	if (const auto flips = Flips(pos, move3))
		return window.clamp(-Eval_2(Play(pos, move3, flips), -window, move1, move2));

	if (score != inf_score)
		return window.lower();

	const auto passed = PlayPass(pos);
	if (HasMoves(passed))
		return -Eval_3(passed, -window, move1, move2, move3);
	return window.clamp(EvalGameOver(pos));
}

int AlphaBetaFailHard::Eval_N(const Position& pos, OpenInterval window)
{
	assert(pos.EmptyCount() >= 3);
	if (pos.EmptyCount() == 3)
		return Eval(pos, window);

	nodes++;
	Moves moves = PossibleMoves(pos);
	if (!moves) {
		const auto passed = PlayPass(pos);
		if (HasMoves(passed))
			return -Eval_N(passed, -window);
		return window.clamp(EvalGameOver(pos));
	}

	for (const auto& move : moves)
	{
		const auto score = -Eval_N(Play(pos, move), -window);
		if (score > window)
			return window.upper();
		window.TryIncreaseLower(score);
	}
	return window.lower();
}

