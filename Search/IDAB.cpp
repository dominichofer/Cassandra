#include "Core/Core.h"
#include "Algorithm.h"
#include <iostream>

using namespace Search;

int IDAB::Eval(const Position& pos, const Intensity& intensity, const OpenInterval&)
{
	return Eval_N(pos, intensity);
}

int IDAB::Eval(const Position& pos, const Intensity& intensity)
{
	return Eval_N(pos, intensity);
}

int IDAB::Eval(const Position& pos, const OpenInterval& window)
{
	return Eval_N(pos, Intensity::Exact(pos));
}

int IDAB::Eval(const Position& pos)
{
	return Eval_N(pos, Intensity::Exact(pos));
}

int IDAB::Eval_N(const Position& pos, const Intensity& request)
{
	const std::vector confidence_levels = { 1.1_sigmas, 1.5_sigmas, Confidence::Certain() };
	const int E = pos.EmptyCount();
	const int D = request.depth;
	int score;

	// Iterative deepening
	for (int d = 6; d < std::min(E - 10, D); d++)
		score = PVS::Eval(pos, { d, confidence_levels[0] });

	// Iterative broadening
	for (int level = 0; confidence_levels[level] < request.certainty; level++)
		score = PVS::Eval(pos, { D, confidence_levels[level] });

	//return score;
	//return PVS::Eval(pos, request);
	return MTD_f(pos, request, score);
}

int IDAB::MTD_f(const Position& pos, const Intensity& request, int guess)
{
	int upperBound = max_score;
	int lowerBound = min_score;
	while (lowerBound < upperBound)
	{
		int beta = std::max(guess, lowerBound + 1);
		guess = PVS::Eval(pos, request, { beta - 1, beta });
		if (guess < beta)
			upperBound = guess;
		else
			lowerBound = guess;
	}
	return guess;
}