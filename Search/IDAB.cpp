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

int IDAB::Eval_N(const Position& pos, const Intensity& intensity)
{
	int score;
	for (int d = 6; d <= pos.EmptyCount() - 10; d++)
	{
		Intensity i(d, 1.1_sigmas);
		score = PVS::Eval(pos, i);
		if (i >= intensity)
			return score;
	}

	{
		Intensity i(pos.EmptyCount(), 1.1_sigmas);
		score = PVS::Eval(pos, i);
		if (i >= intensity)
			return score;
	}
	{
		Intensity i(pos.EmptyCount(), 1.5_sigmas);
		score = PVS::Eval(pos, i);
		if (i >= intensity)
			return score;
	}

	return PVS::Eval(pos);

	//score = MTD_f(pos, { pos.EmptyCount(), 1.0_sigmas }, result.window.lower());
	//score = MTD_f(pos, { pos.EmptyCount(), 2.0_sigmas }, result.window.lower());
	return MTD_f(pos, Intensity::Exact(pos), score);
}

int IDAB::MTD_f(const Position& pos, const Intensity& intensity, int guess)
{
	int upperBound = max_score;
	int lowerBound = min_score;
	while (lowerBound < upperBound)
	{
		int beta = std::max(guess, lowerBound + 1);
		guess = PVS::Eval(pos, intensity, { beta - 1, beta });
		if (guess < beta)
			upperBound = guess;
		else
			lowerBound = guess;
	}
	return guess;
}