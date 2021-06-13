#include "Core/Core.h"
#include "Algorithm.h"
#include <iostream>

using namespace Search;

int IDAB::MTD_f(const Position& pos, const Intensity& intensity, int guess)
{
	int upperBound = max_score;
	int lowerBound = min_score;
	while (lowerBound < upperBound)
	{
		int beta = std::max(guess, lowerBound + 1);
		auto result = PVS::Eval(pos, Request(intensity, { beta - 1, beta }));
		if (result.window < beta)
		{
			guess = result.window.upper();
			upperBound = guess;
		}
		else
		{
			guess = result.window.lower();
			lowerBound = guess;
		}
	}
	return guess;
}

Result IDAB::Eval(const Position& pos, const Request& request)
{
	Result result;
	for (int d = 6; d <= pos.EmptyCount() - 10; d++)
	{
		result = PVS::Eval(pos, { {d, 1.1_sigmas}, OpenInterval::Whole() });
		if (result.intensity >= request.intensity)
			return result;
	}

	result = PVS::Eval(pos, { {pos.EmptyCount(), 1.1_sigmas}, OpenInterval::Whole() });
	if (result.intensity >= request.intensity)
		return result;

	//if (pos.EmptyCount() > 20)
	//{
		result = PVS::Eval(pos, { {pos.EmptyCount(), 1.5_sigmas}, OpenInterval::Whole() });
		if (result.intensity >= request.intensity)
			return result;
	//}
	//if (pos.EmptyCount() > 24)
	//{
	//	result = PVS::Eval(pos, { {pos.EmptyCount(), 2.0_sigmas}, OpenInterval::Whole() });
	//	if (result.intensity >= request.intensity)
	//		return result;
	//}

	return PVS::Eval(pos, Request::Exact(pos));

	//score = MTD_f(pos, { pos.EmptyCount(), 1.0_sigmas }, result.window.lower());
	//score = MTD_f(pos, { pos.EmptyCount(), 2.0_sigmas }, result.window.lower());
	int score = MTD_f(pos, Intensity::Exact(pos), result.Score());
	return Result::Exact(pos, score);
}