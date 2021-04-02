#include "Core/Core.h"
#include "Algorithm.h"
#include <iostream>

using namespace Search;

Result IDAB::Eval(const Position& pos, const Request& request)
{
	Result result = PVS::Eval(pos, Request::Certain(0));
	for (int d = 5; d < pos.EmptyCount() - 10; d++)
	{
		result = PVS::Eval(pos, { {d, 1.0_sigmas}, OpenInterval::Whole() });
		if (result.intensity >= request.intensity)
			return result;
	}

	result = PVS::Eval(pos, { {pos.EmptyCount(), 1.0_sigmas}, OpenInterval::Whole() });
	if (result.intensity >= request.intensity)
		return result;

	result = PVS::Eval(pos, { {pos.EmptyCount(), 2.0_sigmas}, OpenInterval::Whole() });
	if (result.intensity >= request.intensity)
		return result;

	//result = PVS::Eval(pos, { {pos.EmptyCount(), 3.0_sigmas}, OpenInterval::Whole() });
	//if (result.intensity >= request.intensity)
	//	return result;

	//int guess = result.window.lower();
	//result = PVS::Eval(pos, Request::Certain(pos.EmptyCount(), { guess-1, guess+1 }));
	//if (result.window.IsSingleton())
	//	return result;
	//if (result.window < guess)
	//{
	//	guess = result.window.upper();
	//	return PVS::Eval(pos, Request::Certain(pos.EmptyCount(), { min_score, guess+1 }));
	//}
	//else if (result.window > guess)
	//{
	//	guess = result.window.lower();
	//	return PVS::Eval(pos, Request::Certain(pos.EmptyCount(), { guess-1, max_score }));
	//}
	//else
	//	return Result::Exact(pos, -64);

	int guess = result.window.lower();
	int upperBound = max_score;
	int lowerBound = min_score;
	while (lowerBound < upperBound)
	{
		int beta = std::max(guess, lowerBound + 1);
		result = PVS::Eval(pos, Request::Certain(pos.EmptyCount(), { beta-1, beta }));
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
	return Result::Exact(pos, guess);
}