#include "Algorithm.h"

Result MTD::Eval(const Position& pos, OpenInterval window, Intensity intensity)
{
	return Eval(0, pos, window, intensity);
}

Result MTD::Eval(int guess, const Position& pos, OpenInterval window, Intensity intensity)
{
	// From https://en.wikipedia.org/wiki/MTD(f)
	Result result;
	while (window.lower < window.upper)
	{
		Score beta = std::max<Score>(guess, window.lower + 1);

		OpenInterval zero_window{ beta - 1, beta };
		result = alg.Eval(pos, zero_window, intensity);

		if (result.window < zero_window)
		{
			guess = result.window.upper;
			window.upper = guess;
		}
		else
		{
			guess = result.window.lower;
			window.lower = guess;
		}
	}
	return Result::Exact(guess, intensity, result.best_move);
}