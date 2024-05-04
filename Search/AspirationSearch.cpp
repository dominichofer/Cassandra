#include "Algorithm.h"

Result AspirationSearch::Eval(const Position& pos, OpenInterval window, Intensity intensity)
{
	return Eval(0, pos, window, intensity);
}

Result AspirationSearch::Eval(int guess, const Position& pos, OpenInterval window, Intensity intensity)
{
	int low = 1;
	int high = 1;
	Result result;
	while (true)
	{
		OpenInterval guess_window{ guess - low, guess + high };
		OpenInterval search_window;
		if (guess_window < window)
			search_window = OpenInterval{ window.lower, window.lower + 1 };
		else if (guess_window > window)
			search_window = OpenInterval{ window.upper - 1, window.upper };
		else
			search_window = Intersection(window, guess_window);

		result = alg.Eval(pos, search_window, intensity);

		if (not result.window.Overlaps(window))
			return result;

		if (result.window < search_window and low > 0)
		{
			guess = result.window.upper;
			low *= 2;
			high = 0;
		}
		else if (result.window > search_window and high > 0)
		{
			guess = result.window.lower;
			low = 0;
			high *= 2;
		}
		else
			return result;
	}
}