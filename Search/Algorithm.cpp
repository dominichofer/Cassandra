#include "Game/Game.h"
#include "Algorithm.h"
#include <chrono>

Result MTD::Eval(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	return Eval(0, pos, window, depth, confidence_level);
}

Result MTD::Eval(int guess, const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	// From https://en.wikipedia.org/wiki/MTD(f)
	auto g = Result::Exact(guess, -1, 0, Field::PS);
	while (window.lower < window.upper)
	{
		int beta = std::max<int>(g.score, window.lower + 1);

		OpenInterval zero_window{ beta - 1, beta };
		g = alg.Eval(pos, zero_window, depth, confidence_level);

		if (g.score < beta)
			window.upper = g.score;
		else
			window.lower = g.score;
	}
	return Result::Exact(g.score, depth, confidence_level, g.best_move);
}

Result IDAB::Eval(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	depth = std::min(depth, pos.EmptyCount());
	auto result = Result::Exact(0, -1, inf, Field::PS);

	// Iterative deepening
	for (int d = 1; d < depth and d <= pos.EmptyCount() - 10; d++)
		result = MTD::Eval(result.score, pos, window, d, 1.0);

	// Iterative broadening
	for (float cl : std::vector{ 1.0, 1.5 })
		if (cl < confidence_level)
			result = MTD::Eval(result.score, pos, window, depth, cl);

	return MTD::Eval(result.score, pos, window, depth, confidence_level);
}
