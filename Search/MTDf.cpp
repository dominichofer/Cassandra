#include "MTDf.h"
#include <chrono>

ResultTimeNodes MTD::Eval(int guess, const Position& pos)
{
	return Eval(guess, pos, { -inf_score, +inf_score }, pos.EmptyCount(), std::numeric_limits<float>::infinity());
}

ResultTimeNodes MTD::Eval(int guess, const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	nodes = 0;
	auto start = std::chrono::high_resolution_clock::now();
	auto result = Eval_N(guess, pos, window, depth, confidence_level);
	auto time = std::chrono::high_resolution_clock::now() - start;
	return { result, time, nodes };
}

Result MTD::Eval_N(int guess, const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	// From https://en.wikipedia.org/wiki/MTD(f)
	auto g = Result::Exact(guess, -1, 0, Field::PS);
	while (window.lower < window.upper)
	{
		int beta = std::max<int>(g.score, window.lower + 1);

		OpenInterval zero_window(beta - 1, beta);
		g = PVS_N(pos, zero_window, depth, confidence_level);

		if (g.score < beta)
			window.upper = g.score;
		else
			window.lower = g.score;
	}
	return g;
}
