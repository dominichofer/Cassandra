#include "Algorithm.h"

Result IDAB::Eval(const Position& pos, OpenInterval window, Intensity intensity)
{
	intensity.depth = std::min<int8_t>(intensity.depth, pos.EmptyCount());

	// Iterative deepening
	auto result = alg.Eval(pos, window, 0);
	for (int8_t d = 1; d < intensity.depth and d <= pos.EmptyCount() - 10; d++)
		result = alg.Eval(result.GetScore(), pos, window, { d, 1.0f });

	// Iterative broadening
	if (intensity.level >= 1.0f)
		result = alg.Eval(result.GetScore(), pos, window, { intensity.depth, 1.0 });
	return alg.Eval(result.GetScore(), pos, window, intensity);
}