#include "Algorithm.h"

Result PIDAB::Eval(const Position& pos, OpenInterval window, Intensity intensity)
{
	std::vector<Intensity> intensities;
	for (int d = 1; d < intensity.depth and d <= pos.EmptyCount() - 10; d++)
		intensities.emplace_back(d, 1.0f);
	if (intensity.level >= 1.0f)
		intensities.emplace_back(intensity.depth, 1.0f);

	std::vector<Result> results(intensities.size(), Result{});
	#pragma omp parallel for
	for (int i = 0; i < intensities.size(); i++)
		results[i] = alg.Eval(pos, window, intensities[i]);

	return results.back();
}