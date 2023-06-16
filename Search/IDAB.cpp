#include "IDAB.h"

ResultTimeNodes IDAB::Eval(const Position& pos)
{
	return Eval(pos, { -inf_score, +inf_score }, pos.EmptyCount(), std::numeric_limits<float>::infinity());
}

ResultTimeNodes IDAB::Eval(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	nodes = 0;
	auto start = std::chrono::high_resolution_clock::now();
	auto result = Eval_N(pos, window, depth, confidence_level);
	auto time = std::chrono::high_resolution_clock::now() - start;
	return { result, time, nodes };
}

Result IDAB::Eval_N(const Position& pos, OpenInterval window, int depth, float confidence_level)
{
	auto result = Result::Exact(0, -1, 0, Field::PS);

	// Iterative deepening
	for (int d = 0; d <= 5; d++)
		result = PVS::PVS_N(pos, window, d, confidence_level);
	//for (int d = 6; d < depth and d <= pos.EmptyCount() - 10; d++)
	//	result = PVS::PVS_N(pos, window, depth, 1.1);

	// Iterative broadening
	//for (float cl : std::vector{ 1.1 })
	//	if (cl < confidence_level)
	//		result = PVS::PVS_N(pos, window, depth, cl);

	result = PVS::PVS_N(pos, window, depth, 1.1);
	result = MTD::Eval_N(result.score, pos, window, depth, confidence_level);
	return result;
}
