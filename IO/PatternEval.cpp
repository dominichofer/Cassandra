#include "PatternEval.h"
#include "Core/BitBoard.h"
#include "File.h"

PatternEval DefaultPatternEval()
{
	std::vector<BitBoard> pattern = Load<BitBoard>(R"(G:\Reversi\weights\pattern.w)");
	std::vector<Pattern::Weights> weights;
	for (int i = 0; i < 8; i++)
		weights.push_back(Load<Pattern::Weights::value_type>(R"(G:\Reversi\weights\block)" + std::to_string(i) + ".w"));
	weights.push_back(weights.back()); // TODO: Remove!
	weights.push_back(weights.back()); // TODO: Remove!
	weights.push_back(weights.back()); // TODO: Remove!
	weights.push_back(weights.back()); // TODO: Remove!
	weights.push_back(weights.back()); // TODO: Remove!
	weights.push_back(weights.back()); // TODO: Remove!
	weights.push_back(weights.back()); // TODO: Remove!
	weights.push_back(weights.back()); // TODO: Remove!
	weights.push_back(weights.back()); // TODO: Remove!
	return { pattern, weights };
}