#include "PatternEval.h"
#include "Core/BitBoard.h"
#include "File.h"

void SavePattern(const std::vector<BitBoard>& vec)
{
	Save(R"(G:\Reversi\weights\pattern.w)", vec);
}

std::vector<BitBoard> LoadPattern()
{
	return Load<std::vector<BitBoard>>(R"(G:\Reversi\weights\pattern.w)");
}

void SaveModelParameters(const std::vector<float>& vec)
{
	Save(R"(G:\Reversi\weights\model_params.w)", vec);
}

std::vector<float> LoadModelParameters()
{
	return Load<std::vector<float>>(R"(G:\Reversi\weights\model_params.w)");
}

void SaveWeights(const Pattern::Weights& weights, int block)
{
	Save(R"(G:\Reversi\weights\block)" + std::to_string(block) + ".w", weights);
}

Pattern::Weights LoadWeights(int block)
{
	return Load<Pattern::Weights>(R"(G:\Reversi\weights\block)" + std::to_string(block) + ".w");
}

PatternEval DefaultPatternEval()
{
	std::vector<BitBoard> pattern = LoadPattern();
	std::vector<Pattern::Weights> weights;
	for (int i = 0; i < 5; i++)
		weights.push_back(LoadWeights(i));
	weights.push_back(weights.back());
	std::vector<float> params = LoadModelParameters();
	return { pattern, weights, params };
}