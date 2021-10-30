#pragma once
#include "Pattern/Evaluator.h"

void SavePattern(const std::vector<BitBoard>&);
std::vector<BitBoard> LoadPattern();

void SaveModelParameters(const std::vector<float>&);
std::vector<float> LoadModelParameters();

void SaveWeights(const Pattern::Weights&, int block);
Pattern::Weights LoadWeights(int block);

AAGLEM DefaultPatternEval();