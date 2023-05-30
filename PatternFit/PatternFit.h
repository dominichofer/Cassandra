#pragma once
#include "Core/Core.h"
#include "Pattern/Pattern.h"
#include "Search/Search.h"
#include <array>
#include <span>
#include <tuple>
#include <vector>

struct PositionMultiDepthScore
{
    Position pos;
    std::array<int, 60> score_of_depth;

    PositionMultiDepthScore(Position pos) : pos(pos) { score_of_depth.fill(undefined_score); }
};

// Creates a score estimator, fitted to the given data.
ScoreEstimator CreateScoreEstimator(
    const std::vector<BitBoard>& pattern,
    const std::vector<Position>& pos,
    const std::vector<float>& score,
    int iterations = 10);

// Creates a MultiStage Score Estimator by bootstrapping from the given positions.
MSSE CreateMultiStageScoreEstimator(
    int stage_size,
    const std::vector<BitBoard>& pattern,
    const std::vector<Position>& pos,
    Intensity eval_intensity);

// Creates an accuracy model, fitted to the given data.
// Returns accuracy model and R^2.
std::pair<AM, double> CreateAccuracyModel(std::span<const PositionMultiDepthScore>);

// Create Accuracy Aware MultiStage Score Estimator by bootstrapping from the given data.
// Returns AAMSSE and R^2.
std::pair<AAMSSE, double> CreateAAMSSE(
    int stage_size,
    const std::vector<BitBoard>& pattern,
    const std::vector<Position>& train_pos,
    const std::vector<Position>& accuracy_pos,
    Intensity eval_intensity,
    int accuracy_max_depth);