#pragma once
#include "Board/Board.h"
#include "Math/Math.h"
#include "Pattern/Pattern.h"
#include "Search/Search.h"
#include <array>
#include <cstdint>
#include <vector>

struct PositionMultiDepthScore
{
    Position pos;
    std::array<int, 60> score_of_depth;

    PositionMultiDepthScore(Position pos) : pos(pos) { score_of_depth.fill(undefined_score); }
};

Vector FitWeights(
    const std::vector<uint64_t>& pattern,
    const std::vector<Position>& pos,
    const Vector& score,
    int iterations);

void ImproveScoreEstimator(
    PatternBasedEstimator& estimator,
    const std::vector<Position>& pos,
    int depth, float confidence_level,
    int fitting_iterations);
