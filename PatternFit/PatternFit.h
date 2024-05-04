#pragma once
#include "Board/Board.h"
#include "Math/Math.h"
#include "Pattern/Pattern.h"
#include "Game/Game.h"
#include <array>
#include <cstdint>
#include <vector>

// TODO: Remove this?
struct PositionMultiDepthScore
{
    Position pos;
    std::array<int, 60> score_of_depth;

    PositionMultiDepthScore(Position pos) : pos(pos) { score_of_depth.fill(undefined_score); }
};

void EvaluateIteratively(
	PatternBasedEstimator& estimator,
	std::vector<ScoredPosition>&,
	Intensity intensity,
    int fitting_iterations,
    bool reevaluate);
