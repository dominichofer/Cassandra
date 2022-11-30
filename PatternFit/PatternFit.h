#pragma once
#include "Core/Core.h"
#include "Math/Math.h"
#include "Pattern/Pattern.h"
#include "Search/Search.h"
#include <array>
#include <vector>

struct PositionMultiDepthScore
{
    Position pos;
    std::array<int, 60> score_of_depth;

    PositionMultiDepthScore(Position pos) : pos(pos) { score_of_depth.fill(undefined_score); }
};

void Fit(GLEM& model, const std::vector<PosScore>& data, int iterations = 10);

void Fit(GLEM& model, std::vector<Position>::const_iterator pos_begin, std::vector<Position>::const_iterator pos_end, const std::vector<float>& score, int iterations = 10);
void Fit(GLEM& model, const std::vector<Position>& pos, const std::vector<float>& score, int iterations = 10);

// Returns R^2
double Fit(AM& model, const std::vector<PositionMultiDepthScore>& data);

// Returns R^2
double Fit(AAGLEM& model,
    const std::vector<Game>& train_games, int exact_blocks, Intensity train_eval,
    const std::vector<Game>& accuracy_fit_games, int accuracy_fit_eval_all_depth_till, int accuracy_fit_eval_max_depth);