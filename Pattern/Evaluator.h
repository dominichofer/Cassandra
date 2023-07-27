#pragma once
#include "Board/Board.h"
#include "Math/Math.h"
#include "Search/Search.h"
#include <array>
#include <memory>
#include <span>
#include <vector>

std::size_t Stages(int stage_size);

// Pattern based score estimator
class ScoreEstimator
{
	std::vector<uint64_t> masks;
	std::vector<std::vector<float>> w; // expanded weights
	std::vector<uint64_t> pattern;
public:
	ScoreEstimator(std::vector<uint64_t> pattern, std::span<const float> weights);
	ScoreEstimator(std::vector<uint64_t> pattern);

	float Eval(const Position&) const noexcept;

	std::vector<uint64_t> Pattern() const noexcept { return pattern; }
	std::vector<float> Weights() const;
};

class MultiStageScoreEstimator
{
	std::size_t stage_size; // The max size of a stage.
public:
	std::vector<ScoreEstimator> estimators;

	MultiStageScoreEstimator(std::size_t stage_size, std::vector<uint64_t> pattern);

	std::size_t StageSize() const noexcept { return stage_size; }
	std::vector<uint64_t> Pattern() const noexcept { return estimators.front().Pattern(); }

	float Eval(const Position&) const noexcept;
};

class AccuracyModel
{
	std::vector<float> param_values;
public:
	AccuracyModel(std::vector<float> param_values = { -0.16594754, 0.95554334, 0.27973529, -0.01616647, 1.28078789 }) noexcept : param_values(param_values) {}

	const std::vector<float>& ParameterValues() const;

	float Eval(int D, int d, int E) const noexcept;
};

class PatternBasedEstimator final : public Estimator
{
	// Composition
public:
	MultiStageScoreEstimator score;
	AccuracyModel accuracy;

	PatternBasedEstimator(MultiStageScoreEstimator, AccuracyModel);
	PatternBasedEstimator(std::size_t stage_size, std::vector<uint64_t> pattern);

	std::size_t Stages() const noexcept;
	std::size_t StageSize() const noexcept;
	std::vector<uint64_t> Pattern() const noexcept;

	float Score(const Position& pos) const noexcept override;
	float Accuracy(int empty_count, int small_depth, int big_depth) const noexcept override;
};
