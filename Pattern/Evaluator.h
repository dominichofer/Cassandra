#pragma once
#include "Core/Core.h"
#include "Math/Math.h"
#include <array>
#include <memory>
#include <span>
#include <vector>

// Pattern based score estimator
class ScoreEstimator
{
	std::vector<uint64_t> masks;
	std::vector<std::vector<float>> w; // expanded weights
	std::vector<uint64_t> pattern;
public:
	ScoreEstimator(std::vector<uint64_t> pattern, std::span<const float> weights);

	float Eval(const Position&) const noexcept;

	std::vector<uint64_t> Pattern() const noexcept { return pattern; }
	std::vector<float> Weights() const;
};

class MultiStageScoreEstimator
{
	int stage_size; // The max size of a stage.
	std::vector<ScoreEstimator> estimators;
public:
	MultiStageScoreEstimator(int stage_size, std::vector<uint64_t> pattern, std::span<const float> weights);

	std::size_t Stages() const noexcept { return estimators.size(); }
	int StageSize() const noexcept { return stage_size; }
	std::vector<uint64_t> Pattern() const noexcept { return estimators.front().Pattern(); }

	float Eval(const Position&) const noexcept;

	std::vector<float> Weights() const;
	std::vector<float> Weights(int stage) const;
};

class AccuracyModel
{
	std::vector<double> param_values;
public:
	AccuracyModel(std::vector<double> param_values = { -0.2, 0.6, 0.16, -0.01, 2.0 }) noexcept : param_values(param_values) {}

	Vars Variables() const;
	Vars Parameters() const;
	SymExp Function() const;
	const std::vector<double>& ParameterValues() const;

	auto Eval(auto D, auto d, auto E, auto alpha, auto beta, auto gamma, auto delta, auto epsilon) const noexcept
	{
		using std::exp;
		using std::pow;
		return (exp(alpha * d) + beta) * pow(D - d, gamma) * (delta * E + epsilon);
	}
	double Eval(int D, int d, int E) const noexcept;
	double Eval(std::vector<int> values) const noexcept;
};

// Accuracy Aware Multi Stage Score Estimator
class PatternBasedEstimator final : public Estimator
{
	// Composition
public:
	MultiStageScoreEstimator score;
	AccuracyModel accuracy;

	PatternBasedEstimator(MultiStageScoreEstimator, AccuracyModel);

	int Stages() const noexcept;
	int StageSize() const noexcept;
	std::vector<uint64_t> Pattern() const noexcept;

	int Score(const Position& pos) const noexcept override;
	float Accuracy(const Position&, int small_depth, int big_depth) const noexcept override;

	std::vector<float> Weights() const;
	std::vector<float> Weights(int stage) const;
};
