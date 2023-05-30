#pragma once
#include "Core/Core.h"
#include "Math/Math.h"
#include <array>
#include <memory>
#include <span>
#include <vector>

struct MaskAndValue
{
	BitBoard mask{};
	float value{};
};

class ScoreEstimator
{
	std::vector<BitBoard> masks;
	std::vector<std::vector<float>> w; // expanded weights
	std::vector<BitBoard> pattern;
public:
	ScoreEstimator() = default;
	ScoreEstimator(BitBoard pattern);
	ScoreEstimator(BitBoard pattern, std::span<const float> weights);
	ScoreEstimator(std::vector<BitBoard> pattern);
	ScoreEstimator(std::vector<BitBoard> pattern, std::span<const float> weights);

	std::vector<BitBoard> Pattern() const noexcept { return pattern; }

	float Eval(Position pos) const noexcept;
	std::vector<MaskAndValue> DetailedEval(Position pos) const noexcept;

	std::size_t WeightsSize() const;
	std::vector<float> Weights() const;
	void Weights(std::span<const float> weights);
};

// MultiStage Score Estimator
class MSSE
{
	int stage_size;
	std::vector<ScoreEstimator> estimators;
public:
	MSSE() = default;
	MSSE(int stage_size, BitBoard pattern);
	MSSE(int stage_size, BitBoard pattern, std::span<const float> weights);
	MSSE(int stage_size, std::vector<BitBoard> pattern);
	MSSE(int stage_size, std::vector<BitBoard> pattern, std::span<const float> weights);

	int Stages() const noexcept;
	int StageSize() const noexcept;
	std::vector<BitBoard> Pattern() const noexcept;

	float Eval(Position pos) const noexcept;
	std::vector<MaskAndValue> DetailedEval(Position pos) const noexcept;

	std::size_t WeightsSize() const;
	std::vector<float> Weights() const;
	std::vector<float> Weights(int stage) const;
	void Weights(std::span<const float> weights);
	void Weights(int stage, std::span<const float> weights);
};

// Accuracy Model
class AM
{
	std::vector<double> param_values;
public:
	AM(std::vector<double> param_values = { -0.2, 0.6, 0.16, -0.01, 2.0 }) noexcept : param_values(param_values) {}

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

// Accuracy Aware MultiStage Score Estimator
struct AAMSSE
{
	// Composition

	MSSE score_estimator;
	AM accuracy_estimator;

	AAMSSE() = default;
	AAMSSE(MSSE score_estimator, AM accuracy_estimator);
	AAMSSE(int stage_size, std::vector<BitBoard> pattern);

	int Stages() const noexcept;
	int StageSize() const noexcept;
	std::vector<BitBoard> Pattern() const noexcept;

	float Score(Position pos) const noexcept;
	std::vector<MaskAndValue> DetailedScore(Position pos) const noexcept;
	float Accuracy(int D, int d, int E) const noexcept;

	std::size_t WeightsSize() const;
	std::vector<float> Weights() const;
	std::vector<float> Weights(int stage) const;
	void Weights(int stage, std::span<const float> weights);
};

