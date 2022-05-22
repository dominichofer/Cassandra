#pragma once
#include "Core/Core.h"
#include "Math/Math.h"
#include <array>
#include <memory>
#include <span>
#include <vector>
#include <valarray>

// Generalized Linear Evaluation Model
class GLEM
{
	std::vector<BitBoard> masks;
	std::vector<std::vector<float>> w; // expanded weights
	std::vector<int> pattern_index; // index where a new pattern starts
public:
	struct MaskAndValue
	{
		BitBoard mask{};
		float value{};
	};

	GLEM(std::vector<BitBoard> pattern, std::optional<std::span<const float>> weights = std::nullopt);
	GLEM(BitBoard pattern, std::optional<std::span<const float>> weights = std::nullopt);
	GLEM() = default;

	float Eval(const Position& pos) const noexcept;
	std::vector<MaskAndValue> DetailedEval(const Position& pos) const noexcept;
	std::vector<float> Weights() const;
	std::size_t WeightsSize() const;
	std::vector<BitBoard> Pattern() const;
};

// Accuracy Model
struct AM
{
	std::valarray<double> param_values;

	AM(std::valarray<double> param_values = { -0.2, 0.6, 0.16, -0.01, 2.0 }) noexcept : param_values(param_values) {}

	::Vars Vars() const { return { Var{"D"}, Var{"d"}, Var{"E"} }; }
	::Vars Params() const { return { Var{"alpha"}, Var{"beta"}, Var{"gamma"}, Var{"delta"}, Var{"epsilon"} }; }
	SymExp Function() const { return Eval(Var{"D"}, Var{"d"}, Var{"E"}, Var{"alpha"}, Var{"beta"}, Var{"gamma"}, Var{"delta"}, Var{"epsilon"}); }

	auto Eval(auto D, auto d, auto E, auto alpha, auto beta, auto gamma, auto delta, auto epsilon) const noexcept
	{
		using std::exp;
		using std::pow;
		return (exp(alpha * d) + beta) * pow(D - d, gamma) * (delta * E + epsilon);
	}
	auto Eval(int D, int d, int E) const noexcept
	{
		return Eval(D, d, E,
			param_values[0],
			param_values[1],
			param_values[2],
			param_values[3],
			param_values[4]);
	}
};

// Accuracy Aware General Linear Evaluation Model
class AAGLEM
{
	std::array<std::shared_ptr<GLEM>, 65> evals;
	AM accuracy_model;
	std::vector<int> block_boundaries;
public:
	AAGLEM(
		std::vector<BitBoard> patterns,
		std::vector<int> block_boundaries,
		std::valarray<double> accuracy_parameters = { -0.2, 0.6, 0.16, -0.01, 2.0 }
	);
	AAGLEM(
		std::vector<BitBoard> patterns,
		std::vector<int> block_boundaries,
		std::span<const float> weights,
		std::valarray<double> accuracy_parameters = { -0.2, 0.6, 0.16, -0.01, 2.0 }
	);
	AAGLEM();

	float Eval(const Position&) const noexcept;
	std::vector<GLEM::MaskAndValue> DetailedEval(const Position&) const noexcept;

	std::vector<BitBoard> Pattern() const;
	std::vector<int> BlockBoundaries() const;
	AM& AccuracyModel() { return accuracy_model; }
	const AM& AccuracyModel() const { return accuracy_model; }
	std::size_t Blocks() const;
	std::vector<HalfOpenInterval> Boundaries() const;
	HalfOpenInterval Boundaries(int block) const;
	void SetWeights(int block, std::span<const float> weights, std::vector<BitBoard> patterns);
	void SetWeights(int block, std::span<const float> weights);
	std::vector<float> GetWeights(int block) const;
	std::vector<float> GetWeights() const;
	std::size_t GetWeightsSize(int block) const;
	std::size_t GetWeightsSize() const;

	template <typename... Args>
	auto Accuracy(Args&&... args) const noexcept
	{
		return accuracy_model.Eval(std::forward<Args>(args)...);
	}
};
