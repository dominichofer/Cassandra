#pragma once
#include "Core/Core.h"
#include "Math/Math.h"
#include <array>
#include <memory>
#include <span>
#include <vector>
#include <vector>

// Generalized Linear Evaluation Model
class GLEM
{
	std::vector<BitBoard> masks;
	std::vector<std::vector<float>> w; // expanded weights
	std::vector<int> pattern_index; // indices in 'masks' which form the patterns. The other masks are symmetric variations.
public:
	struct MaskAndValue
	{
		BitBoard mask{};
		float value{};
	};

	GLEM() = default;
	GLEM(BitBoard pattern);
	GLEM(BitBoard pattern, std::span<const float> weights);
	GLEM(std::vector<BitBoard> pattern);
	GLEM(std::vector<BitBoard> pattern, std::span<const float> weights);

	float Eval(const Position& pos) const noexcept;
	std::vector<MaskAndValue> DetailedEval(const Position& pos) const noexcept;

	void SetWeights(std::span<const float>);
	std::vector<float> Weights() const;
	std::size_t WeightsSize() const;
	std::vector<BitBoard> Pattern() const;
};

// Accuracy Model
struct AM
{
	std::vector<double> param_values;

	AM(std::vector<double> param_values = { -0.2, 0.6, 0.16, -0.01, 2.0 }) noexcept : param_values(param_values) {}

	Vars Variables() const { return { Var{"D"}, Var{"d"}, Var{"E"} }; }
	Vars Parameters() const { return { Var{"alpha"}, Var{"beta"}, Var{"gamma"}, Var{"delta"}, Var{"epsilon"} }; }
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
	auto Eval(std::vector<int> values) const noexcept
	{
		return Eval(values[0], values[1], values[2]);
	}
};

// Accuracy Aware General Linear Evaluation Model
class AAGLEM
{
	std::array<std::shared_ptr<GLEM>, 65> glems;
	int block_size;
	AM accuracy_model;
	std::vector<BitBoard> pattern;
public:
	AAGLEM();
	AAGLEM(
		std::vector<BitBoard> pattern,
		int block_size,
		std::vector<double> accuracy_parameters = { -0.2, 0.6, 0.16, -0.01, 2.0 }
	);
	AAGLEM(
		std::vector<BitBoard> pattern,
		int block_size,
		std::span<const float> weights,
		std::vector<double> accuracy_parameters = { -0.2, 0.6, 0.16, -0.01, 2.0 }
	);

	float Eval(const Position&) const noexcept;
	std::vector<GLEM::MaskAndValue> DetailedEval(const Position&) const noexcept;

	GLEM& Evaluator(int block) { return *glems[block * block_size]; }
	const GLEM& Evaluator(int block) const { return *glems[block * block_size]; }

	AM& AccuracyModel() { return accuracy_model; }
	const AM& AccuracyModel() const { return accuracy_model; }

	std::vector<BitBoard> Pattern() const { return pattern; }
	int BlockSize() const { return block_size; }
	int Blocks() const;
	std::vector<double> AccuracyParameters() const { return accuracy_model.param_values; }

	void SetWeights(int block, std::span<const float> weights);
	std::vector<float> Weights(int block) const;
	std::vector<float> Weights() const;

	template <typename... Args>
	auto Accuracy(Args&&... args) const noexcept
	{
		return accuracy_model.Eval(std::forward<Args>(args)...);
	}
};
