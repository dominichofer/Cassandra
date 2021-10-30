#pragma once
#include "Core/Core.h"
#include <array>
#include <memory>
#include <vector>
#include <span>

class Position;

namespace Pattern
{
	using Weights = std::vector<float>;

	// Interface
	class Evaluator
	{
	public:
		struct MaskAndValue
		{
			BitBoard mask{};
			float value{};
		};

		virtual float Eval(const Position&) const noexcept = 0;
		virtual std::vector<MaskAndValue> DetailedEval(const Position&) const noexcept = 0;
	};

	std::unique_ptr<Evaluator> CreateEvaluator(BitBoard pattern, std::span<const float> compressed_weights);
	std::unique_ptr<Evaluator> CreateEvaluator(const std::vector<BitBoard>& pattern, std::span<const float> compressed_weights);
}

// Accuracy Aware General Linear Evaluation Model
class AAGLEM
{
	std::array<std::shared_ptr<Pattern::Evaluator>, 65> evals;
	float alpha, beta, gamma, delta, epsilon;
	std::vector<BitBoard> pattern;
public:
	const int block_size;

	AAGLEM() noexcept : block_size(0) {} // TODO: Remove!
	AAGLEM(
		int block_size,
		std::vector<BitBoard> patterns,
		const std::vector<std::vector<float>>& compressed_weights,
		const std::vector<float>& accuracy_parameters
	);

	const std::vector<BitBoard>& Pattern() const noexcept { return pattern; }

	float Eval(const Position& pos) const noexcept { return evals[pos.EmptyCount()]->Eval(pos); }
	auto DetailedEval(const Position& pos) const noexcept { return evals[pos.EmptyCount()]->DetailedEval(pos); }

	auto Accuracy(auto D, auto d, auto E, auto alpha, auto beta, auto gamma, auto delta, auto epsilon) const noexcept
	{
		using std::exp;
		using std::pow;
		return (exp(alpha * d) + beta) * pow(D - d, gamma) * (delta * E + epsilon);
	}
	float Accuracy(int D, int d, int E) const noexcept { return Accuracy(D, d, E, alpha, beta, gamma, delta, epsilon); }
};
