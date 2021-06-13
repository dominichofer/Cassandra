#pragma once
#include "Core/BitBoard.h"
#include <array>
#include <memory>
#include <vector>

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

		virtual float Eval(const Position&) const = 0;
		virtual std::vector<MaskAndValue> DetailedEval(const Position&) const = 0;
	};


	std::unique_ptr<Evaluator> CreateEvaluator(BitBoard pattern, const Weights& compressed);
	std::unique_ptr<Evaluator> CreateEvaluator(const std::vector<BitBoard>& pattern, const Weights& compressed);
}

class PatternEval final : public Pattern::Evaluator
{
	float alpha, beta, gamma, delta, epsilon;
	std::array<std::shared_ptr<Evaluator>, 65> evals;
public:
	static const int block_size = 10;

	PatternEval();
	PatternEval(
		const std::vector<BitBoard>& pattern,
		const std::vector<Pattern::Weights>& compressed,
		const std::vector<float>& accuracy_parameters
	);

	float Eval(const Position&) const override;
	std::vector<MaskAndValue> DetailedEval(const Position&) const override;

	float EvalAccuracy(int d, int D, int E) const;
};
