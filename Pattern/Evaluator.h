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
	const int block_size = 3;
	std::array<std::shared_ptr<Evaluator>, 65> evals;
public:
	PatternEval();
	PatternEval(const std::vector<BitBoard>& pattern, const std::vector<Pattern::Weights>& compressed);

	float Eval(const Position&) const override;
	std::vector<MaskAndValue> DetailedEval(const Position&) const override;
};
