#pragma once
#include "Core/Position.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace Pattern
{
	class Evaluator
	{
	public:
		Evaluator(BitBoard pattern) : Pattern(pattern) {}

		virtual float Eval(const Position&) const = 0;

		const BitBoard Pattern;
	};

	using Weights = std::vector<float>;

	std::unique_ptr<Evaluator> CreateEvaluator(BitBoard pattern, std::vector<Weights>);
	std::unique_ptr<Evaluator> CreateEvaluator(BitBoard pattern, const Weights& compressed);
}