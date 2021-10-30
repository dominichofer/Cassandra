#include "Evaluator.h"
#include "Helpers.h"
#include "DenseIndexer.h"
#include <cassert>
#include <iterator>

using namespace Pattern;

template <int variations>
class Simple final : public Evaluator
{
	static_assert(variations == 4 or variations == 8);

	std::vector<BitBoard> masks;
	std::vector<Weights> weights;
public:
	// masks: The symmetric variations of the pattern.
	// weights: The corresponding weights.
	Simple(std::vector<BitBoard> masks, std::vector<Weights> weights)
		: masks(std::move(masks))
		, weights(std::move(weights))
	{
		assert(this->masks.size() == variations);
		assert(this->weights.size() == variations);
	}

	float Eval(const Position& pos) const noexcept override
	{
		float sum = 0;
		for (int i = 0; i < variations; i++)
			sum += weights[i][FastIndex(pos, masks[i])];
		return sum;
	}

	std::vector<MaskAndValue> DetailedEval(const Position& pos) const noexcept override
	{
		std::vector<MaskAndValue> ret;
		for (int i = 0; i < variations; i++)
			ret.emplace_back(masks[i], weights[i][FastIndex(pos, masks[i])]);
		return ret;
	}
};

// Composite
class SumGroup final : public Evaluator
{
	std::vector<std::unique_ptr<Evaluator>> components;
public:
	SumGroup(std::vector<std::unique_ptr<Evaluator>> evaluators)
		: components(std::move(evaluators))
	{}

	float Eval(const Position& pos) const noexcept override
	{
		float sum = 0;
		for (std::size_t i = 0; i < components.size(); i++)
			sum += components[i]->Eval(pos);
		return sum;
	}

	std::vector<MaskAndValue> DetailedEval(const Position& pos) const noexcept override
	{
		std::vector<MaskAndValue> ret;
		for (std::size_t i = 0; i < components.size(); i++)
		{
			auto tmp = components[i]->DetailedEval(pos);
			ret.insert(ret.end(), tmp.begin(), tmp.end());
		}
		return ret;
	}
};

//class EmptyCountSwitch final : public Evaluator
//{
//	std::array<std::shared_ptr<Evaluator>, 65> evals;
//public:
//	EmptyCountSwitch(std::array<std::shared_ptr<Evaluator>, 65> evaluators)
//		: evals(std::move(evaluators))
//	{}
//
//	float Eval(const Position& pos) const override
//	{
//		return evals[pos.EmptyCount()]->Eval(pos);
//	}
//
//	std::vector<MaskAndValue> DetailedEval(const Position& pos) const override
//	{
//		return evals[pos.EmptyCount()]->DetailedEval(pos);
//	}
//};

class GameOver final : public Evaluator
{
public:
	GameOver() noexcept = default;

	float Eval(const Position& pos) const noexcept override
	{
		return static_cast<float>(EvalGameOver(pos));
	}

	std::vector<MaskAndValue> DetailedEval(const Position& pos) const noexcept override
	{ 
		return { {~BitBoard{}, GameOver::Eval(pos)} };
	}
};

class Zero final : public Evaluator
{
public:
	Zero() noexcept = default;

	float Eval(const Position&) const noexcept override
	{
		return 0;
	}

	std::vector<MaskAndValue> DetailedEval(const Position&) const noexcept override
	{ 
		return { {BitBoard{}, 0} };
	}
};

std::unique_ptr<Evaluator> Pattern::CreateEvaluator(const BitBoard pattern, std::span<const float> compressed_weights)
{
	const auto indexer = CreateDenseIndexer(pattern);
	const auto full_size = pown(3, popcount(pattern));

	// Reserve memory for weights
	std::vector<Weights> weights(indexer->variations, Weights(full_size));

	// Decompress weights
	for (int i = 0; i < indexer->variations; i++)
	{
		BitBoard variation = indexer->PatternVariation(i);
		for (const Position& config : Configurations(variation))
			weights[i][FastIndex(config, variation)] = compressed_weights[indexer->DenseIndex(config, i)];
	}

	std::vector<BitBoard> variations;
	for (int i = 0; i < indexer->variations; i++)
		variations.push_back(indexer->PatternVariation(i));

	if (indexer->variations == 4)
		return std::make_unique<Simple<4>>(std::move(variations), std::move(weights));
	return std::make_unique<Simple<8>>(std::move(variations), std::move(weights));
}

std::unique_ptr<Evaluator> Pattern::CreateEvaluator(const std::vector<BitBoard>& patterns, std::span<const float> compressed_weights)
{
	std::vector<std::unique_ptr<Evaluator>> evaluators;
	evaluators.reserve(patterns.size());
	auto begin = compressed_weights.begin();
	for (BitBoard p : patterns)
	{
		auto size = CreateDenseIndexer(p)->reduced_size;
		evaluators.push_back(CreateEvaluator(p, { begin, begin + size }));
		begin += size;
	}
	return std::make_unique<SumGroup>(std::move(evaluators));
}

AAGLEM::AAGLEM(
	int block_size,
	std::vector<BitBoard> pattern,
	const std::vector<std::vector<float>>& compressed,
	const std::vector<float>& accuracy_parameters)
	: block_size(block_size), pattern(std::move(pattern))
{
	std::shared_ptr<Evaluator> zero_eval = std::make_shared<Zero>();
	for (auto& e : evals)
		e = zero_eval;
	evals[0] = std::make_unique<GameOver>();

	alpha = accuracy_parameters[0];
	beta = accuracy_parameters[1];
	gamma = accuracy_parameters[2];
	delta = accuracy_parameters[3];
	epsilon = accuracy_parameters[4];

	int empty_count = 1;
	for (const auto& w : compressed)
	{
		std::shared_ptr<Evaluator> eval = CreateEvaluator(this->pattern, w);
		for (int i = 0; i < block_size; i++)
			evals[empty_count++] = eval;
	}
}