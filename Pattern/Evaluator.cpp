#include "IO/IO.h"
#include "Evaluator.h"
#include "Helpers.h"
#include "DenseIndexer.h"
#include <array>
#include <cassert>
#include <iterator>

using namespace Pattern;

template <int variations>
class Simple final : public Evaluator
{
	static_assert(variations == 4 || variations == 8);

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

	float Eval(const Position& pos) const override
	{
		float sum = 0;
		for (int i = 0; i < variations; i++)
			sum += weights[i][FastIndex(pos, masks[i])];
		return sum;
	}

	std::vector<MaskAndValue> DetailedEval(const Position& pos) const override
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

	float Eval(const Position& pos) const override
	{
		float sum = 0;
		for (std::size_t i = 0; i < components.size(); i++)
			sum += components[i]->Eval(pos);
		return sum;
	}

	std::vector<MaskAndValue> DetailedEval(const Position& pos) const override
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

class GameOver final : public Evaluator
{
public:
	GameOver() noexcept = default;

	float Eval(const Position& pos) const override
	{
		return static_cast<float>(EvalGameOver(pos));
	}

	std::vector<MaskAndValue> DetailedEval(const Position& pos) const override
	{ 
		return {{~BitBoard{}, static_cast<float>(EvalGameOver(pos))}};
	}
};

class Zero final : public Evaluator
{
public:
	Zero() noexcept = default;

	float Eval(const Position& pos) const override
	{
		return 0;
	}

	std::vector<MaskAndValue> DetailedEval(const Position& pos) const override
	{ 
		return {{BitBoard{}, 0}};
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
//	//std::vector<MaskAndValue> DetailedEval(const Position& pos) const override
//	//{
//	//	return evals[pos.EmptyCount()]->DetailedEval(pos);
//	//}
//};

PatternEval::PatternEval()
{
	std::shared_ptr<Evaluator> zero_eval = std::make_shared<Zero>();
	for (auto& eval : evals)
		eval = zero_eval;
	evals[0] = std::make_unique<GameOver>();
}

PatternEval::PatternEval(const std::vector<BitBoard>& pattern, const std::vector<Pattern::Weights>& compressed)
	: PatternEval()
{
	int empty_count = 1;
	for (const auto& w : compressed)
	{
		std::shared_ptr<Evaluator> eval = CreateEvaluator(pattern, w);
		for (int i = 0; i < block_size; i++)
			evals[empty_count++] = eval;
	}
}

PatternEval DefaultPatternEval()
{
	std::vector<BitBoard> pattern = Load<BitBoard>(R"(G:\Reversi\weights\pattern.w)");
	std::vector<Pattern::Weights> weights;
	for (int i = 0; i < 8; i++)
		weights.push_back(Load<Pattern::Weights::value_type>(R"(G:\Reversi\weights\block)" + std::to_string(i) + ".w"));
	weights.push_back(weights.back());
	weights.push_back(weights.back());
	weights.push_back(weights.back());
	return { pattern, weights };
}

float PatternEval::Eval(const Position& pos) const
{
	return evals[pos.EmptyCount()]->Eval(pos);
}

std::vector<PatternEval::MaskAndValue> PatternEval::DetailedEval(const Position& pos) const
{
	return evals[pos.EmptyCount()]->DetailedEval(pos);
}

std::unique_ptr<Evaluator> Pattern::CreateEvaluator(const BitBoard pattern, const Weights& compressed)
{
	const auto indexer = CreateDenseIndexer(pattern);
	const auto full_size = pown(3, popcount(pattern));

	// Reserve memory for weights
	std::vector<Weights> weights;
	for (int i = 0; i < indexer->variations; i++)
		weights.push_back(Weights(full_size));

	// Decompress weights
	for (int i = 0; i < indexer->variations; i++)
	{
		BitBoard variation = indexer->PatternVariation(i);
		for (const Position& config : Configurations(variation))
			weights[i][FastIndex(config, variation)] = compressed[indexer->DenseIndex(config, i)];
	}

	std::vector<BitBoard> variations;
	for (int i = 0; i < indexer->variations; i++)
		variations.push_back(indexer->PatternVariation(i));

	if (indexer->variations == 4)
		return std::make_unique<Simple<4>>(std::move(variations), std::move(weights));
	return std::make_unique<Simple<8>>(std::move(variations), std::move(weights));
}

std::unique_ptr<Evaluator> Pattern::CreateEvaluator(const std::vector<BitBoard>& patterns, const Weights& compressed)
{
	std::vector<std::unique_ptr<Evaluator>> evals;
	int offset = 0;
	for (const auto& p : patterns)
	{
		auto size = CreateDenseIndexer(p)->reduced_size;
		evals.push_back(Pattern::CreateEvaluator(p, {compressed.begin() + offset, compressed.begin() + offset + size}));
		offset += size;
	}
	return std::make_unique<SumGroup>(std::move(evals));
}
