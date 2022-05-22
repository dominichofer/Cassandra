#include "Evaluator.h"
#include "Helpers.h"
#include "Indexer.h"
#include <stdexcept>

GLEM::GLEM(std::vector<BitBoard> pattern, std::optional<std::span<const float>> weights)
{
	for (BitBoard p : pattern)
	{
		auto dense_indexer = CreateIndexer(p);
		auto size = dense_indexer->index_space_size;
		auto variations = dense_indexer->Variations();
		auto expanded_size = pown(3, popcount(p));

		pattern_index.push_back(static_cast<int>(masks.size()));
		for (int i = 0; i < static_cast<int>(variations.size()); i++)
		{
			masks.push_back(variations[i]);
			w.emplace_back(expanded_size, 0.0f);

			if (weights.has_value())
			{
				// Decompress weights
				for (const Position& config : Configurations(variations[i]))
					w.back()[FastIndex(config, variations[i])] = weights.value()[dense_indexer->DenseIndex(config, i)];
			}

		}
		if (weights.has_value())
			weights.value() = weights.value().subspan(size);
	}
}

GLEM::GLEM(BitBoard pattern, std::optional<std::span<const float>> weights)
	: GLEM(std::vector{ pattern }, weights)
{}

float GLEM::Eval(const Position& pos) const noexcept
{
	float sum = 0.0f;
	for (std::size_t i = 0; i < masks.size(); i += 4)
	{
		sum += w[i][FastIndex(pos, masks[i])]
			+ w[i + 1][FastIndex(pos, masks[i + 1])]
			+ w[i + 2][FastIndex(pos, masks[i + 2])]
			+ w[i + 3][FastIndex(pos, masks[i + 3])];
	}
	return sum;
}

std::vector<GLEM::MaskAndValue> GLEM::DetailedEval(const Position& pos) const noexcept
{
	std::vector<MaskAndValue> ret;
	ret.reserve(masks.size());
	for (std::size_t i = 0; i < masks.size(); i++)
		ret.emplace_back(masks[i], w[i][FastIndex(pos, masks[i])]);
	return ret;
}

std::vector<float> GLEM::Weights() const
{
	std::vector<float> ret;
	for (auto index : pattern_index)
	{
		auto pattern = masks[index];
		auto dense_indexer = CreateIndexer(pattern);
		auto size = dense_indexer->index_space_size;

		auto old_size = ret.size();
		ret.resize(old_size + size);

		// Compress weights
		for (const Position& config : Configurations(pattern))
			ret[old_size + dense_indexer->DenseIndex(config, 0)] = w[index][FastIndex(config, pattern)];
	}
	return ret;
}

std::size_t GLEM::WeightsSize() const
{
	std::size_t size = 0;
	for (auto index : pattern_index)
		size += CreateIndexer(masks[index])->index_space_size;
	return size;
}

std::vector<BitBoard> GLEM::Pattern() const
{
	std::vector<BitBoard> ret;
	ret.reserve(pattern_index.size());
	for (auto index : pattern_index)
		ret.push_back(masks[index]);
	return ret;
}

AAGLEM::AAGLEM(
	std::vector<BitBoard> patterns,
	std::vector<int> block_boundaries,
	std::valarray<double> accuracy_parameters)
	: accuracy_model(std::move(accuracy_parameters))
	, block_boundaries(std::move(block_boundaries))
{
	evals.fill(std::make_shared<GLEM>(patterns));
}

AAGLEM::AAGLEM(
	std::vector<BitBoard> patterns,
	std::vector<int> block_boundaries,
	std::span<const float> weights,
	std::valarray<double> accuracy_parameters)
	: AAGLEM(std::move(patterns), std::move(block_boundaries), std::move(accuracy_parameters))
{
	for (int i = 0; i < Blocks(); i++)
	{
		SetWeights(i, weights);
		weights = weights.subspan(GetWeightsSize(i));
	}
}

AAGLEM::AAGLEM() : AAGLEM({}, { 0,65 }) {}

float AAGLEM::Eval(const Position& pos) const noexcept
{
	return evals[pos.EmptyCount()]->Eval(pos);
}

std::vector<GLEM::MaskAndValue> AAGLEM::DetailedEval(const Position& pos) const noexcept
{
	return evals[pos.EmptyCount()]->DetailedEval(pos);
}

std::vector<BitBoard> AAGLEM::Pattern() const
{
	return evals[0]->Pattern();
}

std::vector<int> AAGLEM::BlockBoundaries() const
{
	return block_boundaries;
}

std::size_t AAGLEM::Blocks() const
{
	return block_boundaries.size() - 1;
}

std::vector<HalfOpenInterval> AAGLEM::Boundaries() const
{
	std::vector<HalfOpenInterval> ret;
	for (std::size_t i = 0; i < block_boundaries.size() - 1; i++)
		ret.emplace_back(block_boundaries[i], block_boundaries[i + 1]);
	return ret;
}

HalfOpenInterval AAGLEM::Boundaries(int block) const
{
	return { block_boundaries[block], block_boundaries[block + 1] };
}

void AAGLEM::SetWeights(int block, std::span<const float> weights, std::vector<BitBoard> patterns)
{
	auto eval = std::make_shared<GLEM>(patterns, weights);
	for (int e = block_boundaries[block]; e < block_boundaries[block + 1]; e++)
		evals[e] = eval;
}

void AAGLEM::SetWeights(int block, std::span<const float> weights)
{
	SetWeights(block, weights, evals[block_boundaries[block]]->Pattern());
}

std::vector<float> AAGLEM::GetWeights(int block) const
{
	return evals[block_boundaries[block]]->Weights();
}

std::vector<float> AAGLEM::GetWeights() const
{
	std::vector<float> ret;
	for (int i = 0; i < Blocks(); i++)
	{
		auto w = GetWeights(i);
		ret.insert(ret.end(), w.begin(), w.end());
	}
	return ret;
}

std::size_t AAGLEM::GetWeightsSize(int block) const
{
	return evals[block_boundaries[block]]->WeightsSize();
}

std::size_t AAGLEM::GetWeightsSize() const
{
	std::size_t size = 0;
	for (int i = 0; i < Blocks(); i++)
		size += GetWeightsSize(i);
	return size;
}
