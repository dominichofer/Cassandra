#include "Evaluator.h"
#include "Helpers.h"
#include "Indexer.h"
#include <cmath>

GLEM::GLEM(BitBoard pattern)
	: GLEM(std::vector{ pattern })
{}

GLEM::GLEM(BitBoard pattern, std::span<const float> weights)
	: GLEM(std::vector{ pattern }, weights)
{}

GLEM::GLEM(std::vector<BitBoard> pattern)
{
	for (BitBoard p : pattern)
	{
		auto dense_indexer = CreateIndexer(p);
		auto variations = dense_indexer->Variations();
		auto expanded_size = pown(3, popcount(p));

		pattern_index.push_back(static_cast<int>(masks.size()));
		for (BitBoard var : variations)
		{
			masks.push_back(var);
			w.emplace_back(expanded_size, 0.0f);
		}
	}
}

GLEM::GLEM(std::vector<BitBoard> pattern, std::span<const float> weights)
	: GLEM(pattern)
{
	SetWeights(weights);
}

float GLEM::Eval(const Position& pos) const noexcept
{
	float sum = 0.0f;
	for (std::size_t i = 0; i < masks.size(); i += 4)
	{
		sum += w[i + 0][FastIndex(pos, masks[i + 0])]
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

void GLEM::SetWeights(std::span<const float> weights)
{
	int w_index = 0;
	for (BitBoard p : Pattern())
	{
		auto dense_indexer = CreateIndexer(p);
		auto variations = dense_indexer->Variations();
		auto expanded_size = pown(3, popcount(p));

		for (int i = 0; i < static_cast<int>(variations.size()); i++)
		{
			// Decompress weights
			for (Position config : Configurations(variations[i]))
				w[w_index][FastIndex(config, variations[i])] = weights[dense_indexer->DenseIndex(config, i)];
			w_index++;
		}
		weights = weights.subspan(dense_indexer->index_space_size); // offset weights span
	}
}

std::vector<float> GLEM::Weights() const
{
	std::vector<float> ret;
	for (auto index : pattern_index)
	{
		auto pattern = masks[index];
		auto dense_indexer = CreateIndexer(pattern);

		// Compress weights
		std::vector<float> dense_weights(dense_indexer->index_space_size, 0.0f);
		for (Position config : Configurations(pattern))
			dense_weights[dense_indexer->DenseIndex(config, 0)] = w[index][FastIndex(config, pattern)];

		ret.insert(ret.end(), dense_weights.begin(), dense_weights.end());
	}
	return ret;
}

std::size_t GLEM::WeightsSize() const
{
	std::size_t size = 0;
	for (int index : pattern_index)
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

AAGLEM::AAGLEM() : block_size(glems.size())
{
	glems.fill(std::make_shared<GLEM>());
}

AAGLEM::AAGLEM(
	std::vector<BitBoard> pattern,
	int block_size,
	std::vector<double> accuracy_parameters
)
	: block_size(block_size)
	, accuracy_model(accuracy_parameters)
	, pattern(pattern)
{
	for (int block = 0; block < Blocks(); block++)
	{
		auto glem = std::make_shared<GLEM>(pattern);
		for (int i = block * block_size; i < (block + 1) * block_size and i < glems.size(); i++)
			glems[i] = glem;
	}
}

AAGLEM::AAGLEM(
	std::vector<BitBoard> pattern,
	int block_size,
	std::span<const float> weights,
	std::vector<double> accuracy_parameters
)
	: block_size(block_size)
	, accuracy_model(accuracy_parameters)
	, pattern(pattern)
{
	std::size_t weights_size = GLEM{ pattern }.WeightsSize();

	for (int block = 0; block < Blocks(); block++)
	{
		SetWeights(block, weights);
		weights = weights.subspan(weights_size); // offset weights span
	}
}

float AAGLEM::Eval(const Position& pos) const noexcept
{
	return glems[pos.EmptyCount()]->Eval(pos);
}

std::vector<GLEM::MaskAndValue> AAGLEM::DetailedEval(const Position& pos) const noexcept
{
	return glems[pos.EmptyCount()]->DetailedEval(pos);
}

int AAGLEM::Blocks() const
{
	return static_cast<int>(std::ceil(static_cast<double>(glems.size()) / block_size));
}

void AAGLEM::SetWeights(int block, std::span<const float> weights)
{
	auto glem = std::make_shared<GLEM>(pattern, weights);
	for (int i = block * block_size; i < (block + 1) * block_size and i < glems.size(); i++)
		glems[i] = glem;
}

std::vector<float> AAGLEM::Weights(int block) const
{
	return glems[block * block_size]->Weights();
}

std::vector<float> AAGLEM::Weights() const
{
	std::vector<float> ret;
	for (int i = 0; i < Blocks(); i++)
	{
		auto w = Weights(i);
		ret.insert(ret.end(), w.begin(), w.end());
	}
	return ret;
}
