#include "Evaluator.h"
#include "Helpers.h"
#include "Indexer.h"
#include <cmath>

std::size_t Stages(int stage_size)
{
	return static_cast<std::size_t>(std::ceil(65.0 / stage_size));
}

ScoreEstimator::ScoreEstimator(std::vector<uint64_t> pattern, std::span<const float> weights)
{
	for (uint64_t p : pattern)
	{
		auto dense_indexer = CreateIndexer(p);
		auto variations = dense_indexer->Variations();
		auto expanded_size = pown(3, std::popcount(p));
		this->pattern.push_back(dense_indexer->pattern);

		for (int i = 0; i < static_cast<int>(variations.size()); i++)
		{
			masks.push_back(variations[i]);
			w.emplace_back(expanded_size, 0.0f);
			for (Position config : Configurations(variations[i]))
				w.back()[FastIndex(config, variations[i])] = weights[dense_indexer->Indices(config)[i]];
		}
		weights = weights.subspan(dense_indexer->index_space_size); // offset weights span
	}
}

ScoreEstimator::ScoreEstimator(std::vector<uint64_t> pattern)
{
	for (uint64_t p : pattern)
	{
		auto dense_indexer = CreateIndexer(p);
		auto variations = dense_indexer->Variations();
		auto expanded_size = pown(3, std::popcount(p));
		this->pattern.push_back(dense_indexer->pattern);

		for (int i = 0; i < static_cast<int>(variations.size()); i++)
		{
			masks.push_back(variations[i]);
			w.emplace_back(expanded_size, 0.0f);
		}
	}
}

float ScoreEstimator::Eval(const Position& pos) const noexcept
{
	float sum = 0.0f;
	for (std::size_t i = 0; i < masks.size(); i++)
		sum += w[i][FastIndex(pos, masks[i])];
	return sum;
}

std::vector<float> ScoreEstimator::Weights() const
{
	std::vector<float> ret;
	std::span<const std::vector<float>> w(this->w);
	for (uint64_t p : pattern)
	{
		auto dense_indexer = CreateIndexer(p);
		auto variations = dense_indexer->Variations();

		std::vector<float> dense_weights(dense_indexer->index_space_size, 0.0f);
		for (Position config : Configurations(p))
			dense_weights[dense_indexer->Index(config)] = w[0][FastIndex(config, p)];

		ret.insert(ret.end(), dense_weights.begin(), dense_weights.end());
		w = w.subspan(variations.size()); // offset w span
	}
	return ret;
}



MultiStageScoreEstimator::MultiStageScoreEstimator(int stage_size, std::vector<uint64_t> pattern)
	: stage_size(stage_size)
{
	int stages = static_cast<int>(std::ceil(65.0 / stage_size));
	for (int i = 0; i < stages; i++)
		estimators.emplace_back(pattern);
}

float MultiStageScoreEstimator::Eval(const Position& pos) const noexcept
{
	return estimators[pos.EmptyCount() / stage_size].Eval(pos);
}



const std::vector<float>& AccuracyModel::ParameterValues() const
{
	return param_values;
}


float AccuracyModel::Eval(int D, int d, int E) const noexcept
{
	float alpha = param_values[0];
	float beta = param_values[1];
	float gamma = param_values[2];
	float delta = param_values[3];
	float epsilon = param_values[4];
	return (std::exp(alpha * d) + beta) * std::pow(static_cast<float>(D - d), gamma) * (delta * E + epsilon);
}


PatternBasedEstimator::PatternBasedEstimator(MultiStageScoreEstimator score, AccuracyModel accuracy)
	: score(std::move(score)), accuracy(std::move(accuracy))
{}

PatternBasedEstimator::PatternBasedEstimator(int stage_size, std::vector<uint64_t> pattern)
	: score({ stage_size, pattern })
{}

int PatternBasedEstimator::Stages() const noexcept
{
	return static_cast<int>(score.estimators.size());
}

int PatternBasedEstimator::StageSize() const noexcept
{
	return score.StageSize();
}

std::vector<uint64_t> PatternBasedEstimator::Pattern() const noexcept
{
	return score.Pattern();
}

float PatternBasedEstimator::Score(const Position& pos) const noexcept
{
	return score.Eval(pos);
}

float PatternBasedEstimator::Accuracy(int empty_count, int small_depth, int big_depth) const noexcept
{
	return accuracy.Eval(big_depth, small_depth, empty_count);
}
