#include "Evaluator.h"
#include "Helpers.h"
#include "Indexer.h"
#include <cmath>

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



MultiStageScoreEstimator::MultiStageScoreEstimator(int stage_size, std::vector<uint64_t> pattern, std::span<const float> weights)
	: stage_size(stage_size)
{
	int stages = static_cast<int>(std::ceil(65.0 / stage_size));
	for (int stage = 0; stage < stages; stage++)
	{
		estimators.emplace_back(pattern, weights);
		weights = weights.subspan(ConfigurationsOfPattern(pattern)); // offset weights span
	}
}

float MultiStageScoreEstimator::Eval(const Position& pos) const noexcept
{
	int stage = static_cast<int>(pos.EmptyCount() / stage_size);
	return estimators[stage].Eval(pos);
}

std::vector<float> MultiStageScoreEstimator::Weights() const
{
	std::vector<float> ret;
	for (const ScoreEstimator& e : estimators)
	{
		auto weights = e.Weights();
		ret.insert(ret.end(), weights.begin(), weights.end());
	}
	return ret;
}

std::vector<float> MultiStageScoreEstimator::Weights(int stage) const
{
	return estimators[stage].Weights();
}



Vars AccuracyModel::Variables() const
{
	return { Var{"D"}, Var{"d"}, Var{"E"} };
}

Vars AccuracyModel::Parameters() const
{
	return { Var{"alpha"}, Var{"beta"}, Var{"gamma"}, Var{"delta"}, Var{"epsilon"} };
}

SymExp AccuracyModel::Function() const
{
	return Eval(Var{ "D" }, Var{ "d" }, Var{ "E" }, Var{ "alpha" }, Var{ "beta" }, Var{ "gamma" }, Var{ "delta" }, Var{ "epsilon" });
}

const std::vector<double>& AccuracyModel::ParameterValues() const
{
	return param_values;
}

double AccuracyModel::Eval(int D, int d, int E) const noexcept
{
	return Eval(D, d, E,
		param_values[0],
		param_values[1],
		param_values[2],
		param_values[3],
		param_values[4]);
}

double AccuracyModel::Eval(std::vector<int> values) const noexcept
{
	return Eval(values[0], values[1], values[2]);
}



PatternBasedEstimator::PatternBasedEstimator(MultiStageScoreEstimator score, AccuracyModel accuracy)
	: score(std::move(score)), accuracy(std::move(accuracy))
{}

int PatternBasedEstimator::Stages() const noexcept
{
	return score.Stages();
}

int PatternBasedEstimator::StageSize() const noexcept
{
	return score.StageSize();
}

std::vector<uint64_t> PatternBasedEstimator::Pattern() const noexcept
{
	return score.Pattern();
}

int PatternBasedEstimator::Score(const Position& pos) const noexcept
{
	return static_cast<int>(score.Eval(pos));
}

float PatternBasedEstimator::Accuracy(const Position& pos, int small_depth, int big_depth) const noexcept
{
	return accuracy.Eval(big_depth, small_depth, pos.EmptyCount());
}

std::vector<float> PatternBasedEstimator::Weights() const
{
	return score.Weights();
}

std::vector<float> PatternBasedEstimator::Weights(int stage) const
{
	return score.Weights(stage);
}
