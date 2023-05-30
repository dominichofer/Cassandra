#include "Evaluator.h"
#include "Helpers.h"
#include "Indexer.h"
#include <cmath>

ScoreEstimator::ScoreEstimator(BitBoard pattern)
	: ScoreEstimator(std::vector{ pattern })
{}

ScoreEstimator::ScoreEstimator(BitBoard pattern, std::span<const float> weights)
	: ScoreEstimator(std::vector{ pattern }, weights)
{}

ScoreEstimator::ScoreEstimator(std::vector<BitBoard> pattern)
{
	for (BitBoard p : pattern)
	{
		auto dense_indexer = CreateIndexer(p);
		auto variations = dense_indexer->Variations();
		auto expanded_size = pown(3, popcount(p));

		for (BitBoard v : variations)
		{
			masks.push_back(v);
			w.emplace_back(expanded_size, 0.0f);
		}
		this->pattern.push_back(dense_indexer->pattern);
	}
}

ScoreEstimator::ScoreEstimator(std::vector<BitBoard> pattern, std::span<const float> weights)
	: ScoreEstimator(pattern)
{
	Weights(weights);
}

float ScoreEstimator::Eval(Position pos) const noexcept
{
	float sum = 0.0f;
	for (std::size_t i = 0; i < masks.size(); i++)
		sum += w[i][FastIndex(pos, masks[i])];
	return sum;
}

std::vector<MaskAndValue> ScoreEstimator::DetailedEval(Position pos) const noexcept
{
	std::vector<MaskAndValue> ret;
	ret.reserve(masks.size());
	for (std::size_t i = 0; i < masks.size(); i++)
		ret.emplace_back(masks[i], w[i][FastIndex(pos, masks[i])]);
	return ret;
}

std::size_t ScoreEstimator::WeightsSize() const
{
	std::size_t size = 0;
	for (BitBoard p : pattern)
		size += CreateIndexer(p)->index_space_size;
	return size;
}

void ScoreEstimator::Weights(std::span<const float> weights)
{
	std::span<std::vector<float>> w(this->w);
	for (BitBoard p : pattern)
	{
		auto dense_indexer = CreateIndexer(p);
		auto variations = dense_indexer->Variations();

		for (int i = 0; i < static_cast<int>(variations.size()); i++)
			for (Position config : Configurations(variations[i]))
				w[i][FastIndex(config, variations[i])] = weights[dense_indexer->Indices(config)[i]];

		w = w.subspan(variations.size()); // offset w span
		weights = weights.subspan(dense_indexer->index_space_size); // offset weights span
	}
}

std::vector<float> ScoreEstimator::Weights() const
{
	std::vector<float> ret;
	std::span<const std::vector<float>> w(this->w);
	for (BitBoard p : pattern)
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



MSSE::MSSE(int stage_size, BitBoard pattern)
	: MSSE(stage_size, std::vector{ pattern })
{}

MSSE::MSSE(int stage_size, BitBoard pattern, std::span<const float> weights)
	: MSSE(stage_size, std::vector{ pattern }, weights)
{}

MSSE::MSSE(int stage_size, std::vector<BitBoard> pattern)
	: stage_size(stage_size)
{
	int stages = static_cast<int>(std::ceil(65.0 / stage_size));
	estimators = std::vector<ScoreEstimator>(stages, ScoreEstimator(pattern));
}

MSSE::MSSE(int stage_size, std::vector<BitBoard> pattern, std::span<const float> weights)
	: MSSE(stage_size, pattern)
{
	Weights(weights);
}

int MSSE::Stages() const noexcept
{
	return estimators.size();
}

int MSSE::StageSize() const noexcept
{
	return stage_size;
}

std::vector<BitBoard> MSSE::Pattern() const noexcept
{
	return estimators.front().Pattern();
}

float MSSE::Eval(Position pos) const noexcept
{
	int stage = static_cast<int>(pos.EmptyCount() / stage_size);
	return estimators[stage].Eval(pos);
}

std::vector<MaskAndValue> MSSE::DetailedEval(Position pos) const noexcept
{
	int stage = pos.EmptyCount() / stage_size;
	return estimators[stage].DetailedEval(pos);
}

std::size_t MSSE::WeightsSize() const
{
	std::size_t size = 0;
	for (const ScoreEstimator& e : estimators)
		size += e.WeightsSize();
	return size;
}

std::vector<float> MSSE::Weights() const
{
	std::vector<float> ret;
	for (const ScoreEstimator& e : estimators)
	{
		auto weights = e.Weights();
		ret.insert(ret.end(), weights.begin(), weights.end());
	}
	return ret;
}

std::vector<float> MSSE::Weights(int stage) const
{
	return estimators[stage].Weights();
}

void MSSE::Weights(std::span<const float> weights)
{
	for (ScoreEstimator& e : estimators)
	{
		e.Weights(weights);
		weights = weights.subspan(e.WeightsSize()); // offset weights span
	}
}

void MSSE::Weights(int stage, std::span<const float> weights)
{
	estimators[stage].Weights(weights);
}



Vars AM::Variables() const
{
	return { Var{"D"}, Var{"d"}, Var{"E"} };
}

Vars AM::Parameters() const
{
	return { Var{"alpha"}, Var{"beta"}, Var{"gamma"}, Var{"delta"}, Var{"epsilon"} };
}

SymExp AM::Function() const
{
	return Eval(Var{ "D" }, Var{ "d" }, Var{ "E" }, Var{ "alpha" }, Var{ "beta" }, Var{ "gamma" }, Var{ "delta" }, Var{ "epsilon" });
}

const std::vector<double>& AM::ParameterValues() const
{
	return param_values;
}

double AM::Eval(int D, int d, int E) const noexcept
{
	return Eval(D, d, E,
		param_values[0],
		param_values[1],
		param_values[2],
		param_values[3],
		param_values[4]);
}

double AM::Eval(std::vector<int> values) const noexcept
{
	return Eval(values[0], values[1], values[2]);
}



AAMSSE::AAMSSE(MSSE score_estimator, AM accuracy_estimator)
	: score_estimator(std::move(score_estimator)), accuracy_estimator(std::move(accuracy_estimator))
{}

AAMSSE::AAMSSE(int stage_size, std::vector<BitBoard> pattern)
	: AAMSSE({ stage_size, pattern }, {})
{}

int AAMSSE::Stages() const noexcept
{
	return score_estimator.Stages();
}

int AAMSSE::StageSize() const noexcept
{
	return score_estimator.StageSize();
}

std::vector<BitBoard> AAMSSE::Pattern() const noexcept
{
	return score_estimator.Pattern();
}

float AAMSSE::Score(Position pos) const noexcept
{
	return score_estimator.Eval(pos);
}

std::vector<MaskAndValue> AAMSSE::DetailedScore(Position pos) const noexcept
{
	return score_estimator.DetailedEval(pos);
}

float AAMSSE::Accuracy(int D, int d, int E) const noexcept
{
	return accuracy_estimator.Eval(D, d, E);
}

std::size_t AAMSSE::WeightsSize() const
{
	return score_estimator.WeightsSize();
}

std::vector<float> AAMSSE::Weights() const
{
	return score_estimator.Weights();
}

std::vector<float> AAMSSE::Weights(int stage) const
{
	return score_estimator.Weights(stage);
}

void AAMSSE::Weights(int stage, std::span<const float> weights)
{
	score_estimator.Weights(stage, weights);
}
