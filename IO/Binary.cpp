#include "Binary.h"

void Serialize(const ScoreEstimator& se, std::ostream& stream)
{
	Serialize(se.Pattern(), stream);
	Serialize(se.Weights(), stream);
}

template <>
ScoreEstimator Deserialize<ScoreEstimator>(std::istream& stream)
{
	auto pattern = Deserialize<std::vector<uint64_t>>(stream);
	auto weights = Deserialize<std::vector<float>>(stream);
	return ScoreEstimator(std::move(pattern), std::move(weights));
}

void Serialize(const MultiStageScoreEstimator& msse, std::ostream& stream)
{
	Serialize(msse.StageSize(), stream);
	Serialize(msse.Pattern(), stream);
	Serialize(msse.estimators, stream);
}

template <>
MultiStageScoreEstimator Deserialize<MultiStageScoreEstimator>(std::istream& stream)
{
	auto stage_size = Deserialize<std::size_t>(stream);
	auto pattern = Deserialize<std::vector<uint64_t>>(stream);
	auto msse = MultiStageScoreEstimator(stage_size, std::move(pattern));
	msse.estimators = Deserialize<std::vector<ScoreEstimator>>(stream);
	return msse;
}

void Serialize(const AccuracyModel& am, std::ostream& stream)
{
	Serialize(am.ParameterValues(), stream);
}

template <>
AccuracyModel Deserialize<AccuracyModel>(std::istream& stream)
{
	auto param_values = Deserialize<std::vector<float>>(stream);
	return AccuracyModel(std::move(param_values));
}

void Serialize(const PatternBasedEstimator& estimator, std::ostream& stream)
{
	Serialize(estimator.score, stream);
	Serialize(estimator.accuracy, stream);
}

template <>
PatternBasedEstimator Deserialize<PatternBasedEstimator>(std::istream& stream)
{
	auto score = Deserialize<MultiStageScoreEstimator>(stream);
	auto accuracy = Deserialize<AccuracyModel>(stream);
	return PatternBasedEstimator(std::move(score), std::move(accuracy));
}

void Save(const PatternBasedEstimator& estimator, const std::filesystem::path& file)
{
	Serialize(estimator, file);
}

PatternBasedEstimator LoadPatternBasedEstimator(const std::filesystem::path& file)
{
	return Deserialize<PatternBasedEstimator>(file);
}
