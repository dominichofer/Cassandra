#include "Binary.h"

void Serialize(const ScoreEstimator& se, std::ostream& stream)
{
	Serialize(se.Pattern(), stream);
	Serialize(se.Weights(), stream);
}

ScoreEstimator Deserialize_ScoreEstimator(std::istream& stream)
{
	auto pattern = Deserialize<std::vector<uint64_t>>(stream);
	auto weights = Deserialize<std::vector<float>>(stream);
	return ScoreEstimator(std::move(pattern), std::move(weights));
}

void Serialize(const MultiStageScoreEstimator& msse, std::ostream& stream)
{
	Serialize(msse.StageSize(), stream);
	Serialize(msse.Pattern(), stream);
	Serialize(msse.Weights(), stream);
}

MultiStageScoreEstimator Deserialize_MSSE(std::istream& stream)
{
	auto stage_size = Deserialize<int>(stream);
	auto pattern = Deserialize<std::vector<uint64_t>>(stream);
	auto weights = Deserialize<std::vector<float>>(stream);
	return MultiStageScoreEstimator(stage_size, std::move(pattern), std::move(weights));
}

void Serialize(const AccuracyModel& am, std::ostream& stream)
{
	Serialize(am.ParameterValues(), stream);
}

AccuracyModel Deserialize_AM(std::istream& stream)
{
	auto param_values = Deserialize<std::vector<double>>(stream);
	return AccuracyModel(std::move(param_values));
}

void Serialize(const PatternBasedEstimator& estimator, std::ostream& stream)
{
	Serialize(estimator.score, stream);
	Serialize(estimator.accuracy, stream);
}

PatternBasedEstimator Deserialize_PatternBasedEstimator(std::istream& stream)
{
	auto score = Deserialize_MSSE(stream);
	auto accuracy = Deserialize_AM(stream);
	return PatternBasedEstimator(std::move(score), std::move(accuracy));
}

void Save(const PatternBasedEstimator& estimator, const std::filesystem::path& file)
{
	std::fstream stream(file, std::ios::binary | std::ios::out);
	Serialize(estimator, stream);
}

PatternBasedEstimator LoadPatternBasedEstimator(const std::filesystem::path& file)
{
	std::fstream stream(file, std::ios::binary | std::ios::in);
	return Deserialize_PatternBasedEstimator(stream);
}
