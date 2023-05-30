#include "Stream.h"
#include "CoreIO/CoreIO.h"

void Serialize(const ScoreEstimator& estimators, std::ostream& stream)
{
	Serialize(estimators.Pattern(), stream);
	Serialize(estimators.Weights(), stream);
}

template <>
ScoreEstimator Deserialize<ScoreEstimator>(std::istream& stream)
{
	auto pattern = Deserialize<std::vector<BitBoard>>(stream);
	auto weights = Deserialize<std::vector<float>>(stream);
	return { std::move(pattern), std::move(weights) };
}

void Serialize(const MSSE& estimators, std::ostream& stream)
{
	Serialize(estimators.StageSize(), stream);
	Serialize(estimators.Pattern(), stream);
	Serialize(estimators.Weights(), stream);
}

template <>
MSSE Deserialize<MSSE>(std::istream& stream)
{
	auto stage_size = Deserialize<int>(stream);
	auto pattern = Deserialize<std::vector<BitBoard>>(stream);
	auto weights = Deserialize<std::vector<float>>(stream);
	return { stage_size, std::move(pattern), std::move(weights) };
}

void Serialize(const AM& model, std::ostream& stream)
{
	Serialize(model.ParameterValues(), stream);
}

template <>
AM Deserialize<AM>(std::istream& stream)
{
	auto param_values = Deserialize<std::vector<double>>(stream);
	return { std::move(param_values) };
}

void Serialize(const AAMSSE& model, std::ostream& stream)
{
	Serialize(model.score_estimator, stream);
	Serialize(model.accuracy_estimator, stream);
}

template <>
AAMSSE Deserialize<AAMSSE>(std::istream& stream)
{
	auto score_estimator = Deserialize<MSSE>(stream);
	auto accuracy_estimator = Deserialize<AM>(stream);
	return AAMSSE{ std::move(score_estimator), std::move(accuracy_estimator) };
}
