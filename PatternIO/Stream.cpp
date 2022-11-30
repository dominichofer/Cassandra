#include "Stream.h"
#include "CoreIO/CoreIO.h"

void Serialize(const GLEM& model, std::ostream& stream)
{
	Serialize(model.Pattern(), stream);
	Serialize(model.Weights(), stream);
}

template <>
GLEM Deserialize<GLEM>(std::istream& stream)
{
	auto pattern = Deserialize<std::vector<BitBoard>>(stream);
	auto weights = Deserialize<std::vector<float>>(stream);
	return { std::move(pattern), std::move(weights) };
}

void Serialize(const AM& model, std::ostream& stream)
{
	Serialize(model.param_values, stream);
}

template <>
AM Deserialize<AM>(std::istream& stream)
{
	auto param_values = Deserialize<std::vector<double>>(stream);
	return { std::move(param_values) };
}

void Serialize(const AAGLEM& model, std::ostream& stream)
{
	Serialize(model.Pattern(), stream);
	Serialize(model.BlockSize(), stream);
	Serialize(model.Weights(), stream);
	Serialize(model.AccuracyParameters(), stream);
}

template <>
AAGLEM Deserialize<AAGLEM>(std::istream& stream)
{
	auto pattern = Deserialize<std::vector<BitBoard>>(stream);
	auto block_size = Deserialize<int>(stream);
	auto weights = Deserialize<std::vector<float>>(stream);
	auto accuracy_parameters = Deserialize<std::vector<double>>(stream);
	return AAGLEM{
		std::move(pattern),
		std::move(block_size),
		std::move(weights),
		std::move(accuracy_parameters)
	};
}