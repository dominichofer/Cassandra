#include "Stream.h"

void Serialize(const Confidence& c, std::ostream& stream)
{
	Serialize(c.sigmas(), stream);
}

template <>
Confidence Deserialize<Confidence>(std::istream& stream)
{
	return Confidence{ Deserialize<float>(stream) };
}

void Serialize(const Intensity& i, std::ostream& stream)
{
	Serialize(i.depth, stream);
	Serialize(i.certainty, stream);
}

template <>
Intensity Deserialize<Intensity>(std::istream& stream)
{
	auto depth = Deserialize<decltype(Intensity::depth)>(stream);
	auto certainty = Deserialize<decltype(Intensity::certainty)>(stream);
	return { depth, certainty };
}
