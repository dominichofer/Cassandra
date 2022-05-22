#include "Stream.h"

void Serialize(const BitBoard& b, std::ostream& stream)
{
	Serialize(static_cast<uint64_t>(b), stream);
}

template <>
BitBoard Deserialize<BitBoard>(std::istream& stream)
{
	return Deserialize<uint64_t>(stream);
}

void Serialize(const Position& pos, std::ostream& stream)
{
	Serialize(pos.Player(), stream);
	Serialize(pos.Opponent(), stream);
}

template <>
Position Deserialize<Position>(std::istream& stream)
{
	auto P = Deserialize<BitBoard>(stream);
	auto O = Deserialize<BitBoard>(stream);
	return { P, O };
}
