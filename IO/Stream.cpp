#include "Stream.h"

void Serialize(const std::string& str, std::ostream& stream)
{
	Serialize(str.size(), stream);
	stream.write(str.data(), str.size());
}

template <>
std::string Deserialize<std::string>(std::istream& stream)
{
	auto size = Deserialize<std::size_t>(stream);
	std::string str;
	str.resize(size);
	stream.read(str.data(), size);
	return str;
}
