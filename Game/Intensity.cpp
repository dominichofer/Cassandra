#include "Intensity.h"
#include <format>
#include <stdexcept>

Intensity Intensity::FromString(std::string_view str)
{
	int8_t depth;
	auto depth_result = std::from_chars(str.data(), str.data() + str.size(), depth);
	if (depth_result.ec == std::errc::invalid_argument)
		throw std::runtime_error("Invalid depth format");

	std::size_t at_pos = str.find('@');
	if (at_pos == std::string_view::npos)
		return { depth, inf };

	float level;
	auto level_result = std::from_chars(str.data() + at_pos + 1, str.data() + str.size(), level);
	if (level_result.ec == std::errc::invalid_argument)
		throw std::runtime_error("Invalid confidence level format");

	return { depth, ConfidenceLevel{ level } };
}

std::string Intensity::to_string() const
{
	if (level.IsInfinit())
		return std::format("{:2}", depth);
	return std::format("{:2}@{:3.1f}", depth, static_cast<float>(level));
}

std::string to_string(const Intensity& i)
{
	return i.to_string();
}
