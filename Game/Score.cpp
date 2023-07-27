#include "Score.h"
#include <charconv>
#include <stdexcept>
#include <format>
#include <regex>

int EndScore(const Position& pos) noexcept
{
	int P = std::popcount(pos.Player());
	int O = std::popcount(pos.Opponent());
	if (P > O)
		return max_score - O;
	if (P < O)
		return P - max_score;
	return 0;
}

std::string DepthClToString(int depth, float confidence_level)
{
	if (confidence_level == inf)
		return std::format("{:2}", depth);
	return std::format("{:2}@{:3.1f}", depth, confidence_level);
}

std::tuple<int, float> DepthClFromString(std::string_view sv)
{
	int depth;
	auto result = std::from_chars(sv.data(), sv.data() + sv.size(), depth);
	if (result.ec == std::errc::invalid_argument)
		throw std::runtime_error("Invalid depth format");

	std::size_t index = sv.find('@');
	if (index == std::string_view::npos)
		return std::make_tuple(depth, inf);

	float confidence_level;
	result = std::from_chars(sv.data() + index + 1, sv.data() + sv.size(), confidence_level);
	if (result.ec == std::errc::invalid_argument)
		throw std::runtime_error("Invalid confidence level format");
	return std::make_tuple(depth, confidence_level);
}

bool IsScore(std::string_view str)
{
	std::regex pattern("[+-]\\d{2}");
	return std::regex_match(str.begin(), str.end(), pattern);
}

std::string ScoreToString(int value)
{
	return std::format("{:+03}", value * 2);
}

int ScoreFromString(std::string_view str)
{
	int value;
	auto result = std::from_chars(str.data() + 1, str.data() + str.size(), value);
	if (result.ec == std::errc::invalid_argument)
		throw std::runtime_error("Invalid score format");
	return ((str[0] == '-') ? -value : value) / 2;
}