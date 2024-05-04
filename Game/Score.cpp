#include "Score.h"
#include <charconv>
#include <stdexcept>
#include <format>

Score Score::FromString(std::string_view sv)
{
	int value;
	auto result = std::from_chars(sv.data() + 1, sv.data() + sv.size(), value);
	if (result.ec == std::errc::invalid_argument)
		throw std::runtime_error("Invalid score format");
	if (sv[0] == '-')
		value = -value;
	return Score{ value / 2 };
}

std::string to_string(Score score)
{
	return std::format("{:+03}", score * 2);
}

Score EndScore(const Position& pos) noexcept
{
	int P = std::popcount(pos.Player());
	int O = std::popcount(pos.Opponent());
	if (P > O)
		return max_score - O;
	if (P < O)
		return P - max_score;
	return 0;
}

Score StabilityBasedMaxScore(const Position& pos)
{
	return max_score - std::popcount(StableStonesOpponent(pos));
}