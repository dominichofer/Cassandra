#include "PositionScore.h"
#include <regex>

bool IsPositionScore(std::string_view str)
{
	std::regex pattern("[XO-]{64} [XO] % [+-][0-9][0-9]");
	return std::regex_match(str.begin(), str.end(), pattern);
}

std::string to_string(const PosScore& ps)
{
	return to_string(ps.pos) + " % " + ScoreToString(ps.score);
}

PosScore PosScoreFromString(std::string_view str)
{
	Position pos = PositionFromString(str);
	int score = ScoreFromString(str.substr(69));
	return { pos, score };
}
