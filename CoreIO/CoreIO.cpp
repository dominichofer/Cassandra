#include "CoreIO.h"
#include <array>
#include <algorithm>
#include <stdexcept>

const std::array<std::string, 64> field_names = {
	"H8", "G8", "F8", "E8", "D8", "C8", "B8", "A8",
	"H7", "G7", "F7", "E7", "D7", "C7", "B7", "A7",
	"H6", "G6", "F6", "E6", "D6", "C6", "B6", "A6",
	"H5", "G5", "F5", "E5", "D5", "C5", "B5", "A5",
	"H4", "G4", "F4", "E4", "D4", "C4", "B4", "A4",
	"H3", "G3", "F3", "E3", "D3", "C3", "B3", "A3",
	"H2", "G2", "F2", "E2", "D2", "C2", "B2", "A2",
	"H1", "G1", "F1", "E1", "D1", "C1", "B1", "A1"
};

Field ParseField(std::string_view str)
{
	auto it = std::find(field_names.begin(), field_names.end(), str);
	auto index = std::distance(field_names.begin(), it);
	return static_cast<Field>(index);
}

Position ParsePosition_SingleLine(std::string_view str) noexcept(false)
{
	BitBoard P, O;

	for (int i = 0; i < 64; i++)
	{
		if (str[i] == 'X')
			P.Set(static_cast<Field>(63 - i));
		if (str[i] == 'O')
			O.Set(static_cast<Field>(63 - i));
	}

	if (str[65] == 'X')
		return Position{ P, O };
	if (str[65] == 'O')
		return Position{ O, P };
	throw std::runtime_error("Position string needs to end with 'X' or 'O' to denote who is to play.");
}

std::vector<Position> ParsePositionFile(const std::string& file) noexcept(false)
{
	std::vector<Position> ret;
	std::fstream stream(file, std::ios::in);
	for (std::string line; std::getline(stream, line); )
		ret.push_back(ParsePosition_SingleLine(line));
	return ret;
}

PosScore ParsePosScore_SingleLine(const std::string& str) noexcept(false)
{
	Position pos = ParsePosition_SingleLine(str);
	int score = std::stoi(str.substr(69, 3)) / 2;
	return { pos, score };
}

std::vector<PosScore> ParsePosScoreFile(const std::string& file) noexcept(false)
{
	std::vector<PosScore> ret;
	std::fstream stream(file, std::ios::in);
	for (std::string line; std::getline(stream, line); )
		ret.push_back(ParsePosScore_SingleLine(line));
	return ret;
}