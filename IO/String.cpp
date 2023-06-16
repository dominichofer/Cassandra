#pragma once
#include "String.h"
#include <cassert>
#include <charconv>
#include <format>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <limits>

int DepthFromString(std::string_view sv)
{
	int depth;
	auto result = std::from_chars(sv.data(), sv.data() + sv.size(), depth);
	if (result.ec == std::errc::invalid_argument)
		throw std::runtime_error("Coult not convert to int");
	return depth;
}

float ConfidenceLevelFromString(std::string_view sv)
{
	std::size_t index = sv.find('@');
	if (index == std::string_view::npos)
		return std::numeric_limits<float>::infinity();

	float confidence_level;
	auto result = std::from_chars(sv.data() + index + 1, sv.data() + sv.size(), confidence_level);
	if (result.ec == std::errc::invalid_argument)
		throw std::runtime_error("Coult not convert to int");
	return confidence_level;
}

bool IsField(std::string_view str)
{
	std::regex pattern("[A-H][1-8]|PS");
	return std::regex_match(str.begin(), str.end(), pattern);
}

Field FieldFromString(std::string_view str)
{
	static const std::unordered_map<std::string_view, Field> field_map = {
		   {"H8", Field::H8}, {"G8", Field::G8}, {"F8", Field::F8}, {"E8", Field::E8}, {"D8", Field::D8}, {"C8", Field::C8}, {"B8", Field::B8}, {"A8", Field::A8},
		   {"H7", Field::H7}, {"G7", Field::G7}, {"F7", Field::F7}, {"E7", Field::E7}, {"D7", Field::D7}, {"C7", Field::C7}, {"B7", Field::B7}, {"A7", Field::A7},
		   {"H6", Field::H6}, {"G6", Field::G6}, {"F6", Field::F6}, {"E6", Field::E6}, {"D6", Field::D6}, {"C6", Field::C6}, {"B6", Field::B6}, {"A6", Field::A6},
		   {"H5", Field::H5}, {"G5", Field::G5}, {"F5", Field::F5}, {"E5", Field::E5}, {"D5", Field::D5}, {"C5", Field::C5}, {"B5", Field::B5}, {"A5", Field::A5},
		   {"H4", Field::H4}, {"G4", Field::G4}, {"F4", Field::F4}, {"E4", Field::E4}, {"D4", Field::D4}, {"C4", Field::C4}, {"B4", Field::B4}, {"A4", Field::A4},
		   {"H3", Field::H3}, {"G3", Field::G3}, {"F3", Field::F3}, {"E3", Field::E3}, {"D3", Field::D3}, {"C3", Field::C3}, {"B3", Field::B3}, {"A3", Field::A3},
		   {"H2", Field::H2}, {"G2", Field::G2}, {"F2", Field::F2}, {"E2", Field::E2}, {"D2", Field::D2}, {"C2", Field::C2}, {"B2", Field::B2}, {"A2", Field::A2},
		   {"H1", Field::H1}, {"G1", Field::G1}, {"F1", Field::F1}, {"E1", Field::E1}, {"D1", Field::D1}, {"C1", Field::C1}, {"B1", Field::B1}, {"A1", Field::A1},
		   {"PS", Field::PS}
	};
	auto it = field_map.find(str);
	if (it != field_map.end())
		return it->second;
	else
		throw std::runtime_error("Invalid field format");
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
	if (int value; std::from_chars(str.data() + 1, str.data() + str.size(), value).ec == std::errc{})
		return ((str[0] == '-') ? -value : value) / 2;
	else
		throw std::runtime_error("Invalid score format");
}

bool IsPosition(std::string_view str)
{
	std::regex pattern("[XO-]{64} [XO]");
	return std::regex_match(str.begin(), str.end(), pattern);
}

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

bool IsGame(std::string_view str)
{
	std::regex pattern("[XO-]{64} [XO]( [A-H][1-8])*");
	return std::regex_match(str.begin(), str.end(), pattern);
}

std::string to_string(const Game& game)
{
	std::string str = to_string(game.StartPosition());
	for (Field move : game.Moves())
		str += ' ' + to_string(move);
	return str;
}

Game GameFromString(std::string_view str)
{
	Position pos = PositionFromString(str);

	std::vector<Field> moves;
	for (int i = 67; i < str.length(); i += 3)
		moves.push_back(FieldFromString(str.substr(i, 2)));

	return { pos, std::move(moves) };
}

bool IsGameScore(std::string_view str)
{
	std::regex pattern("[XO-]{64} [XO]( [A-H][1-8])*( [+-][0-9][0-9])*");
	return std::regex_match(str.begin(), str.end(), pattern);
}

std::string to_string(const GameScore& gs)
{
	std::string str = to_string(gs.game);
	for (int score : gs.scores)
		str += " " + ScoreToString(score);
	return str;
}

GameScore GameScoreFromString(std::string_view str)
{
	Position pos = PositionFromString(str);

	std::vector<Field> moves;
	std::vector<int> scores;
	int i = 67;
	while(i < str.length())
		if (str[i] == '+' or str[i] == '-')
		{
			scores.push_back(ScoreFromString(str.substr(i, 3)));
			i += 4;
		}
		else
		{
			moves.push_back(FieldFromString(str.substr(i, 2)));
			i += 3;
		}

	return GameScore(Game(pos, std::move(moves)), std::move(scores));
}

std::string RichTimeFormat(std::chrono::duration<double> duration)
{
	int hours = std::chrono::duration_cast<std::chrono::hours>(duration).count();
	int minutes = std::chrono::duration_cast<std::chrono::minutes>(duration).count() % 60;
	int seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count() % 60;
	int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;

	std::string hour_str = "";
	if (hours > 0)
		hour_str = std::to_string(hours) + ":";

	std::string minute_str  = "";
	if (hours > 0)
		minute_str = std::format("{:02d}:", minutes);
	else if (minutes > 0)
		minute_str = std::format("{:d}:", minutes);

	std::string second_str = "";
	if (hours > 0 || minutes > 0)
		second_str = std::format("{:02d}.", seconds);
	else
		second_str = std::format("{:d}.", seconds);

	std::string millisecond_str = std::format("{:03d}", milliseconds);

	return hour_str + minute_str + second_str + millisecond_str;
}
