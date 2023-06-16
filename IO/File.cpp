#include "File.h"
#include "String.h"
#include <fstream>
#include <iostream>

std::vector<Position> LoadPosFile(const std::string& filename)
{
	std::ifstream file(filename);
	std::string line;
	std::vector<Position> vec;
	while (std::getline(file, line))
		vec.push_back(PositionFromString(line));
	return vec;
}

std::vector<PosScore> LoadPosScoreFile(const std::string& filename)
{
	std::ifstream file(filename);
	std::string line;
	std::vector<PosScore> vec;
	while (std::getline(file, line))
		vec.push_back(PosScoreFromString(line));
	return vec;
}

std::vector<Game> LoadGameFile(const std::string& filename)
{
	std::ifstream file(filename);
	std::string line;
	std::vector<Game> vec;
	while (std::getline(file, line))
		vec.push_back(GameFromString(line));
	return vec;
}

std::vector<GameScore> LoadGameScoreFile(const std::string& filename)
{
	std::ifstream file(filename);
	std::string line;
	std::vector<GameScore> vec;
	while (std::getline(file, line))
		vec.push_back(GameScoreFromString(line));
	return vec;
}
