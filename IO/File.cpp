#include "File.h"
#include "String.h"
#include <fstream>
#include <iostream>

std::vector<Position> LoadPositionFile(const std::string& filename)
{
	std::ifstream file(filename);
	std::string line;
	std::vector<Position> vec;
	while (std::getline(file, line))
		vec.push_back(Position::FromString(line));
	return vec;
}

std::vector<ScoredPosition> LoadScoredPositionFile(const std::string& filename)
{
	std::ifstream file(filename);
	std::string line;
	std::vector<ScoredPosition> vec;
	while (std::getline(file, line))
		vec.push_back(ScoredPosition::FromString(line));
	return vec;
}

std::vector<Game> LoadGameFile(const std::string& filename)
{
	std::ifstream file(filename);
	std::string line;
	std::vector<Game> vec;
	while (std::getline(file, line))
		vec.push_back(Game::FromString(line));
	return vec;
}

std::vector<ScoredGame> LoadScoredGameFile(const std::string& filename)
{
	std::ifstream file(filename);
	std::string line;
	std::vector<ScoredGame> vec;
	while (std::getline(file, line))
		vec.push_back(ScoredGame::FromString(line));
	return vec;
}
