#pragma once
#include "Core/Core.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

std::vector<Position> LoadPosFile(const std::string& filename);
std::vector<PosScore> LoadPosScoreFile(const std::string& filename);
std::vector<Game> LoadGameFile(const std::string& filename);
std::vector<GameScore> LoadGameScoreFile(const std::string& filename);

// TODO: Make this a range!
template <typename Begin, typename End>
void SaveFile(const std::string& filename, Begin begin, End end)
{
	using std::to_string;
	std::ofstream file(filename);
	for (; begin != end; ++begin)
		file << to_string(*begin) << '\n';
}
