#pragma once
#include "Board/Board.h"
#include "Game/Game.h"
#include <fstream>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

std::vector<Position> LoadPosFile(const std::string& filename);
std::vector<PosScore> LoadPosScoreFile(const std::string& filename);
std::vector<Game> LoadGameFile(const std::string& filename);
std::vector<GameScore> LoadGameScoreFile(const std::string& filename);

template <std::ranges::range Range>
void SaveFile(const std::string& filename, Range&& r)
{
	using std::to_string;
	std::ofstream file(filename);
	for (auto it = r.begin(); it != r.end(); ++it)
		file << to_string(*it) << '\n';
}
