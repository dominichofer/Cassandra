#pragma once
#include "Player.h"
#include <random>
#include <vector>

Position RandomPosition(unsigned int seed = std::random_device{}());
std::vector<Position> RandomPositions(int count, unsigned int seed = std::random_device{}());
class RandomPositionGenerator;

Position RandomPositionWithEmptyCount(int empty_count, unsigned int seed = std::random_device{}()) noexcept(false);
std::vector<Position> RandomPositionsWithEmptyCount(int count, int empty_count, unsigned int seed = std::random_device{}()) noexcept(false);
class RandomPositionGeneratorWithEmptyCount;

Position RandomlyPlayedPositionWithEmptyCount(int empty_count, Position start = Position::Start(), unsigned int seed = std::random_device{}()) noexcept(false);
Position RandomlyPlayedPositionWithEmptyCount(int empty_count, unsigned int seed) noexcept(false);
std::vector<Position> RandomlyPlayedPositionsWithEmptyCount(int count, int empty_count, Position start = Position::Start(), unsigned int seed = std::random_device{}()) noexcept(false);
std::vector<Position> RandomlyPlayedPositionsWithEmptyCount(int count, int empty_count, unsigned int seed) noexcept(false);



class RandomPositionGenerator
{
	std::mt19937_64 rnd_engine;
	std::uniform_int_distribution<uint64> dist{ 0, -1_i64 };
public:
	RandomPositionGenerator(unsigned int seed = std::random_device{}()) : rnd_engine(seed) {}

	Position operator()() noexcept;
};

class RandomPositionGeneratorWithEmptyCount
{
	const int empty_count;
	std::mt19937_64 rnd_engine;
	std::uniform_int_distribution<int> boolean{ 0, 1 };
public:
	RandomPositionGeneratorWithEmptyCount(int empty_count, unsigned int seed = std::random_device{}()) noexcept(false);

	Position operator()() noexcept;
};
