#pragma once
#include "Player.h"
#include <random>
#include <vector>

class RandomPositionGenerator
{
	std::mt19937_64 rnd_engine;
	std::uniform_int_distribution<uint64_t> dist{ 0, 0xFFFFFFFFFFFFFFFFULL };
public:
	RandomPositionGenerator(unsigned int seed = std::random_device{}()) : rnd_engine(seed) {}

	Position operator()() noexcept;
};

Position RandomPosition(unsigned int seed = std::random_device{}());
