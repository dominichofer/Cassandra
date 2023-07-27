#pragma once
#include "Board/Board.h"
#include <random>
#include <mutex>

// Interface
class Player
{
public:
	virtual Field ChooseMove(const Position&) = 0;
};

class RandomPlayer final : public Player
{
	std::mutex mtx;
	std::mt19937_64 rnd_engine;
public:
	RandomPlayer(uint64_t seed = std::random_device{}()) : rnd_engine(seed) {}

	Field ChooseMove(const Position&) override;
};
