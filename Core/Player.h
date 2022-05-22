#pragma once
#include "Position.h"
#include "Moves.h"
#include <random>

class Player
{
public:
	virtual Position Play(const Position&) = 0; // TODO: Remove!
	virtual Field ChooseMove(const Position&) = 0;
};

class RandomPlayer final : public Player
{
	std::mt19937_64 rnd_engine;
public:
	RandomPlayer(uint64_t seed = std::random_device{}()) : rnd_engine(seed) {}

	Position Play(const Position&) override;
	Field ChooseMove(const Position&) override;
};
