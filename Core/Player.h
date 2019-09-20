#pragma once
#include "Position.h"
#include "Moves.h"
#include <random>
#include <stdexcept>

class Player
{
public:
	virtual Position Play(Position) noexcept(false) = 0;
};

class no_moves_exception : public std::exception
{};

class RandomPlayer final : public Player
{
public:
	RandomPlayer(uint64_t seed = std::random_device{}()) : Player(), rnd_engine(seed) {}

	Position Play(Position) noexcept(false) final;

private:
	std::mt19937_64 rnd_engine;

	using Player::Play;
};