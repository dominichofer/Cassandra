#pragma once
#include "Position.h"
#include "Moves.h"
#include "Player.h"
#include <execution>
#include <random>
#include <iterator>
#include <vector>
#include <set>

// Position Generators
namespace PosGen
{
	// Abstract
	class PositionGenerator
	{
	public:
		virtual Position operator()() noexcept = 0;
	};

	class Random final : public PositionGenerator
	{
		std::mt19937_64 rnd_engine;
		std::uniform_int_distribution<uint64> dist{ 0, -1_i64 };
	public:
		Random(unsigned int seed = std::random_device{}()) : rnd_engine(seed) {}

		Position operator()() noexcept override;
	};

	// Generator of random Position with given empty count.
	class RandomWithEmptyCount final : public PositionGenerator
	{
		const int empty_count;
		std::mt19937_64 rnd_engine;
		std::uniform_int_distribution<int> boolean{0, 1};
	public:
		RandomWithEmptyCount(int empty_count, unsigned int seed = std::random_device{}()) : empty_count(empty_count), rnd_engine(seed) {}
		
		Position operator()() noexcept override;
	};
	
	// Generator of played Position with given empty count.
	class Played : public PositionGenerator
	{
	protected:
		Player &first, &second;
		const int empty_count;
		std::vector<Position> start;
		std::mt19937_64 rnd_engine{ std::random_device{}() };
		std::uniform_int_distribution<std::size_t> start_pick;
	public:
		Played(Player& first, Player& second, int empty_count, std::vector<Position> start) noexcept(false);
		Played(Player& first, Player& second, int empty_count, Position start = Position::Start()) noexcept(false) : Played(first, second, empty_count, std::vector{ start }) {}

		Position operator()() noexcept override;
	};

	// Generator of randomly played Position with given empty count.
	class RandomlyPlayed final : public Played
	{
		RandomPlayer first, second;
	public:
		RandomlyPlayed(unsigned int seed1, unsigned int seed2, int empty_count, Position start = Position::Start()) noexcept
			: first(seed1), second(seed2), Played(first, second, empty_count, start)
		{}
		RandomlyPlayed(int empty_count, Position start = Position::Start()) noexcept
			: Played(first, second, empty_count, start) {}
	};

	std::set<Position> generate_n_unique(int count, PositionGenerator&);
	std::set<Position> generate_n_unique(int count, PositionGenerator&&);
	std::set<Position> generate_n_unique(std::execution::parallel_policy, int count, PositionGenerator&);
	std::set<Position> generate_n_unique(std::execution::parallel_policy, int count, PositionGenerator&&);
}


std::vector<Position> AllUnique(Position, int empty_count);