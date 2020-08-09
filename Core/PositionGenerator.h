#pragma once
#include "Algorithm.h"
#include "Position.h"
#include "Moves.h"
#include "Player.h"

#include <random>
#include <stack>
#include <iterator>
#include <optional>
#include <stack>

// PositionGenerator
namespace PosGen
{
	// Generator of random Position.
	class Random
	{
		BitBoard mask;
		std::mt19937_64 rnd_engine;
	public:
		Random(uint64_t seed = std::random_device{}(), BitBoard exclude = {}) : mask(~exclude), rnd_engine(seed) {}

		Position operator()();
	};

	// Generator of random Position with given empty count.
	class Random_with_empty_count
	{
		const std::size_t empty_count;
		std::mt19937_64 rnd_engine;
	public:
		Random_with_empty_count(std::size_t empty_count, uint64_t seed = std::random_device{}()) : empty_count(empty_count), rnd_engine(seed) {}
		
		Position operator()();
	};

	// Generator of all Positions after the n-th ply.
	class All_after_nth_ply
	{
		struct PosMov{ Position pos; Moves moves; };
		const std::size_t plies;
		const std::size_t plies_per_pass;
		std::stack<PosMov> stack;
	public:
		All_after_nth_ply(std::size_t plies, std::size_t plies_per_pass, Position start = Position::Start());

		std::optional<Position> operator()();
	};

	// Generator of all Position with given empty count.
	class All_with_empty_count
	{
		struct PosMov{ Position pos; Moves moves; };
		const std::size_t empty_count;
		std::stack<PosMov> stack;
	public:
		All_with_empty_count(std::size_t empty_count, Position start = Position::Start());

		std::optional<Position> operator()();
	};
	
	// Generator of played Position with given empty count.
	class Played
	{
		Player &first, &second;
		const std::size_t empty_count;
		const Position start;
	public:
		Played(Player& first, Player& second, std::size_t empty_count, Position start = Position::Start());

		Position operator()();
	};
}
