#pragma once
#include "Position.h"
#include "Moves.h"
#include "Player.h"
#include "Machine.h"

#include <random>
#include <stack>
#include <iterator>

class PositionGenerator
{
public:
	PositionGenerator(uint64_t seed = std::random_device{}()) : rnd_engine(seed) {}

	Position Random();
	Position Random(uint64_t empty_count);

	//Position RandomlyPlayed(Position start_pos = Position::Start());
	static Position Played(Player&, std::size_t empty_count, Position start = Position::Start());

	//std::unordered_set<Position> RandomlyPlayed(std::size_t count,                       Position start_pos = Position::Start());
	//std::vector<Position> Played(Player&, std::size_t count, std::size_t empty_count, Position start = Position::Start());
	//std::unordered_set<Position> RandomlyPlayed(std::execution::sequenced_policy&&, std::size_t count,                       Position start_pos = Position::Start());
	//std::unordered_set<Position> RandomlyPlayed(std::execution::sequenced_policy&&, std::size_t count, uint64_t empty_count, Position start_pos = Position::Start());
	//std::unordered_set<Position> RandomlyPlayed(std::execution::parallel_policy&& , std::size_t count,                       Position start_pos = Position::Start());
	//std::unordered_set<Position> RandomlyPlayed(std::execution::parallel_policy&& , std::size_t count, uint64_t empty_count, Position start_pos = Position::Start());
	
	template <class Inserter>
	static void All(Inserter it, std::size_t plies, std::size_t plies_per_pass, Position start = Position::Start())
	{
		all(start, plies, plies_per_pass, it);
	}

	template <class Inserter>
	static void All(Inserter it, std::size_t empty_count, Position start = Position::Start())
	{
		if (start.EmptyCount() >= empty_count)
			all(start, start.EmptyCount() - empty_count, 0, it);
	}

private:
	std::mt19937_64 rnd_engine;
	
	Board RandomMiddle();

	template <class Inserter>
	static void add(Position pos, const std::size_t plies, const std::size_t plies_per_pass, Inserter& inserter)
	{
		if (plies == 0)
		{
			inserter = pos;
			return;
		}

		Moves moves = PossibleMoves(pos);
		if (moves.empty())
		{
			pos = PlayPass(pos);
			if (!PossibleMoves(pos).empty())
				add(pos, plies - plies_per_pass, plies_per_pass, inserter);
			return;
		}

		for (const auto move : moves)
			add(Play(pos, move), plies - 1, plies_per_pass, inserter);
	}

	template <class Inserter>
	static void all(const Position pos, const std::size_t plies, const std::size_t plies_per_pass, Inserter& inserter)
	{
		if (plies)
			add(pos, plies, plies_per_pass, inserter);
		else
			inserter = pos;
	}
};