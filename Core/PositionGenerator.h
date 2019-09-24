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
	Position Played(Player&, std::size_t empty_count, Position start = Position::Start());

	//std::unordered_set<Position> RandomlyPlayed(std::size_t count,                       Position start_pos = Position::Start());
	//std::vector<Position> Played(Player&, std::size_t count, std::size_t empty_count, Position start = Position::Start());
	//std::unordered_set<Position> RandomlyPlayed(std::execution::sequenced_policy&&, std::size_t count,                       Position start_pos = Position::Start());
	//std::unordered_set<Position> RandomlyPlayed(std::execution::sequenced_policy&&, std::size_t count, uint64_t empty_count, Position start_pos = Position::Start());
	//std::unordered_set<Position> RandomlyPlayed(std::execution::parallel_policy&& , std::size_t count,                       Position start_pos = Position::Start());
	//std::unordered_set<Position> RandomlyPlayed(std::execution::parallel_policy&& , std::size_t count, uint64_t empty_count, Position start_pos = Position::Start());


	template <class Container>
	void All(std::insert_iterator<Container> it, std::size_t empty_count, Position start = Position::Start())
	{
		all(it, empty_count, start);
	}

	template <class Container>
	void All(std::back_insert_iterator<Container> it, std::size_t empty_count, Position start = Position::Start())
	{
		all(it, empty_count, start);
	}

	//// symmetrically identical positions are considered distinct.
	//std::vector<Position> AllUnique(std::size_t empty_count, Position start = Position::Start());

	//// symmetrically identical positions are considered identical.
	//std::vector<Position> AllSymmetricUnique(std::size_t empty_count, Position start = Position::Start());

private:
	std::mt19937_64 rnd_engine;
	
	Board RandomMiddle();

	template <class Inserter>
	void all(Inserter inserter, std::size_t empty_count, Position start)
	{
		if (empty_count >= start.EmptyCount())
			return;
		if (empty_count == start.EmptyCount())
		{
			inserter = start;
			return;
		}

		struct pos_mov { Position pos; Moves moves; };
		std::stack<pos_mov> stack;

		stack.push({ start, PossibleMoves(start) });
		while (!stack.empty())
		{
			auto& top = stack.top();
			if (top.moves.empty())
				stack.pop();
			else
			{
				Position new_pos = Play(top.pos, top.moves.Extract());
				if (new_pos.EmptyCount() == empty_count)
					inserter = new_pos;
				else
					stack.push({ new_pos, PossibleMoves(new_pos) });
			}
		}
	}
};