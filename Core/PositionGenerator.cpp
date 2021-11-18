#include "PositionGenerator.h"
#include "Bit.h"
#include <set>

using namespace PosGen;

Position Random::operator()() noexcept
{
	// Each field has a:
	//  25% chance to belong to player,
	//  25% chance to belong to opponent,
	//  50% chance to be empty.

	BitBoard a{ dist(rnd_engine) };
	BitBoard b{ dist(rnd_engine) };
	return { a & ~b, b & ~a };
}

Position RandomWithEmptyCount::operator()() noexcept
{
	Position pos;
	while (pos.EmptyCount() > empty_count)
	{
		int rnd = std::uniform_int_distribution<int>(0, pos.EmptyCount() - 1)(rnd_engine);

		// deposit bit on an empty field
		auto bit = BitBoard(PDep(1ULL << rnd, pos.Empties()));

		if (boolean(rnd_engine))
			pos = Position{ pos.Player() | bit, pos.Opponent() };
		else
			pos = Position{ pos.Player(), pos.Opponent() | bit };
	}
	return pos;
}

Played::Played(Player& first, Player& second, int empty_count, std::vector<Position> start) noexcept(false)
	: first(first), second(second), empty_count(empty_count), start(std::move(start))
{
	start_pick = std::uniform_int_distribution<int>(0, this->start.size() - 1);

	bool any_pos_not_enough_empty_count = std::any_of(this->start.begin(), this->start.end(),
		[empty_count](const Position& pos) { return pos.EmptyCount() < empty_count; });
	if (any_pos_not_enough_empty_count)
		throw;
}

Position Played::operator()() noexcept
{
start:
	Position pos = start[start_pick(rnd_engine)];
	if (pos.EmptyCount() == empty_count)
		return pos;
play:
	Position old = pos;

	pos = first.Play(pos);
	if (pos.EmptyCount() == empty_count)
		return pos;

	pos = second.Play(pos);
	if (pos.EmptyCount() == empty_count)
		return pos;

	if (old == pos) // both players passed
		goto start;
	goto play;
}

std::set<Position> PosGen::generate_n_unique(int count, PositionGenerator& g)
{
	std::set<Position> set;
	while (set.size() < count)
	{
		#pragma omp parallel
		{
			std::set<Position> local_set;
			#pragma omp for nowait
			for (int i = set.size(); i < count; i++)
				local_set.insert(g());
			#pragma omp critical
			set.merge(std::move(local_set));
		}
	}
	return set;
}

std::set<Position> PosGen::generate_n_unique(int count, PositionGenerator&& g) { return generate_n_unique(count, g); }

ChildrenGenerator::Iterator::Iterator(const Position& start, int plies, bool pass_is_a_ply) noexcept
	: plies(plies), pass_is_a_ply(pass_is_a_ply)
{
	stack.reserve(plies);

	if (stack.size() == plies) {
		stack.emplace_back(start, Moves{});
		return;
	}

	const auto moves = PossibleMoves(start);
	if (moves)
		stack.emplace_back(start, moves);
	else
	{
		const auto passed = PlayPass(start);
		const auto passed_moves = PossibleMoves(passed);
		if (passed_moves)
		{
			if (pass_is_a_ply)
			{
				stack.emplace_back(start, Moves{});
				if (stack.size() == plies + 1)
					return;
			}
			stack.emplace_back(passed, passed_moves);
		}
	}
	if (stack.size() == plies + 1)
		return;
	++(*this);
}

ChildrenGenerator::Iterator& ChildrenGenerator::Iterator::operator++()
{
	while (!stack.empty())
	{
		if (stack.size() == plies + 1 || !stack.back().moves) {
			stack.pop_back();
			continue;
		}

		const auto move = stack.back().moves.front();
		stack.back().moves.pop_front();
		const auto pos = Play(stack.back().pos, move);

		if (stack.size() == plies) {
			stack.emplace_back(pos, Moves{});
			return *this;
		}

		const auto moves = PossibleMoves(pos);
		if (moves)
			stack.emplace_back(pos, moves);
		else
		{
			auto passed = PlayPass(pos);
			const auto passed_moves = PossibleMoves(passed);
			if (passed_moves)
			{
				if (pass_is_a_ply)
				{
					stack.emplace_back(pos, Moves{});
					if (stack.size() == plies + 1)
						return *this;
				}
				stack.emplace_back(passed, passed_moves);
			}
		}
		if (stack.size() == plies + 1)
			return *this;
	}
	return *this;
}

ChildrenGenerator Children(Position start, int plies, bool pass_is_a_ply)
{
	assert(plies > 0);
	return {start, plies, pass_is_a_ply};
}

ChildrenGenerator Children(Position start, int empty_count)
{
	assert(start.EmptyCount() > empty_count);
	return {start, start.EmptyCount() - empty_count, false};
}

std::vector<Position> AllUnique(Position start, int empty_count)
{
	std::set<Position> set;
	for (Position pos : Children(start, empty_count))
		set.insert(FlipToUnique(pos));
	return { set.begin(), set.end() };
}
