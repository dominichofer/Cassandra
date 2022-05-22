#include "PositionGenerator.h"
#include "Bit.h"
#include "Children.h"
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
	start_pick = std::uniform_int_distribution<std::size_t>(0, this->start.size() - 1);

	if (std::ranges::any_of(this->start, [empty_count](const Position& pos) { return pos.EmptyCount() < empty_count; }))
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
		set.insert(g());
	return set;
}
std::set<Position> PosGen::generate_n_unique(int count, PositionGenerator&& g)
{
	return generate_n_unique(count, g);
}

std::set<Position> PosGen::generate_n_unique(std::execution::parallel_policy, int count, PositionGenerator& g)
{
	std::set<Position> set;
	while (set.size() < count)
	{
		std::size_t remaining = count - set.size();
		#pragma omp parallel
		{
			std::set<Position> local_set;
			#pragma omp for nowait
			for (int64_t i = 0; i < remaining; i++)
				local_set.insert(g());
			#pragma omp critical
			set.merge(local_set);
		}
	}
	return set;
}

std::set<Position> PosGen::generate_n_unique(std::execution::parallel_policy p, int count, PositionGenerator&& g) { return generate_n_unique(p, count, g); }

std::vector<Position> AllUnique(Position start, int empty_count)
{
	std::set<Position> set;
	for (Position pos : Children(start, empty_count))
		set.insert(FlipToUnique(pos));
	return { set.begin(), set.end() };
}
