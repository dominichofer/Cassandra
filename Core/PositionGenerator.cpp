#include "PositionGenerator.h"
#include "Bit.h"
#include <set>

Position PosGen::Random::operator()()
{
	// Each field has a:
	//  25% chance to belong to player,
	//  25% chance to belong to opponent,
	//  50% chance to be empty.

	auto rnd = [this]() { return std::uniform_int_distribution<uint64_t>(0, 0xFFFFFFFFFFFFFFFFULL)(rnd_engine); };
	BitBoard a = rnd();
	BitBoard b = rnd();
	return { a & ~b, b & ~a };
}

std::vector<Position> PosGen::Random::operator()(int num)
{
	std::set<Position> pos;
	while (pos.size() < num)
		pos.insert((*this)());
	return { pos.begin(), pos.end() };
}

Position PosGen::RandomWithEmptyCount::operator()()
{
	auto dichotron = [this]() { return std::uniform_int_distribution<int>(0, 1)(rnd_engine) == 0; };

	BitBoard P = 0;
	BitBoard O = 0;
	for (int e = 64; e > empty_count; e--)
	{
		auto rnd = std::uniform_int_distribution<int>(0, e - 1)(rnd_engine);
		auto bit = BitBoard(PDep(1ULL << rnd, Position(P, O).Empties()));

		if (dichotron())
			P |= bit;
		else
			O |= bit;
	}
	return { P, O };
}

std::vector<Position> PosGen::RandomWithEmptyCount::operator()(int num)
{
	std::set<Position> pos;
	while (pos.size() < num)
		pos.insert((*this)());
	return { pos.begin(), pos.end() };
}

PosGen::Played::Played(Player& first, Player& second, int empty_count, Position start)
	: first(first), second(second), empty_count(empty_count), start(start)
{
	if (start.EmptyCount() < empty_count)
		throw;
}

Position PosGen::Played::operator()()
{
	Position pos = start;
	if (pos.EmptyCount() == empty_count)
		return pos;

	while (true)
	{
		Position old = pos;

		pos = first.Play(pos);
		if (pos.EmptyCount() == empty_count)
			return pos;

		pos = second.Play(pos);
		if (pos.EmptyCount() == empty_count)
			return pos;

		if (old == pos) // both players passed
			pos = start; // restart
	}
}

std::vector<Position> PosGen::Played::operator()(int num)
{
	std::set<Position> pos;
	while (pos.size() < num)
		pos.insert((*this)());
	return { pos.begin(), pos.end() };
}

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

		const auto move = stack.back().moves.ExtractFirst();
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
			const auto passed = PlayPass(pos);
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
