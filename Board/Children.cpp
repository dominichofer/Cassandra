#include "Children.h"
#include "PossibleMoves.h"

using namespace children;

Iterator::Iterator(const Position& start, int plies, bool pass_is_a_ply) noexcept
	: plies(plies), pass_is_a_ply(pass_is_a_ply)
{
	pos_stack.reserve(plies);
	moves_stack.reserve(plies);

	if (pos_stack.size() == plies) {
		pos_stack.push_back(start);
		moves_stack.push_back({});
		return;
	}

	const auto moves = PossibleMoves(start);
	if (moves) {
		pos_stack.push_back(start);
		moves_stack.push_back(moves);
	}
	else
	{
		const auto passed = PlayPass(start);
		const auto passed_moves = PossibleMoves(passed);
		if (passed_moves)
		{
			if (pass_is_a_ply)
			{
				pos_stack.push_back(start);
				moves_stack.push_back({});
				if (pos_stack.size() == plies + 1)
					return;
			}
			pos_stack.push_back(passed);
			moves_stack.push_back(passed_moves);
		}
	}
	if (pos_stack.size() == plies + 1)
		return;
	++(*this);
}

Iterator& Iterator::operator++()
{
	while (not pos_stack.empty())
	{
		if (pos_stack.size() == plies + 1 || !moves_stack.back()) {
			pos_stack.pop_back();
			moves_stack.pop_back();
			continue;
		}

		const auto move = moves_stack.back().front();
		moves_stack.back().pop_front();
		const auto pos = Play(pos_stack.back(), move);

		if (pos_stack.size() == plies) {
			pos_stack.push_back(pos);
			moves_stack.push_back({});
			return *this;
		}

		const auto moves = PossibleMoves(pos);
		if (moves) {
			pos_stack.push_back(pos);
			moves_stack.push_back(moves);
		}
		else
		{
			const auto passed = PlayPass(pos);
			const auto passed_moves = PossibleMoves(passed);
			if (passed_moves)
			{
				if (pass_is_a_ply)
				{
					pos_stack.push_back(pos);
					moves_stack.push_back({});
					if (pos_stack.size() == plies + 1)
						return *this;
				}
				pos_stack.push_back(passed);
				moves_stack.push_back(passed_moves);
			}
		}
		if (pos_stack.size() == plies + 1)
			return *this;
	}
	return *this;
}

children::Generator Children(Position start, int plies, bool pass_is_a_ply)
{
	assert(plies >= 0);
	return { start, plies, pass_is_a_ply };
}

children::Generator Children(Position start, int empty_count)
{
	assert(start.EmptyCount() >= empty_count);
	return { start, start.EmptyCount() - empty_count, false };
}

std::set<Position> UniqueChildren(Position start, int empty_count)
{
	std::set<Position> ret;
	for (Position pos : Children(start, empty_count))
		ret.insert(FlippedToUnique(pos));
	return ret;
}
